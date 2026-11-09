"""Maximize C4-free edges in Q7 via complementary deletion + local search."""
import time
import random
import numpy as np
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    d = 7
    n = 2 ** d
    num_edges = d * 2 ** (d - 1)  # 448

    # Build edge data
    all_edges = []
    edge_index = {}
    edge_dim = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx
                edge_dim.append(i)

    # Build face-to-edge mapping
    # Each face is determined by (i, j, base) where i < j are the two dimensions
    # and base has 0 bits at positions i and j
    faces = []
    face_edges = []
    edge_to_faces = [[] for _ in range(num_edges)]

    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            mask_ij = mask_i | mask_j
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_ij

                    e0 = edge_index.get((min(v0, v1), max(v0, v1)))
                    e1 = edge_index.get((min(v1, v3), max(v1, v3)))
                    e2 = edge_index.get((min(v0, v2), max(v0, v2)))
                    e3 = edge_index.get((min(v2, v3), max(v2, v3)))

                    if None not in (e0, e1, e2, e3):
                        face_idx = len(faces)
                        faces.append((i, j, base))
                        fe = [e0, e1, e2, e3]
                        face_edges.append(fe)
                        for e in fe:
                            edge_to_faces[e].append(face_idx)

    num_faces = len(faces)
    assert num_faces == 672

    # Precompute edge_to_faces as arrays
    e2f = [np.array(fl, dtype=np.int32) for fl in edge_to_faces]

    def is_valid_add(fc, e):
        """Check if edge e can be added given face counts fc."""
        f = e2f[e]
        if len(f) == 0:
            return True
        return np.max(fc[f]) < 3

    def add_edge(chosen, fc, e):
        chosen[e] = True
        for f in e2f[e]:
            fc[f] += 1

    def remove_edge(chosen, fc, e):
        chosen[e] = False
        for f in e2f[e]:
            fc[f] -= 1

    def greedy_add(chosen, fc, order=None):
        """Greedily add all valid edges in given order."""
        if order is None:
            order = range(num_edges)
        for e in order:
            if not chosen[e] and is_valid_add(fc, e):
                add_edge(chosen, fc, e)

    def complementary_deletion(seed):
        """Start with all edges, remove edges to break all C4s."""
        rng = random.Random(seed)
        chosen = np.ones(num_edges, dtype=bool)
        fc = np.zeros(num_faces, dtype=np.int32)

        # Count how many edges in each face
        for f_idx in range(num_faces):
            for e in face_edges[f_idx]:
                fc[f_idx] += 1

        # Iteratively remove edges from faces with 4 edges
        # Priority: remove edge that's in the most full faces
        for iteration in range(200):
            if time.time() > deadline - 10:
                break

            # Find faces with 4 edges
            full_faces = [f for f in range(num_faces) if fc[f] == 4]
            if not full_faces:
                break

            rng.shuffle(full_faces)

            # For each full face, pick an edge to remove
            # Strategy: remove edge that appears in most full faces
            face_counts_of_edges = {}
            for f in full_faces:
                for e in face_edges[f]:
                    face_counts_of_edges[e] = face_counts_of_edges.get(e, 0) + 1

            # Pick the edge with highest count
            best_e = max(face_counts_of_edges, key=face_counts_of_edges.get)
            remove_edge(chosen, fc, best_e)

        # Now try to add back edges
        order = list(range(num_edges))
        rng.shuffle(order)
        greedy_add(chosen, fc, order)

        return int(chosen.sum()), chosen, fc

    def local_search(chosen, fc, max_rounds=200, seed=0):
        """Local search with 1-for-1 and 2-for-1 swaps."""
        rng = random.Random(seed)
        current_count = int(chosen.sum())
        no_improve = 0

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break

            chosen_edges = [e for e in range(num_edges) if chosen[e]]
            if not chosen_edges:
                break
            rng.shuffle(chosen_edges)

            found = False

            # Try removing one edge and greedily re-adding
            for e_rm in chosen_edges[:10]:
                c2 = chosen.copy()
                fc2 = fc.copy()
                remove_edge(c2, fc2, e_rm)

                order = list(range(num_edges))
                rng.shuffle(order)
                greedy_add(c2, fc2, order)

                new_count = int(c2.sum())
                if new_count > current_count:
                    chosen = c2
                    fc = fc2
                    current_count = new_count
                    found = True
                    no_improve = 0
                    break

            if found:
                continue

            # Try removing two edges and greedily re-adding
            if len(chosen_edges) >= 2:
                rng.shuffle(chosen_edges)
                for trial in range(8):
                    e1 = chosen_edges[trial]
                    e2 = chosen_edges[(trial + 1) % len(chosen_edges)]
                    c2 = chosen.copy()
                    fc2 = fc.copy()
                    remove_edge(c2, fc2, e1)
                    remove_edge(c2, fc2, e2)

                    order = list(range(num_edges))
                    rng.shuffle(order)
                    greedy_add(c2, fc2, order)

                    new_count = int(c2.sum())
                    if new_count > current_count:
                        chosen = c2
                        fc = fc2
                        current_count = new_count
                        found = True
                        no_improve = 0
                        break

            if not found:
                no_improve += 1
                if no_improve > 40:
                    break

        return current_count, chosen, fc

    def multi_removal_search(chosen, fc, num_attempts=50, seed=0):
        """Remove k edges randomly, then greedily re-add."""
        rng = random.Random(seed)
        current_count = int(chosen.sum())

        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break

            c2 = chosen.copy()
            fc2 = fc.copy()

            k = rng.randint(2, 5)
            candidates = [e for e in range(num_edges) if c2[e]]
            rng.shuffle(candidates)
            for e in candidates[:k]:
                remove_edge(c2, fc2, e)

            order = list(range(num_edges))
            rng.shuffle(order)
            greedy_add(c2, fc2, order)

            new_count = int(c2.sum())
            if new_count > current_count:
                chosen = c2
                fc = fc2
                current_count = new_count

        return current_count, chosen, fc

    # Phase 1: Multiple complementary deletion restarts
    best_count = 0
    best_chosen = None
    best_fc = None

    for seed in range(30):
        if time.time() > deadline - 45:
            break
        count, chosen, fc = complementary_deletion(seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_fc = fc.copy()

    # Phase 2: Local search from best
    if best_chosen is not None:
        best_count, best_chosen, best_fc = local_search(
            best_chosen, best_fc, max_rounds=250, seed=42)

    # Phase 3: Multi-removal perturbation
    best_count, best_chosen, best_fc = multi_removal_search(
        best_chosen, best_fc, num_attempts=60, seed=123)

    # Phase 4: Another local search
    best_count, best_chosen, best_fc = local_search(
        best_chosen, best_fc, max_rounds=200, seed=999)

    # Phase 5: Try more complementary deletions with different strategies
    for seed in range(30, 55):
        if time.time() > deadline - 12:
            break
        count, chosen, fc = complementary_deletion(seed)
        # Quick local search
        count, chosen, fc = local_search(chosen, fc, max_rounds=50, seed=seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_fc = fc.copy()

    # Final local search
    if best_chosen is not None and time.time() < deadline - 5:
        best_count, best_chosen, best_fc = local_search(
            best_chosen, best_fc, max_rounds=120, seed=7777)

    # Convert to output
    out = [list(all_edges[idx]) for idx in range(num_edges) if best_chosen[idx]]
    return out
# EVOLVE-BLOCK-END
