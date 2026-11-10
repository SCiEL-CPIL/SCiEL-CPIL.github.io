"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 1 << d
    deadline = time.time() + 135  # safety margin before 150s kill

    # Precompute all edges: edge (u, v) where v = u | (1<<i), bit i of u is 0
    # edge_id = u * d + i
    edges = []
    for u in range(n):
        for i in range(d):
            if not (u & (1 << i)):
                v = u | (1 << i)
                edges.append((u, v, i))
    num_edges = len(edges)

    # Precompute face IDs
    # A face is identified by (pair of dimensions, rest_bits)
    pair_index = {}
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            pair_index[(i, j)] = idx
            idx += 1
    num_pairs = len(pair_index)
    total_faces = num_pairs * (1 << (d - 2))

    def get_compact_rest(u, i, j):
        val = 0
        bit_pos = 0
        for k in range(d):
            if k == i or k == j:
                continue
            if u & (1 << k):
                val |= (1 << bit_pos)
            bit_pos += 1
        return val

    # For each edge, precompute the list of face IDs it belongs to
    edge_faces = []
    for u, v, i in edges:
        faces = []
        for j in range(d):
            if j == i:
                continue
            if i < j:
                p_idx = pair_index[(i, j)]
            else:
                p_idx = pair_index[(j, i)]
            rest_compact = get_compact_rest(u, i, j)
            face_id = p_idx * (1 << (d - 2)) + rest_compact
            faces.append(face_id)
        edge_faces.append(faces)

    # State
    face_counts = [0] * total_faces
    selected = set()

    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected.add(e_idx)

    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected.remove(e_idx)

    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True

    def greedy_fill(order):
        for e_idx in order:
            if can_add(e_idx):
                add_edge(e_idx)

    def try_add_missing():
        missing = [i for i in range(num_edges) if i not in selected]
        random.shuffle(missing)
        added_any = False
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any

    best_size = 0
    best_selected = set()

    # Initial greedy with random order
    order = list(range(num_edges))
    random.shuffle(order)
    greedy_fill(order)
    best_size = len(selected)
    best_selected = set(selected)

    # Local search loop
    while time.time() < deadline:
        if not try_add_missing():
            # Stuck - do a kick
            if len(selected) < 2:
                # Restart
                selected.clear()
                face_counts = [0] * total_faces
                order = list(range(num_edges))
                random.shuffle(order)
                greedy_fill(order)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                continue

            # Save state
            old_face_counts = face_counts[:]
            old_selected = set(selected)

            # Remove 2 random edges
            to_remove = random.sample(list(selected), 2)
            for e_idx in to_remove:
                remove_edge(e_idx)

            # Greedy re-add
            missing = [i for i in range(num_edges) if i not in selected]
            random.shuffle(missing)
            for e_idx in missing:
                if can_add(e_idx):
                    add_edge(e_idx)

            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
            elif len(selected) < len(old_selected):
                # Revert
                selected = old_selected
                face_counts = old_face_counts

        # Periodic restart to escape deep local optima
        if random.random() < 0.01:
            selected.clear()
            face_counts = [0] * total_faces
            order = list(range(num_edges))
            random.shuffle(order)
            greedy_fill(order)
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)

    # Restore best solution
    selected = best_selected

    # Build result
    result = []
    for e_idx in best_selected:
        u, v, _ = edges[e_idx]
        result.append([u, v])

    return result
# EVOLVE-BLOCK-END
