import random
import time

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s kill

    d = 7
    n = 1 << d  # 128 vertices

    # Precompute all edges and for each edge, the 5 faces it belongs to
    all_edges = []
    edge_set = set()
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
                edge_set.add((u, v))

    E = len(all_edges)  # 448

    # For each edge, precompute the faces (as frozensets of 4 edge-tuples) it participates in
    # A face is determined by two bits i,j and a vertex x that is 0 in both i,j
    # Face edges: (x, x^i), (x, x^j), (x^i, x^i^j), (x^j, x^i^j)
    edge_to_faces = {e: [] for e in all_edges}
    all_faces = []
    for i in range(d):
        for j in range(i + 1, d):
            # x must have 0 in bits i and j
            for x in range(n):
                if (x >> i) & 1 or (x >> j) & 1:
                    continue
                a = x
                b = x ^ (1 << i)
                c = x ^ (1 << j)
                e4 = x ^ (1 << i) ^ (1 << j)
                e_ab = (min(a, b), max(a, b))
                e_ac = (min(a, c), max(a, c))
                e_be = (min(b, e4), max(b, e4))
                e_ce = (min(c, e4), max(c, e4))
                face = frozenset([e_ab, e_ac, e_be, e_ce])
                all_faces.append(face)
                for fe in (e_ab, e_ac, e_be, e_ce):
                    edge_to_faces[fe].append(face)

    def is_valid_add(chosen, edge):
        # Check if adding edge would complete any face
        for face in edge_to_faces[edge]:
            if all(e in chosen for e in face):
                return False
        return True

    def face_count(chosen):
        # Number of faces with all 4 edges present (should be 0 for valid)
        cnt = 0
        for face in all_faces:
            if all(e in chosen for e in face):
                cnt += 1
        return cnt

    def greedy_random(rng):
        chosen = set()
        edges = list(all_edges)
        rng.shuffle(edges)
        for e in edges:
            if is_valid_add(chosen, e):
                chosen.add(e)
        return chosen

    def local_search(chosen, rng, time_limit):
        chosen = set(chosen)
        improved = True
        # Track faces that are "almost full" (3 edges present)
        # For removal: try removing an edge, then greedily add back
        # For addition: try adding a removed edge
        removed = set()
        # Build removed set
        for e in all_edges:
            if e not in chosen:
                removed.add(e)

        while improved and time.time() < time_limit:
            improved = False
            # Try to add edges
            add_candidates = list(removed)
            rng.shuffle(add_candidates)
            for e in add_candidates:
                if is_valid_add(chosen, e):
                    chosen.add(e)
                    removed.discard(e)
                    improved = True
            if time.time() >= time_limit:
                break
            # Try removal + refill: remove an edge, then greedily add back as many as possible
            # Only beneficial if we can add more than we removed
            # Try a few random edges to remove
            trial_removals = list(chosen)
            rng.shuffle(trial_removals)
            for e in trial_removals[:200]:  # limit trials
                if time.time() >= time_limit:
                    break
                chosen.discard(e)
                removed.add(e)
                # Now try to add back many
                added_count = 0
                add_candidates2 = list(removed)
                rng.shuffle(add_candidates2)
                for e2 in add_candidates2:
                    if is_valid_add(chosen, e2):
                        chosen.add(e2)
                        removed.discard(e2)
                        added_count += 1
                if added_count > 1:
                    improved = True
                else:
                    # revert: put e back, remove the one we added
                    if added_count == 1:
                        # find what was added
                        for e2 in list(chosen):
                            if e2 in removed:
                                pass
                        # simpler: just keep it if net positive
                        pass
                    else:
                        # put e back
                        chosen.add(e)
                        removed.discard(e)

        return chosen

    best = None
    best_size = 0

    # Multiple random greedy starts
    rng = random.Random(42)
    num_starts = 8
    for s in range(num_starts):
        if time.time() > deadline - 30:
            break
        chosen = greedy_random(rng)
        size = len(chosen)
        if size > best_size:
            best = chosen
            best_size = size
        # Local search from this start
        time_limit = deadline - 25
        chosen2 = local_search(chosen, rng, time_limit)
        size2 = len(chosen2)
        if size2 > best_size:
            best = chosen2
            best_size = size2
        if time.time() > deadline - 15:
            break

    # Final attempt: pure greedy add from best
    if best is not None and time.time() < deadline - 10:
        chosen = set(best)
        removed = set(all_edges) - chosen
        add_candidates = list(removed)
        rng.shuffle(add_candidates)
        for e in add_candidates:
            if is_valid_add(chosen, e):
                chosen.add(e)
        if len(chosen) > best_size:
            best = chosen
            best_size = len(chosen)

    # Convert to list of [u, v]
    out = []
    for e in best:
        out.append([e[0], e[1]])
    return out
