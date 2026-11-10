import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448

    # Precompute edges
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v, i))

    # Precompute face membership: each edge belongs to (d-1)=6 faces
    pair_index = {}
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            pair_index[(i, j)] = idx
            idx += 1
    num_pairs = d * (d - 1) // 2  # 21
    num_rest = 1 << (d - 2)  # 32
    total_faces = num_pairs * num_rest  # 672

    edge_faces = np.zeros((num_edges, d - 1), dtype=np.int32)
    for e_idx, (u, v, i) in enumerate(edges):
        col = 0
        for j in range(d):
            if j == i:
                continue
            p_idx = pair_index[(i, j)] if i < j else pair_index[(j, i)]
            rest = 0
            bit_pos = 0
            for k in range(d):
                if k == i or k == j:
                    continue
                if u & (1 << k):
                    rest |= (1 << bit_pos)
                bit_pos += 1
            edge_faces[e_idx, col] = p_idx * num_rest + rest
            col += 1

    # Build inverse: for each face, which edges touch it
    # face_edges[fid] = list of edge indices
    face_edges = [[] for _ in range(total_faces)]
    for e_idx in range(num_edges):
        for fid in edge_faces[e_idx]:
            face_edges[fid].append(e_idx)

    # Convert to arrays for faster lookup
    face_edges_arr = []
    for fid in range(total_faces):
        face_edges_arr.append(np.array(face_edges[fid], dtype=np.int32))

    def solve_one_restart(budget):
        """One restart: random greedy fill + 2-opt local search."""
        selected = np.zeros(num_edges, dtype=bool)
        face_counts = np.zeros(total_faces, dtype=np.int32)

        def add_edge(e_idx):
            face_counts[edge_faces[e_idx]] += 1
            selected[e_idx] = True

        def remove_edge(e_idx):
            face_counts[edge_faces[e_idx]] -= 1
            selected[e_idx] = False

        def can_add(e_idx):
            return np.all(face_counts[edge_faces[e_idx]] < 3)

        # Random greedy fill
        order = np.arange(num_edges)
        np.random.shuffle(order)
        for e_idx in order:
            if can_add(e_idx):
                add_edge(e_idx)

        cur_size = int(np.sum(selected))
        best_size = cur_size
        best_selected = selected.copy()

        # 2-opt local search: try all (remove, add) pairs
        t_end = time.time() + budget
        improved = True
        while improved and time.time() < t_end:
            improved = False
            # Try 1-1 swaps
            for e_rm in np.where(selected)[0]:
                if time.time() >= t_end:
                    break
                remove_edge(e_rm)
                # Find best unselected edge we can add
                best_add = -1
                for e_add in np.where(~selected)[0]:
                    if can_add(e_add):
                        best_add = e_add
                        break
                if best_add >= 0:
                    add_edge(best_add)
                    improved = True
                    # Net zero change in size, but may enable further improvements
                    # Actually size stays same. Let's check if we can do 1-2.
                    # For 1-1, size is unchanged. We need 1-2 for net gain.
                    # Let's just track and move on.
                else:
                    add_edge(e_rm)

            # Try 1-2 swaps: remove one, add two
            for e_rm in np.where(selected)[0]:
                if time.time() >= t_end:
                    break
                remove_edge(e_rm)
                added = []
                # Try to add two edges
                for e_add1 in np.where(~selected)[0]:
                    if can_add(e_add1):
                        add_edge(e_add1)
                        added.append(e_add1)
                        for e_add2 in np.where(~selected)[0]:
                            if e_add2 == e_add1:
                                continue
                            if can_add(e_add2):
                                add_edge(e_add2)
                                added.append(e_add2)
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    for e in added:
                        remove_edge(e)
                    add_edge(e_rm)

        cur_size = int(np.sum(selected))
        if cur_size > best_size:
            best_size = cur_size
            best_selected = selected.copy()

        return best_size, best_selected

    # Multiple restarts
    best_size = 0
    best_selected = None
    t_start = time.time()

    # Give each restart a budget, with more time for later restarts if going well
    num_restarts_max = 8
    for r in range(num_restarts_max):
        elapsed = time.time() - t_start
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        # Budget for this restart: remaining / (num_restarts_max - r), capped
        budget = min(remaining / (num_restarts_max - r), remaining - 3.0)
        if budget < 1.0:
            break

        size, sel = solve_one_restart(budget)
        if size > best_size:
            best_size = size
            best_selected = sel.copy()

        # Early exit if we hit the known best
        if best_size >= 304:
            break

    # Final polish: one more local search pass on the best solution
    if best_selected is not None and time.time() < deadline - 5:
        selected = best_selected.copy()
        face_counts = np.zeros(total_faces, dtype=np.int32)
        for e_idx in np.where(selected)[0]:
            face_counts[edge_faces[e_idx]] += 1

        def add_edge(e_idx):
            face_counts[edge_faces[e_idx]] += 1
            selected[e_idx] = True

        def remove_edge(e_idx):
            face_counts[edge_faces[e_idx]] -= 1
            selected[e_idx] = False

        def can_add(e_idx):
            return np.all(face_counts[edge_faces[e_idx]] < 3)

        t_end = deadline - 2
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for e_rm in np.where(selected)[0]:
                if time.time() >= t_end:
                    break
                remove_edge(e_rm)
                added = []
                for e_add1 in np.where(~selected)[0]:
                    if can_add(e_add1):
                        add_edge(e_add1)
                        added.append(e_add1)
                        for e_add2 in np.where(~selected)[0]:
                            if e_add2 == e_add1:
                                continue
                            if can_add(e_add2):
                                add_edge(e_add2)
                                added.append(e_add2)
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    for e in added:
                        remove_edge(e)
                    add_edge(e_rm)

        final_size = int(np.sum(selected))
        if final_size > best_size:
            best_size = final_size
            best_selected = selected.copy()

    # Build result
    if best_selected is None:
        # Fallback: return empty (shouldn't happen)
        return []

    result = []
    for e_idx in range(num_edges):
        if best_selected[e_idx]:
            u, v, _ = edges[e_idx]
            result.append([u, v])

    return result
