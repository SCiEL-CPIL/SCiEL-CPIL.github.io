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

    # Precompute face membership
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
    face_edges = [[] for _ in range(total_faces)]
    for e_idx in range(num_edges):
        for fid in edge_faces[e_idx]:
            face_edges[fid].append(e_idx)

    face_edges_arr = []
    for fid in range(total_faces):
        face_edges_arr.append(np.array(face_edges[fid], dtype=np.int32))

    def solve_one_restart(budget):
        """One restart: random greedy fill + local search with 1-2 exchanges."""
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

        # Local search: try 1-2 swaps (remove one, add two) for net +1
        t_end = time.time() + budget
        improved = True
        while improved and time.time() < t_end:
            improved = False
            selected_indices = np.where(selected)[0]
            unselected_indices = np.where(~selected)[0]

            # Randomize order for diversification
            np.random.shuffle(selected_indices)
            np.random.shuffle(unselected_indices)

            for e_rm in selected_indices:
                if time.time() >= t_end:
                    break
                remove_edge(e_rm)
                added = []
                found = False
                for e_add1 in unselected_indices:
                    if can_add(e_add1):
                        add_edge(e_add1)
                        added.append(e_add1)
                        for e_add2 in unselected_indices:
                            if e_add2 == e_add1:
                                continue
                            if can_add(e_add2):
                                add_edge(e_add2)
                                added.append(e_add2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                if not found:
                    for e in added:
                        remove_edge(e)
                    add_edge(e_rm)

            # Also try 2-3 swaps for net +1
            if time.time() < t_end:
                selected_indices = np.where(selected)[0]
                unselected_indices = np.where(~selected)[0]
                np.random.shuffle(selected_indices)
                np.random.shuffle(unselected_indices)

                for i in range(min(len(selected_indices), 100)):
                    if time.time() >= t_end:
                        break
                    e_rm1 = selected_indices[i]
                    # Try removing one edge and adding two
                    remove_edge(e_rm1)
                    added = []
                    found = False
                    for e_add1 in unselected_indices:
                        if can_add(e_add1):
                            add_edge(e_add1)
                            added.append(e_add1)
                            for e_add2 in unselected_indices:
                                if e_add2 == e_add1:
                                    continue
                                if can_add(e_add2):
                                    add_edge(e_add2)
                                    added.append(e_add2)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                    if not found:
                        for e in added:
                            remove_edge(e)
                        add_edge(e_rm1)

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
            selected_indices = np.where(selected)[0]
            unselected_indices = np.where(~selected)[0]
            np.random.shuffle(selected_indices)
            np.random.shuffle(unselected_indices)

            for e_rm in selected_indices:
                if time.time() >= t_end:
                    break
                remove_edge(e_rm)
                added = []
                found = False
                for e_add1 in unselected_indices:
                    if can_add(e_add1):
                        add_edge(e_add1)
                        added.append(e_add1)
                        for e_add2 in unselected_indices:
                            if e_add2 == e_add1:
                                continue
                            if can_add(e_add2):
                                add_edge(e_add2)
                                added.append(e_add2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                if not found:
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
