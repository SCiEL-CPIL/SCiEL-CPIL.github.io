"""Edge-colourings of K16: every K4 sees >=5 colours. Minimize colours used."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120

    edge_idx = {}
    for idx, (a, b) in enumerate(edges):
        edge_idx[(a, b)] = idx

    # All K4s
    from itertools import combinations
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)  # 1820

    # k4_edge_idx[k4_idx] = array of 6 edge indices
    k4_edge_idx = np.zeros((num_k4s, 6), dtype=np.int32)
    for k4_i, (a, b, c, d) in enumerate(k4s):
        k4_edge_idx[k4_i] = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]

    # edge_to_k4s[e] = list of K4 indices containing edge e
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_i in range(num_k4s):
        for e in k4_edge_idx[k4_i]:
            edge_to_k4s[e].append(k4_i)

    # Vectorized: compute distinct colour counts for all K4s at once
    def compute_k4_counts(colors):
        """colors: shape (120,). Returns array of length num_k4s with distinct colour counts."""
        # k4_colors shape (num_k4s, 6)
        k4_colors = k4_edge_idx  # indices into colors
        k4_colors = colors[k4_edge_idx]  # (num_k4s, 6)
        # Sort each row
        k4_sorted = np.sort(k4_colors, axis=1)
        # Count distinct: number of positions where element != previous (or first)
        # distinct = 1 + sum(k4_sorted[:,1:] != k4_sorted[:,:-1])
        distinct = 1 + np.sum(k4_sorted[:, 1:] != k4_sorted[:, :-1], axis=1)
        return distinct

    def is_valid(colors):
        counts = compute_k4_counts(colors)
        return bool(np.all(counts >= 5))

    def min_count(colors):
        return int(np.min(compute_k4_counts(colors)))

    # ---- Initial valid coloring: (i+j) mod 17 uses 17 colours ----
    colors = np.zeros(num_edges, dtype=np.int32)
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
    assert is_valid(colors), "initial coloring must be valid"

    best_colors = colors.copy()
    best_num_colors = int(colors.max()) + 1  # 17

    def try_greedy(max_colors, num_restarts=30):
        """Try to build a valid coloring using at most max_colors colours via randomized greedy."""
        for r in range(num_restarts):
            if time.time() > deadline - 10.0:
                return None
            cur = np.zeros(num_edges, dtype=np.int32)
            assigned = np.zeros(num_edges, dtype=bool)
            order = np.random.permutation(num_edges)
            ok = True
            for e in order:
                best_c = -1
                # Try colours in random order
                c_order = np.random.permutation(max_colors)
                for c in c_order:
                    cur[e] = c
                    # Check all K4s containing e that are fully assigned
                    valid = True
                    for k4_i in edge_to_k4s[e]:
                        e_list = k4_edge_idx[k4_i]
                        if not np.all(assigned[e_list]):
                            continue
                        sub = cur[e_list]
                        # distinct count
                        if len(set(sub.tolist())) < 5:
                            valid = False
                            break
                    if valid:
                        best_c = c
                        break
                if best_c == -1:
                    ok = False
                    break
                cur[e] = best_c
                assigned[e] = True
            if ok and is_valid(cur):
                return cur
        return None

    # Try to get down to 15, 14, 13 via greedy
    for target in [15, 14, 13]:
        if time.time() > deadline - 20.0:
            break
        res = try_greedy(target, num_restarts=50)
        if res is not None:
            nc = int(res.max()) + 1
            if nc < best_num_colors:
                best_colors = res
                best_num_colors = nc
        # Also try from current best via local search to reduce

    # ---- Local search / simulated annealing to reduce colour count ----
    def local_search_reduce(colors, target_max, time_limit=15.0):
        """Try to reduce max colour to target_max using hill-climbing with restarts."""
        cur = colors.copy()
        cur_max = int(cur.max()) + 1
        if cur_max <= target_max:
            return cur
        end = time.time() + time_limit
        best_local = cur.copy()
        best_local_max = cur_max
        while time.time() < end:
            # Pick a random edge with the highest colour
            max_c = int(cur.max())
            if max_c < target_max:
                break
            candidates = np.where(cur == max_c)[0]
            if len(candidates) == 0:
                break
            e = int(candidates[np.random.randint(len(candidates))])
            old_c = cur[e]
            # Try to recolor to a lower colour
            new_c = int(np.random.randint(0, max_c))
            cur[e] = new_c
            # Check validity
            valid = True
            for k4_i in edge_to_k4s[e]:
                sub = cur[k4_edge_idx[k4_i]]
                if len(set(sub.tolist())) < 5:
                    valid = False
                    break
            if not valid:
                cur[e] = old_c
                continue
            new_max = int(cur.max()) + 1
            if new_max < best_local_max:
                best_local = cur.copy()
                best_local_max = new_max
        return best_local

    # Attempt reduction via local search
    for target in range(best_num_colors - 1, 12, -1):
        if time.time() > deadline - 15.0:
            break
        res = local_search_reduce(best_colors, target, time_limit=10.0)
        nc = int(res.max()) + 1
        if nc < best_num_colors:
            best_colors = res
            best_num_colors = nc

    # ---- Simulated annealing on the best found so far ----
    if time.time() < deadline - 5.0:
        cur = best_colors.copy()
        cur_max = int(cur.max()) + 1
        temp = 1.0
        cool = 0.9995
        best_sa = cur.copy()
        best_sa_max = cur_max
        end = deadline - 3.0
        while time.time() < end:
            e = int(np.random.randint(num_edges))
            old_c = cur[e]
            max_possible = min(cur_max - 1, 15)
            if max_possible < 0:
                break
            new_c = int(np.random.randint(0, max_possible + 1))
            if new_c == old_c:
                continue
            cur[e] = new_c
            valid = True
            for k4_i in edge_to_k4s[e]:
                sub = cur[k4_edge_idx[k4_i]]
                if len(set(sub.tolist())) < 5:
                    valid = False
                    break
            if not valid:
                cur[e] = old_c
                continue
            new_max = int(cur.max()) + 1
            if new_max < best_sa_max:
                best_sa = cur.copy()
                best_sa_max = new_max
                if new_max < best_num_colors:
                    best_colors = cur.copy()
                    best_num_colors = new_max
            elif new_max == best_sa_max:
                # accept with small probability
                if random.random() < 0.1:
                    pass
                else:
                    cur[e] = old_c
            temp *= cool
            if temp < 0.05:
                temp = 0.05

    # Final validation
    if not is_valid(best_colors):
        # Fallback
        for idx, (i, j) in enumerate(edges):
            best_colors[idx] = (i + j) % 17

    return [int(c) for c in best_colors]
