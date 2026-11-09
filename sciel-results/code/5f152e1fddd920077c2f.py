import time
import random
import numpy as np
from collections import Counter

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 295

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)

    # Precompute distance matrix D[i,j] = squared distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)

    def build_data_structures(S_list):
        """Build dist_freq and dist_set for a given set S_list."""
        n = len(S_list)
        if n == 0:
            return {}, {}
        S_arr = np.array(S_list, dtype=np.int32)
        D_SS = D[np.ix_(S_arr, S_arr)]
        dist_freq = {}
        dist_set = {}
        for i in range(n):
            d_row = D_SS[i].tolist()
            dist_freq[S_list[i]] = Counter(d_row)
            dist_set[S_list[i]] = set(d_row)
        return dist_freq, dist_set

    def check_validity(p, S_list, dist_freq, dist_set):
        """Check if point p can be added to S. Returns (bool, d_p_S_array)."""
        if not S_list:
            return True, np.array([], dtype=np.int32)

        S_arr = np.array(S_list, dtype=np.int32)
        d_p_S = D[p, S_arr]

        # Condition 1: all distances from p to S must be distinct
        if len(np.unique(d_p_S)) != len(S_arr):
            return False, d_p_S

        # Condition 2: for each s in S, d(p,s) must not be in dist_set[s]
        for i, s in enumerate(S_list):
            if d_p_S[i] in dist_set[s]:
                return False, d_p_S

        return True, d_p_S

    def add_to_set(p, S_list, dist_freq, dist_set):
        """Add point p to S and update data structures."""
        S_list.append(p)
        S_arr = np.array(S_list, dtype=np.int32)
        d_p_all = D[p, S_arr]  # distances from p to all in S (including itself = 0)

        # Update dist_freq and dist_set for all existing points
        for i, s in enumerate(S_list):
            if s == p:
                continue
            d_val = d_p_all[i]
            dist_freq[s][d_val] += 1
            dist_set[s].add(d_val)

        # Initialize p's data structure
        dist_freq[p] = Counter(d_p_all.tolist())
        dist_set[p] = set(d_p_all.tolist())

    def remove_from_set(p, S_list, dist_freq, dist_set):
        """Remove point p from S and update data structures."""
        S_list.remove(p)
        S_arr = np.array(S_list, dtype=np.int32) if S_list else np.array([], dtype=np.int32)

        # Update dist_freq and dist_set for all remaining points
        for s in S_list:
            d_val = D[s, p]
            dist_freq[s][d_val] -= 1
            if dist_freq[s][d_val] == 0:
                del dist_freq[s][d_val]
                dist_set[s].discard(d_val)

        # Remove p's data
        del dist_freq[p]
        del dist_set[p]

    best_S = []
    best_size = 0

    # Phase 1: Multiple greedy restarts with different random orders
    for restart in range(200):
        if time.time() > deadline - 15:
            break

        S = []
        dist_freq = {}
        dist_set = {}
        candidates = list(range(M))
        random.shuffle(candidates)

        for p in candidates:
            if time.time() > deadline - 15:
                break
            valid, _ = check_validity(p, S, dist_freq, dist_set)
            if valid:
                add_to_set(p, S, dist_freq, dist_set)

        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    # Phase 2: Iterated Local Search (ILS)
    current_S = best_S.copy()
    current_size = best_size
    if current_size > 0:
        cur_dist_freq, cur_dist_set = build_data_structures(current_S)

    for iteration in range(5000):
        if time.time() > deadline - 10:
            break

        if current_size == 0:
            break

        # Perturbation: remove k random points
        k = random.randint(1, min(8, current_size))
        to_remove = random.sample(current_S, k)
        for r in to_remove:
            remove_from_set(r, current_S, cur_dist_freq, cur_dist_set)

        # Try to add points back / new points
        S_set = set(current_S)
        candidates = [p for p in range(M) if p not in S_set]
        random.shuffle(candidates)

        for p in candidates:
            if time.time() > deadline - 10:
                break
            valid, _ = check_validity(p, current_S, cur_dist_freq, cur_dist_set)
            if valid:
                add_to_set(p, current_S, cur_dist_freq, cur_dist_set)

        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = current_S.copy()
        else:
            # Accept with some probability (simulated annealing style)
            if random.random() < 0.15:
                pass  # Keep current as-is (already perturbed)
            else:
                # Revert to best
                current_S = best_S.copy()
                current_size = best_size
                if current_size > 0:
                    cur_dist_freq, cur_dist_set = build_data_structures(current_S)
                else:
                    cur_dist_freq = {}
                    cur_dist_set = {}
            current_size = len(current_S)

    # Phase 3: Single-point swap local search
    for iteration in range(10000):
        if time.time() > deadline - 5:
            break

        if best_size == 0:
            break

        S = best_S.copy()
        S_set = set(S)
        dist_freq, dist_set = build_data_structures(S)

        # Try swapping one point out and one in
        p_out = random.choice(S)
        remove_from_set(p_out, S, dist_freq, dist_set)

        candidates = [p for p in range(M) if p not in S_set and p != p_out]
        random.shuffle(candidates)

        improved = False
        for p_in in candidates[:500]:
            if time.time() > deadline - 5:
                break
            valid, _ = check_validity(p_in, S, dist_freq, dist_set)
            if valid:
                add_to_set(p_in, S, dist_freq, dist_set)
                improved = True
                break

        if improved and len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    return [tuple(points[i]) for i in best_S]
