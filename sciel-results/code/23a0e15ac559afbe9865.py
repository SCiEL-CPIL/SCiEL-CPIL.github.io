"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

TOTAL_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
for i in range(TOTAL_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, TOTAL_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist2[i, j] = d
        dist2[j, i] = d

# Precompute iso_pairs
# iso_pairs[i][j] is a set of k such that {i, j, k} is isosceles
iso_pairs = [[set() for _ in range(TOTAL_POINTS)] for __ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    for j in range(i + 1, TOTAL_POINTS):
        dij = dist2[i, j]
        for k in range(TOTAL_POINTS):
            if k == i or k == j:
                continue
            dik = dist2[i, k]
            djk = dist2[j, k]
            if dij == dik or dij == djk or dik == djk:
                iso_pairs[i][j].add(k)
                iso_pairs[j][i].add(k)
                iso_pairs[k][i].add(j)
                iso_pairs[k][j].add(i)

def construct():
    start_time = time.time()
    deadline = 300.0
    best_set = None
    best_size = TOTAL_POINTS + 1

    def is_compatible(p, S):
        for a in S:
            if iso_pairs[p][a] & S:
                return False
        return True

    def get_new_blocks(p, S, available):
        count = 0
        for q in available:
            if q == p:
                continue
            if iso_pairs[q][p] & S:
                count += 1
        return count

    def greedy_construct(random_seed):
        random.seed(random_seed)
        S = set()
        start_p = random.randint(0, TOTAL_POINTS - 1)
        S.add(start_p)
        available = set(range(TOTAL_POINTS))
        available.remove(start_p)

        while available:
            if time.time() - start_time > deadline - 5.0:
                return S

            best_candidate = None
            best_score = -1
            
            # Optimization: If S is large, is_compatible is expensive.
            # But N=16, S is small.
            
            for p in available:
                if not is_compatible(p, S):
                    continue
                
                score = get_new_blocks(p, S, available)
                # Add noise for diversity
                score += random.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_candidate = p

            if best_candidate is None:
                break

            S.add(best_candidate)
            available.remove(best_candidate)

            # Update available
            to_remove = []
            for q in available:
                if iso_pairs[q][best_candidate] & S:
                    to_remove.append(q)
            
            for q in to_remove:
                available.remove(q)

        return S

    # Run restarts
    num_restarts = 0
    while time.time() - start_time < deadline - 10.0:
        s = greedy_construct(num_restarts)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            if best_size <= 15: # Heuristic threshold
                # Try a few more to see if we can go lower, but don't spend too much time
                pass
        num_restarts += 1

    if best_set is None:
        best_set = greedy_construct(0)

    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
