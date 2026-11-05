"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute arrival index for each edge (1-based)
edge_idx = {}
for i, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    edge_idx[key] = i + 1

def escore(u, v):
    key = (min(u, v), max(u, v))
    return edge_idx[key]

def cycle_max(cycle):
    mx = 0
    for i in range(n):
        s = escore(cycle[i], cycle[(i+1)%n])
        if s > mx:
            mx = s
    return mx

def greedy_build(start, rng=None):
    """Greedy: always pick the unvisited vertex with minimum edge arrival index from current."""
    if rng is None:
        rng = random.Random(0)
    visited = [False]*n
    visited[start] = True
    path = [start]
    for _ in range(n-1):
        cur = path[-1]
        cands = [v for v in range(n) if not visited[v]]
        cands.sort(key=lambda v: escore(cur, v))
        # Pick the best that doesn't strand us
        chosen = None
        for v in cands:
            if len(path) == n-1:
                chosen = v
                break
            else:
                # Ensure v has at least one unvisited neighbor or can connect to start
                has_future = False
                for w in range(n):
                    if not visited[w] and w != v:
                        has_future = True
                        break
                if has_future or escore(v, start) < 999999:
                    chosen = v
                    break
        if chosen is None:
            chosen = cands[0]
        visited[chosen] = True
        path.append(chosen)
    return path

def two_opt(cycle, max_iter=5000, rng=None):
    """2-opt improvement: try reversing segments to reduce max edge score."""
    if rng is None:
        rng = random.Random(0)
    cur = list(cycle)
    cur_score = cycle_max(cur)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(n-1):
            for j in range(i+1, n):
                if i == 0 and j == n-1:
                    continue
                # Reverse segment [i+1, j]
                new_cur = cur[:i+1] + cur[i+1:j+1][::-1] + cur[j+1:]
                new_s = cycle_max(new_cur)
                if new_s < cur_score:
                    cur = new_cur
                    cur_score = new_s
                    improved = True
                    break
            if improved:
                break
    return cur, cur_score

def perturb(cycle, rng):
    c = list(cycle)
    i = rng.randint(0, n-1)
    j = rng.randint(0, n-1)
    while abs(i-j) <= 1 or abs(i-j) >= n-1:
        i = rng.randint(0, n-1)
        j = rng.randint(0, n-1)
    c[i], c[j] = c[j], c[i]
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 78.0
    
    best_cycle = list(range(n))
    best_score = cycle_max(best_cycle)
    
    # Initial greedy from multiple starts
    for s in range(n):
        if time.time() > deadline - 10:
            break
        c = greedy_build(s)
        sc = cycle_max(c)
        if sc < best_score:
            best_score = sc
            best_cycle = c
    
    # Improve best
    if time.time() < deadline - 5:
        best_cycle, best_score = two_opt(best_cycle, max_iter=10000)
    
    # Iterated local search
    rng = random.Random(999)
    count = 0
    while time.time() < deadline - 2:
        count += 1
        p = perturb(best_cycle, rng)
        c, sc = two_opt(p, max_iter=3000)
        if sc < best_score:
            best_score = sc
            best_cycle = c
        # Occasionally fresh start
        if count % 15 == 0:
            s = rng.randint(0, n-1)
            c2 = greedy_build(s, rng)
            sc2 = cycle_max(c2)
            if sc2 < best_score:
                best_score = sc2
                best_cycle = c2
    
    return best_cycle
# EVOLVE-BLOCK-END
