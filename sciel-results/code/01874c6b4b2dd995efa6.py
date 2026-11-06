import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def _cycle_sum(cycle):
    return sum(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin for 90s limit
    
    # Precompute edge indices matrix for fast lookup
    idx_mat = [[0]*N for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if u != v:
                idx_mat[u][v] = _edge_idx(u, v)
    
    # Trivial cycle
    trivial = list(range(N))
    best_cycle = trivial[:]
    best_score = _cycle_score(trivial)
    best_sum = _cycle_sum(trivial)
    
    # Optimized 2-opt for minimizing SUM of edge indices
    def two_opt_sum(cycle, time_limit):
        n = N
        improved = True
        deadline_local = time.time() + time_limit
        best = cycle[:]
        best_sc = _cycle_sum(best)
        
        while improved and time.time() < deadline_local:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline_local: break
                for j in range(i + 2, n):
                    # Reverse segment i..j
                    # Calculate delta sum efficiently?
                    # For N=24, full recalculation is cheap enough if optimized.
                    # But let's try to be a bit smarter.
                    # Edges removed: (cycle[i-1], cycle[i]) and (cycle[j], cycle[j+1])
                    # Edges added: (cycle[i-1], cycle[j]) and (cycle[i], cycle[j+1])
                    # Note: cycle is 0-indexed, wrap around.
                    u1 = cycle[i-1]
                    v1 = cycle[i]
                    u2 = cycle[j]
                    v2 = cycle[(j+1) % n]
                    
                    old_sum_part = idx_mat[u1][v1] + idx_mat[u2][v2]
                    new_sum_part = idx_mat[u1][v2] + idx_mat[v1][u2]
                    
                    if new_sum_part < old_sum_part:
                        best[i:j+1] = reversed(best[i:j+1])
                        best_sc += new_sum_part - old_sum_part
                        improved = True
                    # If not improved, we don't reverse, so no change to best
                    # But wait, the logic above assumes we apply the move.
                    # Standard 2-opt: try move, if better, keep it.
                    # My loop structure:
                    #   for i...
                    #     for j...
                    #       if better: reverse, update score, improved=True
                    #       else: do nothing (since best is not modified yet)
                    # This is correct.
        
        return best, best_sc

    # Optimized 2-opt for minimizing MAX edge index
    def two_opt_max(cycle, time_limit):
        n = N
        improved = True
        deadline_local = time.time() + time_limit
        best = cycle[:]
        best_sc = _cycle_score(best)
        
        while improved and time.time() < deadline_local:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline_local: break
                for j in range(i + 2, n):
                    u1 = cycle[i-1]
                    v1 = cycle[i]
                    u2 = cycle[j]
                    v2 = cycle[(j+1) % n]
                    
                    old_max = max(idx_mat[u1][v1], idx_mat[u2][v2])
                    new_max = max(idx_mat[u1][v2], idx_mat[v1][u2])
                    
                    # We want to reduce the global max.
                    # If the global max is not in the edges being changed, the max won't decrease.
                    # If the global max IS in the edges being changed, and new_max < best_sc, then we improve.
                    # Actually, simply checking if new_max < best_sc is sufficient for improvement of the objective.
                    # But we must ensure the new edges don't introduce a *larger* max than the current global max?
                    # No, the objective is min(global_max).
                    # If we replace two edges, the new global max is max( max(other_edges), new_max ).
                    # Since other_edges are fixed, if new_max < best_sc, it might still result in a global max of best_sc
                    # if some other edge has index best_sc.
                    # However, if we strictly decrease the max of the *changed* edges, and that max was the bottleneck, we improve.
                    # A simpler heuristic: accept if new_max < best_sc.
                    # This is safe because it can't increase the global max.
                    # It might not decrease it if the bottleneck is elsewhere, but it's a valid non-worsening move.
                    
                    if new_max < best_sc:
                        best[i:j+1] = reversed(best[i:j+1])
                        # Recalculate score to be safe, or update incrementally?
                        # Recalculating max for N=24 is fast.
                        best_sc = _cycle_score(best)
                        improved = True
        
        return best, best_sc

    rng = random.Random(42)
    
    # Phase 1: Sum minimization from multiple random starts
    # This helps find a "dense" cycle in the early edges.
    while time.time() < deadline - 20.0:
        perm = list(range(N))
        rng.shuffle(perm)
        
        # Optimize sum
        sum_cycle, sum_score = two_opt_sum(perm, 2.0)
        
        # Optimize max on the result
        max_cycle, max_score = two_opt_max(sum_cycle, 2.0)
        
        if max_score < best_score:
            best_score = max_score
            best_cycle = max_cycle[:]
            best_sum = sum_score

    # Phase 2: Intensive local search on the best found so far
    # Alternate between sum and max optimization to escape local optima
    current_cycle = best_cycle[:]
    current_sum = _cycle_sum(current_cycle)
    current_max = _cycle_score(current_cycle)
    
    while time.time() < deadline - 2.0:
        # Try improving sum
        sum_cycle, sum_score = two_opt_sum(current_cycle, 1.0)
        if sum_score < current_sum:
            current_cycle = sum_cycle
            current_sum = sum_score
        
        # Try improving max
        max_cycle, max_score = two_opt_max(current_cycle, 1.0)
        if max_score < current_max:
            current_cycle = max_cycle
            current_max = max_score
            best_score = max_score
            best_cycle = current_cycle[:]
            best_sum = current_sum

    return best_cycle
