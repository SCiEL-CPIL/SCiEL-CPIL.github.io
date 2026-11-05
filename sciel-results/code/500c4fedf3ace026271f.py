"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
# Build edge index: maps (u,v) with u<v to its 1-based arrival index
edge_index = {}
for idx, (u, v) in enumerate(P['edge_order'], 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = idx

def edge_score(u, v):
    a, b = (u, v) if u < v else (v, u)
    return edge_index[(a, b)]

def cycle_score(cycle):
    """Return the max edge arrival index for a Hamilton cycle given as a permutation."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = edge_score(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave margin before 90s limit
    
    # Start with the trivial cycle 0,1,2,...,23
    best_cycle = list(range(N))
    best_score = cycle_score(best_cycle)
    
    # Build adjacency sorted by edge arrival index for quick access
    # adj[u][v] = edge_index
    adj = [[0] * N for _ in range(N)]
    for (u, v), idx in edge_index.items():
        adj[u][v] = idx
        adj[v][u] = idx
    
    # Local search: 2-opt style moves for Hamilton cycles
    # A 2-opt move: reverse a segment of the cycle
    # Also try single edge swaps
    
    import random
    rng = random.Random(42)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all 2-opt moves
        n = len(best_cycle)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Same cycle reversed
                
                # Compute new cycle score efficiently
                # The edges that change:
                # Old edges: (cycle[i-1], cycle[i]) and (cycle[j], cycle[(j+1)%n])
                # New edges: (cycle[i-1], cycle[j]) and (cycle[i], cycle[(j+1)%n])
                
                u1 = best_cycle[(i - 1) % n]
                u2 = best_cycle[i]
                v1 = best_cycle[j]
                v2 = best_cycle[(j + 1) % n]
                
                old_e1 = edge_score(u1, u2)
                old_e2 = edge_score(v1, v2)
                new_e1 = edge_score(u1, v1)
                new_e2 = edge_score(u2, v2)
                
                # Compute max of all edges except the two old ones, plus the two new ones
                # Quick check: if both new edges are worse than the current max, skip
                # More precisely: new_max = max(max of unchanged edges, new_e1, new_e2)
                
                # The unchanged edges are all edges except (u1,u2) and (v1,v2)
                # We can compute this, but it's O(n). Let's just do it for simplicity
                # since n=24 is small
                
                new_max = 0
                # Check all edges in new cycle
                new_cycle = best_cycle[:i] + best_cycle[i:j+1][::-1] + best_cycle[j+1:]
                
                # Actually, let's just compute the score directly
                sc = cycle_score(new_cycle)
                if sc < best_score:
                    best_score = sc
                    best_cycle = new_cycle
                    improved = True
                    break
            if improved:
                break
        
        # Try random single swaps
        if not improved and time.time() < deadline:
            for _ in range(1000):
                if time.time() > deadline:
                    break
                i, j = rng.sample(range(N), 2)
                candidate = best_cycle[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                sc = cycle_score(candidate)
                if sc < best_score:
                    best_score = sc
                    best_cycle = candidate
                    improved = True
                    break
        
        # If stuck, try a random restart with perturbation
        if not improved and time.time() < deadline:
            # Perturb: random 2-opt on a copy, then local search
            perturbed = best_cycle[:]
            i, j = rng.sample(range(N), 2)
            i, j = min(i, j), max(i, j)
            if i != 0 or j != N - 1:
                perturbed = perturbed[:i] + perturbed[i:j+1][::-1] + perturbed[j+1:]
            
            # Quick local search on perturbed
            p_score = cycle_score(perturbed)
            if p_score < best_score:
                best_score = p_score
                best_cycle = perturbed
                improved = True
            else:
                # Try 2-opt on perturbed
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    i, j = rng.sample(range(N), 2)
                    i, j = min(i, j), max(i, j)
                    if i == 0 and j == N - 1:
                        continue
                    cand = perturbed[:i] + perturbed[i:j+1][::-1] + perturbed[j+1:]
                    sc = cycle_score(cand)
                    if sc < p_score:
                        p_score = sc
                        perturbed = cand
                        if p_score < best_score:
                            best_score = p_score
                            best_cycle = perturbed
                            improved = True
                            break
    
    return list(best_cycle)
# EVOLVE-BLOCK-END
