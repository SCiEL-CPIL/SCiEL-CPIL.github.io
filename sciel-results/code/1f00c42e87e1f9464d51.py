"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    # Deadline guard: 110 seconds to be safe within 120/150s limits
    deadline = time.time() + 110.0
    
    # Helper to compute cost
    def get_cost(p):
        # p is a permutation array
        return np.sum(a * b[np.ix_(p, p)])

    # Heuristic start: Rank matching
    def rank_match_start():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # Local search: 2-opt
    def local_search(p, current_cost):
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(n):
                u = p[i]
                # Compute deltas for swapping i with all j
                # We vectorize this over j
                # j ranges from 0 to n-1, j != i
                js = np.array([j for j in range(n) if j != i], dtype=np.int64)
                vs = p[js]
                
                # Term 1: Sum over k != i, j of (a[i,k] + a[j,k]) * (b[u,k] - b[v,k])
                # This is tricky to fully vectorize over j and k simultaneously without O(n^2) memory per i,
                # but n=100 is small.
                # Let's compute it efficiently.
                # For each j, we need sum_k (a[i,k] + a[j,k]) * (b[u,k] - b[p[j],k])
                
                # Precompute parts
                # a[i, :] is fixed for this i
                ai = a[i]
                bu = b[u]
                
                # We need to compute for each j in js:
                # S_j = sum_{k != i, j} (ai[k] + a[j,k]) * (bu[k] - b[p[j],k])
                
                # Expand:
                # S_j = sum_{k != i, j} ai[k]*bu[k] - ai[k]*b[p[j],k] + a[j,k]*bu[k] - a[j,k]*b[p[j],k]
                
                # This is getting complex. Let's use a simpler, robust O(n^2) per i approach
                # which is still fast enough for n=100.
                
                best_delta = 0
                best_j = -1
                
                for j in js:
                    v = p[j]
                    # Calculate delta for swapping i and j
                    # Formula:
                    # delta = 2 * sum_{k != i, j} (a[i,k] + a[j,k]) * (b[u,k] - b[v,k]) 
                    #         + 2 * a[i,j] * (b[u,v] - b[v,u]) 
                    #         + a[i,i] * (b[u,u] - b[v,v]) 
                    #         + a[j,j] * (b[v,v] - b[u,u])
                    
                    # Vectorized sum over k
                    # Mask k != i, j
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    
                    # (a[i,k] + a[j,k]) * (b[u,k] - b[v,k])
                    term1 = np.sum((a[i, mask] + a[j, mask]) * (b[u, mask] - b[v, mask]))
                    
                    term2 = 2 * a[i, j] * (b[u, v] - b[v, u])
                    term3 = a[i, i] * (b[u, u] - b[v, v])
                    term4 = a[j, j] * (b[v, v] - b[u, u])
                    
                    delta = term1 + term2 + term3 + term4
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                        
                if best_j != -1 and best_delta < 0:
                    # Perform swap
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += best_delta
                    improved = True
                    # Restart the loop to ensure we catch all improvements
                    break
                    
        return p, current_cost

    best_p = None
    best_cost = float('inf')
    
    # Initial run with rank matching
    p = rank_match_start()
    cost = get_cost(p)
    p, cost = local_search(p, cost)
    
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
        
    # Multiple random restarts if time permits
    while time.time() < deadline - 10.0:
        p = np.random.permutation(n).astype(np.int64)
        cost = get_cost(p)
        
        # Quick check: if random start is already very bad, maybe skip? 
        # But local search is fast.
        
        p, cost = local_search(p, cost)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
