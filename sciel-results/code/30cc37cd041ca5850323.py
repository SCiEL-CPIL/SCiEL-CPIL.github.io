"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill (directive says 150s but use 120s for safety)
    
    # Ensure float64 for faster computation
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    def cost(p):
        """Compute exact cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return np.sum(a * b[np.ix_(p, p)])
    
    def cost_delta_2swap(p, i, j):
        """
        Compute change in cost when swapping positions i and j in permutation p.
        p is a list/array where p[k] = facility at location k? 
        Wait, the interface says: "Facility i is placed at location p[i]"
        So p[i] is the location of facility i.
        Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        Let's re-derive the delta for swapping p[i] and p[j].
        Let x = p[i], y = p[j].
        New p[i] = y, new p[j] = x.
        
        Terms involving i or j change.
        Old terms involving i: sum_k a[i,k] * b[x, p[k]] + sum_k a[k,i] * b[p[k], x]
        But note: sum_{i,j} includes all pairs.
        Let's use the standard QAP delta formula.
        
        Let C(p) = sum_{i,j} a[i,j] b[p[i], p[j]]
        
        Delta for swapping i and j:
        Let u = p[i], v = p[j].
        
        The change comes from:
        1. Terms where one index is i or j, and the other is k (k != i, j).
        2. Terms where both indices are i and j.
        
        Old contribution of (i, j) and (j, i):
        a[i,j] b[u,v] + a[j,i] b[v,u]
        
        New contribution of (i, j) and (j, i):
        a[i,j] b[v,u] + a[j,i] b[u,v]
        
        Change in (i,j) and (j,i) terms:
        (a[i,j] b[v,u] + a[j,i] b[u,v]) - (a[i,j] b[u,v] + a[j,i] b[v,u])
        = (a[i,j] - a[j,i]) (b[v,u] - b[u,v])
        
        Now terms with i and k (k != i, j):
        Old: sum_{k != i,j} a[i,k] b[u, p[k]]
        New: sum_{k != i,j} a[i,k] b[v, p[k]]
        Change: sum_{k != i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
        
        Terms with j and k (k != i, j):
        Old: sum_{k != i,j} a[j,k] b[v, p[k]]
        New: sum_{k != i,j} a[j,k] b[u, p[k]]
        Change: sum_{k != i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
        
        Terms with k and i (k != i, j):
        Old: sum_{k != i,j} a[k,i] b[p[k], u]
        New: sum_{k != i,j} a[k,i] b[p[k], v]
        Change: sum_{k != i,j} a[k,i] (b[p[k], v] - b[p[k], u])
        
        Terms with k and j (k != i, j):
        Old: sum_{k != i,j} a[k,j] b[p[k], v]
        New: sum_{k != i,j} a[k,j] b[p[k], u]
        Change: sum_{k != i,j} a[k,j] (b[p[k], u] - b[p[k], v])
        
        Combining the sums over k:
        Sum_{k != i,j} [ a[i,k] b[v, p[k]] - a[i,k] b[u, p[k]] + a[j,k] b[u, p[k]] - a[j,k] b[v, p[k]] + a[k,i] b[p[k], v] - a[k,i] b[p[k], u] + a[k,j] b[p[k], u] - a[k,j] b[p[k], v] ]
        
        = Sum_{k != i,j} [ (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]]) + (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u]) ]
        
        Note that b is symmetric? The problem doesn't state b is symmetric, but QAP instances often are. tai256c is likely symmetric. Let's assume symmetry for speed if possible, but keep general formula for correctness.
        Actually, let's precompute vectors to speed up the delta calculation.
        
        For a fixed p, define:
        r_i = sum_k a[i,k] b[u, p[k]]  ... no, this depends on u.
        
        Let's precompute:
        For each facility i, and each location l, we can't precompute everything.
        
        However, we can precompute:
        S1[i, l] = sum_{k} a[i,k] b[l, p[k]]  (This is expensive to update, but we can recompute or maintain)
        Actually, maintaining S1 is O(n) per swap.
        
        Let's try a different approach: compute the delta directly using precomputed sums.
        Let P = p (current permutation).
        Let u = P[i], v = P[j].
        
        We need:
        Sum_{k != i,j} (a[i,k] - a[j,k]) (b[v, P[k]] - b[u, P[k]])
        + Sum_{k != i,j} (a[k,i] - a[k,j]) (b[P[k], v] - b[P[k], u])
        + (a[i,j] - a[j,i]) (b[v,u] - b[u,v])
        
        Let's define:
        D1[k] = a[i,k] - a[j,k]
        D2[k] = a[k,i] - a[k,j]
        
        Term 1: Sum_{k != i,j} D1[k] * (b[v, P[k]] - b[u, P[k]])
        Term 2: Sum_{k != i,j} D2[k] * (b[P[k], v] - b[P[k], u])
        
        If b is symmetric, b[v, P[k]] = b[P[k], v], so Term 1 and Term 2 are related.
        Assuming symmetry of b:
        Term 1 + Term 2 = Sum_{k != i,j} (D1[k] + D2[k]) * (b[v, P[k]] - b[u, P[k]])
        
        Let's check if b is symmetric for tai256c. Most QAP instances have symmetric distance matrices. I will assume symmetry for speed, but add a check or just use the general form if I can make it fast enough.
        
        To make it fast, I will precompute:
        For each pair (i, j), the delta is:
        Delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v]) + 
                Sum_{k != i,j} [ (a[i,k] - a[j,k] + a[k,i] - a[k,j]) * (b[v, P[k]] - b[u, P[k]]) ]
        
        Let C[k] = a[i,k] - a[j,k] + a[k,i] - a[k,j]
        
        We need Sum_{k != i,j} C[k] * b[v, P[k]] - Sum_{k != i,j} C[k] * b[u, P[k]]
        
        Let S_l = Sum_{k} C[k] * b[l, P[k]]
        Then Sum_{k != i,j} C[k] * b[v, P[k]] = S_v - C[i] * b[v, P[i]] - C[j] * b[v, P[j]]
        Note P[i] = u, P[j] = v.
        So S_v - C[i] * b[v, u] - C[j] * b[v, v]
        Similarly for u: S_u - C[i] * b[u, u] - C[j] * b[u, v]
        
        So Delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v]) + 
                   (S_v - C[i] b[v,u] - C[j] b[v,v]) - (S_u - C[i] b[u,u] - C[j] b[u,v])
        
        This requires computing S_v and S_u, which are sums over all k.
        S_l = Sum_{k} C[k] * b[l, P[k]]
        
        Computing S_l for a specific l takes O(n).
        So each delta evaluation is O(n).
        With n=256, O(n) is fast.
        Number of pairs is n*(n-1)/2 ~ 32k.
        32k * 256 ~ 8 million operations per full pass.
        In Python, this might be slow if done in loops.
        
        Let's vectorize the delta calculation for all pairs (i, j) at once?
        That would be O(n^3) memory/time which is 256^3 ~ 16 million, manageable.
        
        Let's try to vectorize the full 2-opt scan.
        
        For each i, we can compute the delta for all j > i.
        
        Let's stick to a simple ILS with vectorized delta for a few moves per iteration, or just use a greedy local search that is vectorized.
        
        Actually, let's use a simpler, robust approach:
        1. Random start.
        2. Local search: iterate through all pairs (i, j), compute delta, if delta < 0, swap. Repeat until no improvement.
        
        To speed up, we can precompute the "attraction" vectors.
        
        Let's implement a fast local search.
        
        Precompute:
        a_sym = (a + a.T) / 2 ? No, a might not be symmetric.
        b_sym = (b + b.T) / 2 ? Assume b is symmetric.
        
        Let's assume b is symmetric.
        Then Delta(i, j) = (a[i,j] + a[j,i] - a[i,i] - a[j,j]) * (b[v,u] - b[u,v])? No.
        
        Let's use the formula:
        Delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v]) + 
                Sum_{k != i,j} (a[i,k] - a[j,k] + a[k,i] - a[k,j]) * (b[v, P[k]] - b[u, P[k]])
        
        Let's precompute for each i, the vector d_i = a[i, :] + a[:, i] - 2*a[i, :]? No.
        
        Let's just do the O(n^3) vectorized scan if it's fast enough.
        n=256.
        n^3 = 16,777,216.
        NumPy can do this in ~0.1s.
        So we can do many iterations.
        
        Algorithm:
        1. Generate random permutation p.
        2. Compute cost.
        3. Local search:
           - Repeat:
             - Compute delta for all pairs (i, j) with i < j.
             - Find the most negative delta.
             - If delta < 0, swap p[i], p[j].
             - Else break.
        
        This is a steepest descent local search.
        
        To compute delta for all pairs:
        For each i, for each j > i:
        u = p[i], v = p[j]
        C = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        S_v = C @ b[:, v]  (Sum_k C[k] * b[k, v] = Sum_k C[k] * b[v, k] if symmetric)
        S_u = C @ b[:, u]
        Delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v]) + (S_v - C[i]*b[v,u] - C[j]*b[v,v]) - (S_u - C[i]*b[u,u] - C[j]*b[u,v])
        
        This is O(n^3) per iteration.
        
        Let's implement this.
        """
        pass
    
    # Initial solution: random
    p = np.random.permutation(n)
    best_p = p.copy()
    best_cost = cost(p)
    
    # Local search
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Vectorized delta computation for all pairs
        # This is O(n^3)
        # We can do it in chunks or just loop over i
        
        for i in range(n - 1):
            if time.time() > deadline:
                break
                
            u = p[i]
            # Compute C for all j > i
            # C[j] = a[i, :] - a[j, :] + a[:, i] - a[:, j]
            # We want to find the best j > i
            
            # Precompute parts
            ai = a[i, :]
            ai_col = a[:, i]
            
            # For each j, C = ai - a[j, :] + ai_col - a[:, j]
            # S_v = C @ b[:, v]
            # S_u = C @ b[:, u]
            
            # Let's loop over j for simplicity first, optimize later if needed
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                v = p[j]
                
                # Compute C
                C = ai - a[j, :] + ai_col - a[:, j]
                
                # Compute S_v and S_u
                # S_v = sum_k C[k] * b[v, k]
                # S_u = sum_k C[k] * b[u, k]
                
                S_v = np.dot(C, b[v, :])
                S_u = np.dot(C, b[u, :])
                
                # Delta
                term1 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                term2 = (S_v - C[i] * b[v, u] - C[j] * b[v, v]) - (S_u - C[i] * b[u, u] - C[j] * b[u, v])
                delta = term1 + term2
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            # Swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            new_cost = cost(p)
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
            improved = True
        else:
            improved = False
    
    # Add some random restarts / ILS if time permits
    # For now, return best found
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
