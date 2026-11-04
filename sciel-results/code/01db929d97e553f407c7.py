import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 110  # 10s safety margin
    
    # 1. Initial Heuristic: Rank Matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_sums, kind='stable')
    central = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute diagonal
    diag_a = np.diag(a)
    diag_b = np.diag(b)
    
    # Helper to compute cost
    def compute_cost(p):
        return np.einsum('ij,ij->', a, b[np.ix_(p, p)])
    
    best_p = p.copy()
    best_cost = compute_cost(p)
    
    # Initialize Q
    # Q[i, l] = sum_{k != i} a[i, k] * b[l, p[k]]
    B_perm = b[:, p] # B_perm[l, k] = b[l, p[k]]
    
    # Q = a @ B_perm
    # But we need to exclude k=i term: a[i, i] * b[l, p[i]]
    # Term is a[i, i] * B_perm[l, i]
    # So Q = a @ B_perm - diag(a)[:, np.newaxis] * B_perm
    Q = a @ B_perm
    Q -= diag_a[:, np.newaxis] * B_perm
    
    # Local Search Function
    def local_search(p, Q, cost):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Find best swap
            for i in range(n):
                u = p[i]
                a_ii = diag_a[i]
                b_u_u = diag_b[u]
                
                # deltas for all j
                # delta(i, j) = (a_ii - a_jj) * (b_pj_pj - b_u_u) + 2 * (Q[i, p[j]] - Q[i, u])
                a_jj = diag_a
                b_pj_pj = diag_b[p]
                
                term1 = (a_ii - a_jj) * (b_pj_pj - b_u_u)
                term2 = 2 * (Q[i, p] - Q[i, u])
                
                deltas = term1 + term2
                
                # Exclude j=i (swap with self is 0)
                deltas[i] = 0 # or large positive
                
                min_idx = np.argmin(deltas)
                min_val = deltas[min_idx]
                
                if min_val < best_delta:
                    best_delta = min_val
                    best_i = i
                    best_j = min_idx
            
            if best_delta < 0:
                # Apply swap
                i, j = best_i, best_j
                u, v = p[i], p[j]
                
                # Update cost
                cost += best_delta
                
                # Update p
                p[i], p[j] = v, u
                
                # Update Q efficiently
                # Q[i, l] += a[i, j] * (b[l, u] - b[l, v])
                # Q[j, l] += a[j, i] * (b[l, v] - b[l, u])
                diff_b_u_v = b[:, u] - b[:, v]
                
                Q[i, :] += a[i, j] * diff_b_u_v
                Q[j, :] += a[j, i] * (-diff_b_u_v)
                
                improved = True
                
        return p, cost

    # ILS / SA Loop
    current_p = p.copy()
    current_Q = Q.copy()
    current_cost = best_cost
    
    while time.time() < deadline:
        # Local Search
        current_p, current_cost = local_search(current_p, current_Q, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
        
        # Perturb
        # Random 5 swaps
        for _ in range(5):
            if time.time() > deadline:
                break
            i, j = np.random.choice(n, 2, replace=False)
            u, v = current_p[i], current_p[j]
            
            # Compute delta for this specific swap
            delta = (diag_a[i] - diag_a[j]) * (diag_b[v] - diag_b[u]) + 2 * (current_Q[i, v] - current_Q[i, u])
            
            current_cost += delta
            
            # Update p
            current_p[i], current_p[j] = v, u
            
            # Update Q efficiently
            diff_b_u_v = b[:, u] - b[:, v]
            
            current_Q[i, :] += a[i, j] * diff_b_u_v
            current_Q[j, :] += a[j, i] * (-diff_b_u_v)
            
    return best_p.tolist()
