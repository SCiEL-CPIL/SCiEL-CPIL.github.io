import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    # Symmetrize just in case, though wil100 is symmetric
    # The cost is sum a[i,j] b[p[i], p[j]].
    # If we symmetrize A and B, does the cost change?
    # sum (A+A^T)[i,j] (B+B^T)[p[i], p[j]]
    # = sum A[i,j]B[p[i],p[j]] + A[j,i]B[p[i],p[j]] + A[i,j]B[p[j],p[i]] + A[j,i]B[p[j],p[i]]
    # If A,B symmetric, this is 4 * original cost? No.
    # Original: sum A[i,j] B[p[i], p[j]].
    # If A,B symmetric, sum A[i,j] B[p[i], p[j]] is the same as sum A[j,i] B[p[j], p[i]].
    # Let's just use the original matrices.
    
    # Heuristic Initial Solution
    # Sort by row sums
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Heavy flows to central locations (small distance sums)
    heavy_idx = np.argsort(-a_sums, kind='stable')
    central_idx = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    best_p = p.copy()
    best_cost = calculate_cost(a, b, best_p)
    
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Multi-start
    restarts = 0
    while time.time() - start_time < time_limit:
        # Perturb or Randomize
        if restarts == 0:
            current_p = best_p.copy()
        else:
            # Random restart or perturbation
            # Try a few random permutations and pick the best initial one?
            # Or just random.
            current_p = np.random.permutation(n).astype(np.int64)
            # Maybe do a quick local search on random?
            # Let's just start local search.
            
        # Local Search
        current_p, current_cost = local_search(a, b, current_p, time.time() - start_time, time_limit)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
            
        restarts += 1
        
        # If we found a very good solution, maybe reduce randomness?
        # For now, just keep restarting.
        
    return best_p.tolist()

def calculate_cost(a, b, p):
    # sum a[i,j] b[p[i], p[j]]
    # Vectorized:
    # b_p = b[np.ix_(p, p)]
    # return np.sum(a * b_p)
    b_p = b[p][:, p]
    return np.sum(a * b_p)

def local_search(a, b, p, elapsed, limit):
    n = len(p)
    current_cost = calculate_cost(a, b, p)
    
    # Precompute M = A @ B[:, p]
    # M[i, loc] = sum_k A[i,k] B[loc, p[k]]
    # Note: B[:, p] is B with columns permuted by p.
    # So B[:, p][loc, k] = B[loc, p[k]].
    # M[i, loc] = sum_k A[i,k] B[loc, p[k]].
    
    while True:
        if time.time() - elapsed > limit - 5.0: # Safety check
            break
            
        # Compute M
        B_perm = b[:, p]
        M = a @ B_perm
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Check all pairs
        # We can vectorize over j for each i
        for i in range(n):
            xi = p[i]
            # Vector of deltas for all j
            # S_ij = M[i, xi] - M[i, p[j]] - M[j, xi] + M[j, p[j]]
            # M[i, p[j]] is vector M[i, :] indexed by p[j]? No.
            # M is n x n. M[i, :] is row i.
            # We need M[i, p[j]] for all j.
            # Let V1 = M[i, :] -> shape (n,)
            # We need V1[p[j]] for all j.
            # p is the permutation. p[j] is the location of facility j.
            # So we want an array D1 where D1[j] = M[i, p[j]].
            # D1 = M[i, :][p] -> M[i, p]
            
            D1 = M[i, :][p] # D1[j] = M[i, p[j]]
            
            # M[j, xi]
            # We want an array D2 where D2[j] = M[j, xi].
            # This is column xi of M.
            D2 = M[:, xi] # D2[j] = M[j, xi]
            
            # M[j, p[j]]
            # Diagonal of M indexed by p?
            # We want D3[j] = M[j, p[j]].
            D3 = M[np.arange(n), p] # D3[j] = M[j, p[j]]
            
            # S_full = M[i, xi] - D1 - D2 + D3
            S_full = M[i, xi] - D1 - D2 + D3
            
            # Corrections for k=i and k=j
            # T_i = (A[i,i] - A[j,i]) * (B[xi, xi] - B[p[j], xi])
            # T_j = (A[i,j] - A[j,j]) * (B[xi, p[j]] - B[p[j], p[j]])
            
            # A[i,i] is scalar.
            # A[j,i] is vector A[:, i] indexed by j? No, A[j,i] for all j is column i of A.
            A_col_i = a[:, i]
            
            # B[xi, xi] is scalar.
            # B[p[j], xi] is vector B[p, xi].
            B_col_xi = b[:, xi] # B_col_xi[loc] = B[loc, xi]
            B_p_xi = B_col_xi[p] # B_p_xi[j] = B[p[j], xi]
            
            # T_i[j] = (a[i,i] - A_col_i[j]) * (b[xi,xi] - B_p_xi[j])
            T_i = (a[i,i] - A_col_i) * (b[xi,xi] - B_p_xi)
            
            # T_j
            # A[i,j] is vector A[i, :] indexed by j? A[i, j] for all j is row i of A.
            A_row_i = a[i, :]
            
            # A[j,j] is vector diag(A) indexed by j?
            # diag_A[j] = A[j,j].
            diag_A = np.diag(a)
            
            # B[xi, p[j]] is vector B[xi, :] indexed by p[j]?
            # B_row_xi = b[xi, :]
            # B_xi_p = B_row_xi[p] -> B_xi_p[j] = B[xi, p[j]]
            B_row_xi = b[xi, :]
            B_xi_p = B_row_xi[p]
            
            # B[p[j], p[j]] is vector diag(B) indexed by p[j]?
            # diag_B = np.diag(b)
            # B_pp = diag_B[p] -> B_pp[j] = B[p[j], p[j]]
            diag_B = np.diag(b)
            B_pp = diag_B[p]
            
            T_j = (A_row_i - diag_A) * (B_xi_p - B_pp)
            
            # Delta = 2 * (S_full - T_i - T_j)
            deltas = 2.0 * (S_full - T_i - T_j)
            
            # Exclude j=i
            deltas[i] = 0 # Or inf, we want min, so set to 0 if no improvement allowed? 
            # We want negative deltas. Set to 0 so it's not picked if no other negative.
            # Actually, if all are positive, we stop.
            
            # Find min
            min_val = np.min(deltas)
            if min_val < best_delta:
                best_delta = min_val
                best_j = np.argmin(deltas)
                best_i = i
                
        if best_delta >= 0:
            break
            
        # Apply swap
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
    return p, current_cost
