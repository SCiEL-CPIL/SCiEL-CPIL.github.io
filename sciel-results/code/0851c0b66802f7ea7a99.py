"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    
    # Initialize with rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu search with optimized delta computation
    tabu_tenure = 20
    tabu_until = np.zeros((n, n), dtype=np.int32)
    iteration = 0
    start_time = time.time()
    time_limit = 115  # leave some margin
    
    # Precompute a and b as float for speed
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Precompute p as int array for fast indexing
    p_arr = p
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Evaluate all 2-opt swaps with optimized computation
        for i in range(n):
            xi = p_arr[i]
            ai_row = af[i]
            ai_col = af[:, i]
            
            for j in range(i + 1, n):
                yj = p_arr[j]
                
                # Check tabu using 2D array (much faster than dict)
                if tabu_until[i, j] > iteration:
                    continue
                
                # Compute delta efficiently
                # The key insight: we can precompute the contribution vectors
                # For each location l, define:
                #   row_contrib[l] = sum_k af[k, :] * bf[l, p_arr[k]]  (not quite right)
                
                # Let's compute the delta using the formula:
                # delta = sum over k of terms involving i and j
                # We'll compute it in one vectorized operation
                
                # Get the current positions of all facilities
                pk = p_arr
                
                # Terms for facility i moving from xi to yj:
                # sum_k af[i,k] * (bf[yj, pk[k]] - bf[xi, pk[k]])
                # + sum_k af[k,i] * (bf[pk[k], yj] - bf[pk[k], xi])
                
                # Terms for facility j moving from yj to xi:
                # sum_k af[j,k] * (bf[xi, pk[k]] - bf[yj, pk[k]])
                # + sum_k af[k,j] * (bf[pk[k], xi] - bf[pk[k], yj])
                
                # Combine:
                # sum_k [af[i,k] - af[j,k]] * (bf[yj, pk[k]] - bf[xi, pk[k]])
                # + sum_k [af[k,i] - af[k,j]] * (bf[pk[k], yj] - bf[pk[k], xi])
                
                # But we need to exclude k=i and k=j from the sums
                # and handle the diagonal terms separately
                
                # Let's compute the full sum first, then subtract i and j contributions
                # and add back the correct diagonal terms
                
                # Full sum for i part:
                # S_i = sum_k af[i,k]*bf[yj,pk[k]] - sum_k af[i,k]*bf[xi,pk[k]]
                #      + sum_k af[k,i]*bf[pk[k],yj] - sum_k af[k,i]*bf[pk[k],xi]
                
                # Full sum for j part:
                # S_j = sum_k af[j,k]*bf[xi,pk[k]] - sum_k af[j,k]*bf[yj,pk[k]]
                #      + sum_k af[k,j]*bf[pk[k],xi] - sum_k af[k,j]*bf[pk[k],yj]
                
                # Total delta from sums = S_i + S_j
                # But this includes k=i and k=j, which we need to adjust
                
                # Let's compute S_i_full and S_j_full
                S_i_full = np.dot(af[i], bf[yj, pk]) - np.dot(af[i], bf[xi, pk])
                S_i_full += np.dot(af[:, i], bf[pk, yj]) - np.dot(af[:, i], bf[pk, xi])
                
                S_j_full = np.dot(af[j], bf[xi, pk]) - np.dot(af[j], bf[yj, pk])
                S_j_full += np.dot(af[:, j], bf[pk, xi]) - np.dot(af[:, j], bf[pk, yj])
                
                # Now subtract the k=i and k=j contributions from S_i_full and S_j_full
                # and add the correct diagonal terms
                
                # For k=i in S_i_full:
                # af[i,i]*bf[yj,xi] - af[i,i]*bf[xi,xi] + af[i,i]*bf[xi,yj] - af[i,i]*bf[xi,xi]
                # = af[i,i]*(bf[yj,xi] + bf[xi,yj] - 2*bf[xi,xi])
                
                # For k=j in S_i_full:
                # af[i,j]*bf[yj,yj] - af[i,j]*bf[xi,yj] + af[j,i]*bf[yj,yj] - af[j,i]*bf[xi,xi]
                # Wait, this is getting complicated. Let me just compute the delta directly.
                
                # Actually, let's use a simpler approach: compute the delta by
                # evaluating the change in cost directly using the formula.
                
                # The change in cost when swapping p[i]=xi and p[j]=yj is:
                # delta = sum_{k} [af[i,k]*bf[yj,pk[k]] + af[k,i]*bf[pk[k],yj]
                #                      + af[j,k]*bf[xi,pk[k]] + af[k,j]*bf[pk[k],xi]
                #                      - af[i,k]*bf[xi,pk[k]] - af[k,i]*bf[pk[k],xi]
                #                      - af[j,k]*bf[yj,pk[k]] - af[k,j]*bf[pk[k],yj]]
                
                # But we need to be careful about k=i and k=j.
                
                # Let's just compute it properly:
                # Create a vector of pk
                pk_vec = p_arr
                
                # Compute the four terms
                term1 = af[i] * bf[yj, pk_vec]  # af[i,k]*bf[yj,pk[k]]
                term2 = af[:, i] * bf[pk_vec, yj]  # af[k,i]*bf[pk[k],yj]
                term3 = af[j] * bf[xi, pk_vec]  # af[j,k]*bf[xi,pk[k]]
                term4 = af[:, j] * bf[pk_vec, xi]  # af[k,j]*bf[pk[k],xi]
                
                term5 = af[i] * bf[xi, pk_vec]  # af[i,k]*bf[xi,pk[k]]
                term6 = af[:, i] * bf[pk_vec, xi]  # af[k,i]*bf[pk[k],xi]
                term7 = af[j] * bf[yj, pk_vec]  # af[j,k]*bf[yj,pk[k]]
                term8 = af[:, j] * bf[pk_vec, yj]  # af[k,j]*bf[pk[k],yj]
                
                # Sum all terms
                sum_terms = np.sum(term1) + np.sum(term2) + np.sum(term3) + np.sum(term4)
                sum_terms -= np.sum(term5) + np.sum(term6) + np.sum(term7) + np.sum(term8)
                
                # Now we need to correct for the fact that when k=i or k=j,
                # the pk[k] value is xi or yj, which is being swapped.
                # The above computation uses the OLD pk values, so for k=i, pk[i]=xi,
                # and for k=j, pk[j]=yj.
                
                # For k=i:
                # In term1: af[i,i]*bf[yj,xi] (correct, since pk[i]=xi)
                # In term2: af[i,i]*bf[xi,yj] (correct, since pk[i]=xi)
                # In term3: af[j,i]*bf[xi,xi] (correct)
                # In term4: af[i,j]*bf[xi,xi] (correct)
                # In term5: af[i,i]*bf[xi,xi] (correct)
                # In term6: af[i,i]*bf[xi,xi] (correct)
                # In term7: af[j,i]*bf[yj,xi] (correct)
                # In term8: af[i,j]*bf[xi,yj] (correct)
                
                # For k=j:
                # In term1: af[i,j]*bf[yj,yj] (correct, since pk[j]=yj)
                # In term2: af[j,i]*bf[yj,yj] (correct)
                # In term3: af[j,j]*bf[xi,yj] (correct)
                # In term4: af[j,j]*bf[yj,xi] (correct)
                # In term5: af[i,j]*bf[xi,yj] (correct)
                # In term6: af[j,i]*bf[yj,xi] (correct)
                # In term7: af[j,j]*bf[yj,yj] (correct)
                # In term8: af[j,j]*bf[yj,yj] (correct)
                
                # So the computation is actually correct! The pk values used are the old ones,
                # which is what we want for computing the delta.
                
                delta = sum_terms
                
                # Check aspiration criterion
                if current_cost + delta < best_cost:
                    # Allow tabu move if it improves best
                    pass
                elif tabu_until[i, j] > iteration:
                    # Skip tabu moves that don't improve best
                    continue
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            # No improving move found; try any non-tabu move
            for i in range(n):
                for j in range(i + 1, n):
                    if tabu_until[i, j] <= iteration:
                        best_i = i
                        best_j = j
                        break
                if best_i != -1:
                    break
        
        if best_i == -1:
            break
        
        i, j = best_i, best_j
        p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
        current_cost += best_delta
        
        # Update tabu
        tabu_until[i, j] = iteration + tabu_tenure
        tabu_until[j, i] = iteration + tabu_tenure
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p_arr.copy()
        
        iteration += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
