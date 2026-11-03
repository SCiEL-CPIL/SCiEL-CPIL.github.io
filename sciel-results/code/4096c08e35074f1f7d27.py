# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115
    
    # 2-Opt search
    # We will use a simple list for tabu to avoid dict overhead
    tabu_moves = [] # list of (i, j, iter)
    tabu_tenure = 20
    iteration = 0
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_swap = None
        
        # Clean old tabu moves
        if len(tabu_moves) > 1000:
            tabu_moves = [(i, j, t) for i, j, t in tabu_moves if t + tabu_tenure > iteration]
            
        # Evaluate swaps
        for i in range(n):
            x = p[i]
            # Vectorized computation for all j > i?
            # Let's stick to the loop but optimize the inner sum
            for j in range(i + 1, n):
                y = p[j]
                if x == y:
                    continue
                
                # Check tabu
                is_tabu = False
                for ti, tj, tt in tabu_moves:
                    if (ti == i and tj == j) or (ti == j and tj == i):
                        if tt + tabu_tenure > iteration:
                            is_tabu = True
                            break
                            
                # If tabu, only allow if improving best
                # We compute delta first
                
                # Compute delta
                # Mask for k != i, j
                # Instead of creating mask every time, we can use the full sum and subtract i,j terms
                # Sum_k (a[i,k]*(bf[y,p[k]] - bf[x,p[k]]) + a[k,i]*(bf[p[k],y] - bf[p[k],x]) + ...)
                
                # Vectorized over all k
                # diff_yx_row = bf[y, :] - bf[x, :]
                # diff_yx_col = bf[:, y] - bf[:, x]
                # This is O(n).
                
                # Let's compute the full sum over all k, then subtract k=i and k=j terms.
                # Full sum S = sum_k (af[i,k]*(bf[y,p[k]] - bf[x,p[k]]) + af[k,i]*(bf[p[k],y] - bf[p[k],x])
                #               + af[j,k]*(bf[x,p[k]] - bf[y,p[k]]) + af[k,j]*(bf[p[k],x] - bf[p[k],y]))
                
                # This is 4 dot products.
                # t1 = af[i, :] @ (bf[y, p] - bf[x, p])
                # t2 = af[:, i] @ (bf[p, y] - bf[p, x])
                # t3 = af[j, :] @ (bf[x, p] - bf[y, p])
                # t4 = af[:, j] @ (bf[p, x] - bf[p, y])
                
                # Note: bf[y, p] is a vector of length n.
                # af[i, :] is a vector of length n.
                
                # Let's precompute vectors?
                # p is fixed for this iteration.
                # bf_y_p = bf[y, p]
                # bf_x_p = bf[x, p]
                # bf_p_y = bf[p, y]
                # bf_p_x = bf[p, x]
                
                # We can compute these outside the j loop?
                # No, y depends on j.
                
                # So for each j, we compute 4 dot products of length n.
                # This is 4 * 150 multiplications/additions.
                # 11k swaps * 600 ops = 6.6M ops.
                # In NumPy, this is fast.
                
                diff_yx_row = bf[y, p] - bf[x, p]
                diff_yx_col = bf[p, y] - bf[p, x]
                
                t1 = np.dot(af[i, :], diff_yx_row)
                t2 = np.dot(af[:, i], diff_yx_col)
                t3 = np.dot(af[j, :], -diff_yx_row) # af[j,k]*(bf[x,p[k]] - bf[y,p[k]])
                t4 = np.dot(af[:, j], -diff_yx_col) # af[k,j]*(bf[p[k],x] - bf[p[k],y])
                
                sum_k = t1 + t2 + t3 + t4
                
                # Subtract k=i and k=j terms from sum_k
                # Term k=i:
                # af[i,i]*(bf[y,p[i]] - bf[x,p[i]]) + af[i,i]*(bf[p[i],y] - bf[p[i],x]) + af[j,i]*(bf[x,p[i]] - bf[y,p[i]]) + af[i,j]*(bf[p[i],x] - bf[p[i],y])
                # p[i] = x
                # bf[y,x] - bf[x,x]
                # bf[x,y] - bf[x,x]
                # bf[x,x] - bf[y,x]
                # bf[x,x] - bf[x,y]
                
                # This is getting messy. Let's just compute the sum for k != i, j directly using masks.
                # It's safer.
                
                # Create mask
                # mask = np.ones(n, dtype=bool); mask[i]=False; mask[j]=False
                # This allocation is expensive.
                
                # Alternative: Compute full sum and subtract specific terms.
                # Full sum S includes k=i and k=j.
                # S = sum_{k=0}^{n-1} ...
                # We want S_excl = S - term_i - term_j.
                
                # term_i (k=i):
                # af[i,i]*(bf[y,x] - bf[x,x]) + af[i,i]*(bf[x,y] - bf[x,x]) + af[j,i]*(bf[x,x] - bf[y,x]) + af[i,j]*(bf[x,x] - bf[x,y])
                
                # term_j (k=j):
                # af[i,j]*(bf[y,y] - bf[x,y]) + af[j,i]*(bf[y,y] - bf[x,y]) + af[j,j]*(bf[x,y] - bf[y,y]) + af[j,j]*(bf[x,y] - bf[y,y])
                # Wait, p[j] = y.
                # bf[y, p[j]] = bf[y, y]
                # bf[x, p[j]] = bf[x, y]
                # bf[p[j], y] = bf[y, y]
                # bf[p[j], x] = bf[y, x]
                
                # Let's compute these manually.
                
                # term_i:
                # a_ii * (bf[y,x] - bf[x,x]) + a_ii * (bf[x,y] - bf[x,x])
                # + a_ji * (bf[x,x] - bf[y,x]) + a_ij * (bf[x,x] - bf[x,y])
                
                # term_j:
                # a_ij * (bf[y,y] - bf[x,y]) + a_ji * (bf[y,y] - bf[x,y])
                # + a_jj * (bf[x,y] - bf[y,y]) + a_jj * (bf[x,y] - bf[y,y])
                # Note: bf[x,y] - bf[y,y] is same as -(bf[y,y] - bf[x,y])
                
                # This is 8 lookups.
                
                # Let's do it.
                a_ii = af[i, i]
                a_ji = af[j, i]
                a_ij = af[i, j]
                a_jj = af[j, j]
                
                b_yx = bf[y, x]
                b_xy = bf[x, y]
                b_xx = bf[x, x]
                b_yy = bf[y, y]
                
                term_i = a_ii * (b_yx - b_xx) + a_ii * (b_xy - b_xx) + a_ji * (b_xx - b_yx) + a_ij * (b_xx - b_xy)
                term_j = a_ij * (b_yy - b_xy) + a_ji * (b_yy - b_xy) + a_jj * (b_xy - b_yy) + a_jj * (b_xy - b_yy)
                
                sum_k = sum_k - term_i - term_j
                
                # Diagonal terms (i,j interaction)
                diag_old = af[i, j] * bf[x, y] + af[j, i] * bf[y, x]
                diag_new = af[i, j] * bf[y, x] + af[j, i] * bf[x, y]
                
                delta = diag_new - diag_old + sum_k
                
                # Aspiration
                if is_tabu:
                    if current_cost + delta >= best_cost:
                        continue
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (i, j)
        
        if best_swap is None:
            # No improving move, break or perturb
            break
        
        i, j = best_swap
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        tabu_moves.append((i, j, iteration))
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        iteration += 1
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
