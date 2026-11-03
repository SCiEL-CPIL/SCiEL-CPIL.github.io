"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # 10s safety margin before 120s limit
    
    # Initial solution: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(perm):
        return np.sum(a * b[np.ix_(perm, perm)])
    
    best_cost = compute_cost(p)
    best_p = p.copy()
    
    # 2-opt local search with efficient delta computation
    def two_opt_search(perm, current_cost, max_time):
        n = len(perm)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() >= max_time:
                    break
                pi = perm[i]
                b_pi = b[pi]
                row_i = a[i]
                col_i = a[:, i]
                
                for j in range(i + 1, n):
                    if time.time() >= max_time:
                        break
                    
                    pj = perm[j]
                    b_pj = b[pj]
                    row_j = a[j]
                    col_j = a[:, j]
                    
                    # Compute delta for swapping positions i and j
                    # delta = sum_{k != i,j} [ a[i,k](b[pj,pk]-b[pi,pk]) + a[k,i](b[pk,pj]-b[pk,pi]) 
                    #                     + a[j,k](b[pi,pk]-b[pj,pk]) + a[k,j](b[pk,pi]-b[pk,pj]) ]
                    #       + a[i,j](b[pi,pj]-b[pj,pi]) + a[j,i](b[pj,pi]-b[pi,pj])
                    
                    # Vectorized sum for k != i, j
                    # We can compute the full sum and subtract i and j terms
                    # Sum_i = row_i @ (b_pj - b_pi) + col_i @ (b_pj - b_pi)  -- Wait, b is not symmetric necessarily?
                    # b[pj, pk] is b[pj, perm[k]].
                    # Let's stick to the O(n) loop for correctness and speed in Python is acceptable for n=256 if optimized.
                    
                    # Optimization: Precompute diff vectors
                    # diff_b_row = b_pj - b_pi  (Length n, indexed by location)
                    # But we need b[pj, perm[k]] which is b_pj[perm[k]]
                    # So we need b_pj[perm] - b_pi[perm]
                    
                    diff_b_row = b_pj[perm] - b_pi[perm]
                    diff_b_col = b[:, pj][perm] - b[:, pi][perm] # b[pk, pj] - b[pk, pi]
                    
                    # Sum over all k
                    sum_i = np.dot(row_i, diff_b_row) + np.dot(col_i, diff_b_col)
                    sum_j = np.dot(row_j, -diff_b_row) + np.dot(col_j, -diff_b_col)
                    
                    # Subtract terms for k=i and k=j
                    # For k=i:
                    # term_i_k_i = a[i,i](b[pj,pi]-b[pi,pi]) + a[i,i](b[pi,pj]-b[pi,pi])? No.
                    # Let's just compute the pair terms separately and subtract the k=i and k=j contributions from the vectorized sum.
                    
                    # Contribution of k=i to sum_i:
                    # a[i,i] * (b[pj, pi] - b[pi, pi]) + a[i,i] * (b[pi, pj] - b[pi, pi])
                    # Wait, col_i is a[:, i]. So a[k,i] with k=i is a[i,i].
                    # b[pk, pj] with k=i is b[pi, pj]. b[pk, pi] with k=i is b[pi, pi].
                    # So term_k_i_i = a[i,i] * (b[pi, pj] - b[pi, pi])
                    # And row_i part: a[i,i] * (b[pj, pi] - b[pi, pi])
                    
                    # Contribution of k=j to sum_i:
                    # a[i,j] * (b[pj, pj] - b[pi, pj]) + a[j,i] * (b[pj, pj] - b[pj, pi])
                    
                    # This is getting messy. The O(n) loop is safer and n=256 is small.
                    delta = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        pk = perm[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                        delta += a[k, i] * (b[pk, pj] - b[pk, pi])
                        delta += a[j, k] * (b[pi, pk] - b[pj, pk])
                        delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                    
                    # i-j interaction
                    delta += a[i, j] * (b[pi, pj] - b[pj, pi])
                    delta += a[j, i] * (b[pj, pi] - b[pi, pj])
                    
                    if delta < 0:
                        perm[i], perm[j] = perm[j], perm[i]
                        current_cost += delta
                        improved = True
        
        return current_cost, perm
    
    # Run 2-opt with time limit
    max_time = deadline - 5.0
    cost, p = two_opt_search(p, best_cost, max_time)
    
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
