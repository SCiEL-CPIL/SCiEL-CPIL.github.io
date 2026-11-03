"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Ensure int64 for precision
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # 1. Initial Solution: Greedy Rank Matching
    # Sort facilities by sum of flows (heaviest first)
    row_sums_a = a.sum(axis=1)
    heavy = np.argsort(-row_sums_a, kind="stable")
    
    # Sort locations by sum of distances (most central/shortest first)
    row_sums_b = b.sum(axis=1)
    central = np.argsort(row_sums_b, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: (a * b[p[:, None], p[None, :]]).sum()
        # b[p[:, None], p[None, :]] creates an (n, n) matrix where M[i,j] = b[p[i], p[j]]
        # This is memory heavy (256x256 is fine) but might be slow if done repeatedly.
        # Better: np.einsum or matrix multiplication trick?
        # C = sum_i sum_j a_ij b_{p_i, p_j}
        # Let B_p be the permuted matrix B. B_p[i,j] = b[p[i], p[j]].
        # Then cost = sum(a * B_p)
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)
        
    current_cost = calc_cost(p)
    
    # Deadline guard
    start_time = time.time()
    time_limit = 115.0  # 115s to be safe
    
    # 2. Local Search: 2-opt (First Improvement)
    # We iterate through all pairs (i, j) and swap if it improves cost.
    # To speed up, we pre-extract rows of A and B?
    # A is fixed. B is fixed.
    
    # Pre-compute B rows as a list of arrays for faster access?
    # b is (n, n). b[k] is row k.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate i from 0 to n-1
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            k1 = p[i]
            # Iterate j from i+1 to n-1
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit:
                    break
                
                k2 = p[j]
                
                # Calculate delta cost for swapping p[i] and p[j]
                # Delta = sum_{u != i,j} (a[i,u] - a[j,u]) * (b[k2, p[u]] - b[k1, p[u]]) 
                #       + a[i,j]*(b[k2,k1] - b[k1,k2]) + a[j,i]*(b[k1,k2] - b[k2,k1])
                
                # Vectorized part:
                # v = b[k2, p] - b[k1, p]
                # diff_a = a[i, :] - a[j, :]
                # dot = np.dot(diff_a, v)
                
                # The dot product includes u=i and u=j.
                # Term for u=i in dot: (a[i,i] - a[j,i]) * (b[k2, p[i]] - b[k1, p[i]])
                #                    = (a[i,i] - a[j,i]) * (b[k2, k1] - b[k1, k1])
                # Term for u=j in dot: (a[i,j] - a[j,j]) * (b[k2, p[j]] - b[k1, p[j]])
                #                    = (a[i,j] - a[j,j]) * (b[k2, k2] - b[k1, k2])
                
                # The formula derived earlier:
                # Delta = sum_{u != i,j} (a[i,u] - a[j,u]) * (b[k2, p[u]] - b[k1, p[u]]) 
                #       + a[i,j]*(b[k2,k1] - b[k1,k2]) + a[j,i]*(b[k1,k2] - b[k2,k1])
                
                # Let's compute the full dot product and then correct for i, j.
                # Full Dot = sum_{u=0}^{n-1} (a[i,u] - a[j,u]) * (b[k2, p[u]] - b[k1, p[u]])
                
                # We want Delta.
                # Note that b[k1, k1] and b[k2, k2] terms might appear.
                
                # Let's just compute the new cost of the affected terms directly to be safe, 
                # or trust the algebra.
                
                # Algebra check:
                # Cost = Sum_{x,y} a[x,y] b[p[x], p[y]]
                # Swap i, j.
                # Terms involving i or j change.
                # New Cost = Sum_{x,y not i,j} ... + Terms(i,j)
                # Old Cost = Sum_{x,y not i,j} ... + Terms(i,j)
                
                # Terms(i,j) in Old:
                # a[i,i]b[k1,k1] + a[i,j]b[k1,k2] + a[j,i]b[k2,k1] + a[j,j]b[k2,k2]
                # + Sum_{u not i,j} (a[i,u]b[k1,p[u]] + a[u,i]b[p[u],k1] + a[j,u]b[k2,p[u]] + a[u,j]b[p[u],k2])
                
                # Terms(i,j) in New:
                # a[i,i]b[k2,k2] + a[i,j]b[k2,k1] + a[j,i]b[k1,k2] + a[j,j]b[k1,k1]
                # + Sum_{u not i,j} (a[i,u]b[k2,p[u]] + a[u,i]b[p[u],k2] + a[j,u]b[k1,p[u]] + a[u,j]b[p[u],k1])
                
                # Delta = New - Old
                # = (a[i,i](b[k2,k2]-b[k1,k1]) + a[j,j](b[k1,k1]-b[k2,k2])
                #    + a[i,j](b[k2,k1]-b[k1,k2]) + a[j,i](b[k1,k2]-b[k2,k1]))
                # + Sum_{u not i,j} [ a[i,u](b[k2,p[u]]-b[k1,p[u]]) 
                #                    + a[u,i](b[p[u],k2]-b[p[u],k1])
                #                    + a[j,u](b[k1,p[u]]-b[k2,p[u]])
                #                    + a[u,j](b[p[u],k1]-b[p[u],k2]) ]
                
                # Note: b[p[u], k] is NOT the same as b[k, p[u]]. B is not symmetric.
                # My previous vectorized formula assumed symmetry or used b[k, p[u]] for all terms.
                # The term a[u,i] b[p[u], k1] involves column k1 of B, not row.
                # So we need b[:, k1] and b[:, k2] as well.
                
                # Let R_k = b[k, :] (Row k)
                # Let C_k = b[:, k] (Column k)
                
                # Sum term for u:
                # a[i,u] (R_k2[u] - R_k1[u]) 
                # + a[u,i] (C_k2[u] - C_k1[u])  -- Wait, b[p[u], k2] is C_k2[p[u]].
                # The index is p[u]. So it's C_k2[p[u]] - C_k1[p[u]].
                
                # So we need:
                # V_R = b[k2, p] - b[k1, p]  (Vector of length n, indexed by u)
                # V_C = b[p, k2] - b[p, k1]  (Vector of length n, indexed by u)
                
                # Delta_sum = Sum_{u != i,j} ( a[i,u] * V_R[u] + a[u,i] * V_C[u] 
                #                              - a[j,u] * V_R[u] - a[u,j] * V_C[u] )
                # = Sum_{u != i,j} ( (a[i,u] - a[j,u]) * V_R[u] + (a[u,i] - a[u,j]) * V_C[u] )
                
                # Let's compute full sums and subtract i, j.
                # S_R = Sum_{u} (a[i,u] - a[j,u]) * V_R[u] = np.dot(a[i, :] - a[j, :], V_R)
                # S_C = Sum_{u} (a[u,i] - a[u,j]) * V_C[u] = np.dot(a[:, i] - a[:, j], V_C)
                
                # Full Delta = S_R + S_C + Direct Terms
                # But S_R and S_C include u=i and u=j.
                # We need to subtract the u=i and u=j contributions from S_R and S_C.
                
                # u=i term in S_R: (a[i,i] - a[j,i]) * V_R[i]
                # V_R[i] = b[k2, p[i]] - b[k1, p[i]] = b[k2, k1] - b[k1, k1]
                
                # u=j term in S_R: (a[i,j] - a[j,j]) * V_R[j]
                # V_R[j] = b[k2, p[j]] - b[k1, p[j]] = b[k2, k2] - b[k1, k2]
                
                # u=i term in S_C: (a[i,i] - a[i,j]) * V_C[i]
                # V_C[i] = b[p[i], k2] - b[p[i], k1] = b[k1, k2] - b[k1, k1]
                
                # u=j term in S_C: (a[j,i] - a[j,j]) * V_C[j]
                # V_C[j] = b[p[j], k2] - b[p[j], k1] = b[k2, k2] - b[k2, k1]
                
                # Direct Terms (from diagonal/swap of i,j positions themselves in the cost sum):
                # a[i,i](b[k2,k2]-b[k1,k1]) + a[j,j](b[k1,k1]-b[k2,k2])
                # + a[i,j](b[k2,k1]-b[k1,k2]) + a[j,i](b[k1,k2]-b[k2,k1])
                
                # Let's combine everything.
                
                V_R = b[k2, p] - b[k1, p]
                V_C = b[p, k2] - b[p, k1]
                
                diff_a_row = a[i, :] - a[j, :]
                diff_a_col = a[:, i] - a[:, j]
                
                S_R = np.dot(diff_a_row, V_R)
                S_C = np.dot(diff_a_col, V_C)
                
                # Subtract u=i, j from S_R
                term_i_R = (a[i, i] - a[j, i]) * (b[k2, k1] - b[k1, k1])
                term_j_R = (a[i, j] - a[j, j]) * (b[k2, k2] - b[k1, k2])
                S_R_adj = S_R - term_i_R - term_j_R
                
                # Subtract u=i, j from S_C
                term_i_C = (a[i, i] - a[i, j]) * (b[k1, k2] - b[k1, k1])
                term_j_C = (a[j, i] - a[j, j]) * (b[k2, k2] - b[k2, k1])
                S_C_adj = S_C - term_i_C - term_j_C
                
                # Direct terms
                direct = (a[i, i] * (b[k2, k2] - b[k1, k1]) 
                         + a[j, j] * (b[k1, k1] - b[k2, k2])
                         + a[i, j] * (b[k2, k1] - b[k1, k2])
                         + a[j, i] * (b[k1, k2] - b[k2, k1]))
                
                delta = S_R_adj + S_C_adj + direct
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # Restart or continue? First improvement usually continues, 
                    # but restarting the inner loop for i might be better to catch cascading effects.
                    # However, for speed, we can just continue to next j.
                    # A full restart of the i-loop is safer for convergence.
                    # Let's break and restart i-loop for simplicity and robustness.
                    break
            
            if improved:
                # Restart i-loop
                # To avoid infinite loops in case of numerical issues (shouldn't happen with int),
                # we just let the while loop handle it.
                # But we need to reset i to 0.
                # The for loop will end, and the while loop will check improved.
                # So we break the inner j loop, and then break the outer i loop?
                # No, if we break j, we exit j loop. Then we exit i loop?
                # If we just break j, we go to next i.
                # If we want to restart, we should break i as well.
                # Let's use a flag.
                pass

    return p.tolist()
# EVOLVE-BLOCK-END
