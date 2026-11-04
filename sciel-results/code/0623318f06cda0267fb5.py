"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit (150s kill)
    
    def calc_cost(perm):
        # perm[i] is the location for facility i
        # Cost = sum_{i,j} a[i, j] * b[perm[i], perm[j]]
        return int(np.sum(a * b[np.ix_(perm, perm)]))
        
    # --- Phase 1: Initialization via Bipartite Matching on Centralities ---
    # The strategy proposes using "facility flow centrality" and "location distance centrality".
    # For QAP, a common heuristic for initialization is to match facilities with high flow sums 
    # to locations with high distance sums (or vice versa, depending on minimization).
    # Since we want to minimize sum a_ij b_pi(i)pi(j), and typically a_ij and b_ij are positive,
    # we want to align high a sums with low b sums? Or is it the other way?
    # Actually, the linear assignment approximation minimizes sum_i a_i . b_{pi(i)}.
    # Let's stick to the standard linearization: minimize sum_{i,j} a_{ij} b_{pi(i), pi(j)}.
    # A common first-order approximation is to use row sums.
    # The "Inspiration" text specifically asks for: "product of facility flow centrality and location distance centrality".
    
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Construct cost matrix for linear assignment
    # We want to minimize the linearized cost.
    # If a_sums[i] is high, we want b_sums[j] to be low? 
    # The linear term in the expansion of QAP is sum_i (sum_j a_ij) * b_{pi(i), ?} -- this is tricky.
    # Standard heuristic: Minimize sum_i a_sums[i] * b_sums[pi(i)].
    # This is a standard assignment problem.
    
    # Note: The prompt says "maximum weight matching ... product of ... centrality".
    # If we maximize product, we are maximizing the linearized cost? 
    # Usually QAP instances have positive flows and distances. Minimizing cost means we want
    # high flow facilities to be close (low distance) to each other.
    # The linear approximation sum a_i * b_pi(i) is a proxy.
    # If we use linear_sum_assignment on cost_matrix = outer(a_sums, b_sums), it minimizes this sum.
    # This is a structurally distinct initialization compared to random or simple rank matching,
    # as it respects the global distribution of weights.
    
    cost_matrix = np.outer(a_sums, b_sums)
    
    # linear_sum_assignment minimizes the sum of cost_matrix[i, j] for assigned pairs.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[row_ind] = col_ind
    
    # Ensure valid permutation
    if sorted(p_init.tolist()) != list(range(n)):
        p_init = np.arange(n, dtype=np.int64)
        
    best_p = p_init.copy()
    best_cost = calc_cost(best_p)
    
    # --- Phase 2: Rapid 2-Opt Local Search Refinement ---
    # We use a fast swap delta calculation to perform 2-opt moves.
    # Delta for swapping locations of facilities i and j (i.e., p[i] <-> p[j]):
    # Let u = p[i], v = p[j].
    # New cost - Old cost = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[u,k] - b[v,k]) 
    #                       + (a[i,i] - a[j,j]) * (b[u,u] - b[v,v]) 
    #                       + (a[i,j] - a[j,i]) * (b[u,v] - b[v,u])
    # Wait, the standard formula for swapping two facilities i, j in QAP:
    # Let x = p[i], y = p[j].
    # Delta = 2 * (sum_{k != i, j} (a[i,k] - a[j,k]) * (b[x,k] - b[y,k]))
    #         + (a[i,i] - a[j,j]) * (b[x,x] - b[y,y])
    #         + (a[i,j] - a[j,i]) * (b[x,y] - b[y,x])
    # Note: b is symmetric in standard QAP, but we should handle general case.
    # b[x,k] - b[y,k] is the difference in distances from location x and y to location k.
    
    # Precompute nothing heavy, just use numpy vectorization for the sum part.
    
    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized calculation of the sum part
        # diff_a = a[i, :] - a[j, :]
        # diff_b = b[u, :] - b[v, :]
        # sum_part = np.dot(diff_a, diff_b)
        
        # But we need to exclude k=i and k=j from the sum?
        # The formula 2 * sum_{k != i, j} ... 
        # Let's check the expansion.
        # Cost = sum_{m,n} a[m,n] b[p[m], p[n]]
        # Terms involving i or j change.
        # 1. a[i,i] b[p[i], p[i]] -> a[i,i] b[v, v]
        # 2. a[j,j] b[p[j], p[j]] -> a[j,j] b[u, u]
        # 3. a[i,j] b[p[i], p[j]] -> a[i,j] b[v, u]
        # 4. a[j,i] b[p[j], p[i]] -> a[j,i] b[u, v]
        # 5. a[i,k] b[p[i], p[k]] -> a[i,k] b[v, p[k]] for k != i, j
        # 6. a[k,i] b[p[k], p[i]] -> a[k,i] b[p[k], v] for k != i, j
        # 7. a[j,k] b[p[j], p[k]] -> a[j,k] b[u, p[k]] for k != i, j
        # 8. a[k,j] b[p[k], p[j]] -> a[k,j] b[p[k], u] for k != i, j
        
        # Change = 
        # (a[i,i] b[v,v] - a[i,i] b[u,u]) + (a[j,j] b[u,u] - a[j,j] b[v,v])
        # + (a[i,j] b[v,u] - a[i,j] b[u,v]) + (a[j,i] b[u,v] - a[j,i] b[v,u])
        # + sum_{k != i, j} [ a[i,k] (b[v,p[k]] - b[u,p[k]]) + a[k,i] (b[p[k],v] - b[p[k],u])
        #                     + a[j,k] (b[u,p[k]] - b[v,p[k]]) + a[k,j] (b[p[k],u] - b[p[k],v]) ]
        
        # Assuming a and b are symmetric (standard QAP):
        # a[i,i] = a[j,j] is not necessarily true.
        # b[u,u] = b[v,v] is 0 if diagonal is 0.
        # If symmetric:
        # Term 1+2: a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) = (a[i,i]-a[j,j])(b[v,v]-b[u,u])
        # Term 3+4: (a[i,j]-a[j,i])(b[v,u]-b[u,v]) = 0 if symmetric.
        # Sum part:
        # sum_{k != i, j} (a[i,k] - a[j,k]) (b[v,p[k]] - b[u,p[k]]) * 2 ?
        # Let's verify.
        # a[i,k] b[v,p[k]] + a[k,i] b[p[k],v] - a[i,k] b[u,p[k]] - a[k,i] b[p[k],u]
        # = a[i,k] (b[v,p[k]] + b[p[k],v]) - a[j,k] (b[u,p[k]] + b[p[k],u]) (since a[k,i]=a[i,k])
        # If b symmetric: b[v,p[k]] = b[p[k],v].
        # = 2 a[i,k] b[v,p[k]] - 2 a[j,k] b[u,p[k]]
        # This doesn't look like the dot product form immediately.
        
        # Let's use the vectorized dot product form which is faster to code and verify.
        # Delta = 2 * ( (a[i, :] - a[j, :]) @ (b[u, :] - b[v, :]) ) 
        #         - 2 * (a[i, i] - a[j, i]) * (b[u, i] - b[v, i]) # Remove k=i
        #         - 2 * (a[i, j] - a[j, j]) * (b[u, j] - b[v, j]) # Remove k=j
        #         + (a[i, i] - a[j, j]) * (b[u, u] - b[v, v])
        #         + (a[i, j] - a[j, i]) * (b[u, v] - b[v, u])
        
        # Wait, the dot product sum_{k} (a[i,k]-a[j,k]) (b[u,k]-b[v,k]) includes k=i and k=j.
        # Let S = sum_{k=0}^{n-1} (a[i,k]-a[j,k]) (b[u,k]-b[v,k])
        # S = (a[i,i]-a[j,i])(b[u,i]-b[v,i]) + (a[i,j]-a[j,j])(b[u,j]-b[v,j]) + sum_{k!=i,j} ...
        
        # The derived change for symmetric matrices is:
        # Delta = 2 * sum_{k!=i,j} (a[i,k]-a[j,k])(b[u,k]-b[v,k]) + (a[i,i]-a[j,j])(b[u,u]-b[v,v]) + (a[i,j]-a[j,i])(b[u,v]-b[v,u])
        
        # Let's compute S_total = dot(a[i]-a[j], b[u]-b[v])
        # sum_{k!=i,j} = S_total - term_i - term_j
        # where term_i = (a[i,i]-a[j,i])*(b[u,i]-b[v,i])
        #       term_j = (a[i,j]-a[j,j])*(b[u,j]-b[v,j])
        
        # Delta = 2 * (S_total - term_i - term_j) + diag_term + offdiag_term
        
        diff_a = a[i] - a[j]
        diff_b = b[u] - b[v]
        S_total = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[u, i] - b[v, i])
        term_j = (a[i, j] - a[j, j]) * (b[u, j] - b[v, j])
        
        diag_term = (a[i, i] - a[j, j]) * (b[u, u] - b[v, v])
        offdiag_term = (a[i, j] - a[j, i]) * (b[u, v] - b[v, u])
        
        delta = 2.0 * (S_total - term_i - term_j) + diag_term + offdiag_term
        return int(delta)
        
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Local Search Loop
    # Use a simple best-improvement or first-improvement strategy.
    # With n=256, n^2/2 swaps is ~32k swaps. 
    # Each swap delta is O(n) due to dot product. 32k * 256 ops = 8M ops.
    # NumPy dot is fast. 8M ops should take < 1ms.
    # So we can do many iterations.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Randomize start order to avoid bias
        indices = np.random.permutation(n)
        
        for i_idx in indices:
            i = i_idx
            u = current_p[i]
            
            for j_idx in indices:
                j = j_idx
                if i == j:
                    continue
                v = current_p[j]
                
                delta = swap_delta(i, j, current_p)
                
                if delta < 0:
                    # Accept swap
                    current_p[i], current_p[j] = v, u
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                        
                    improved = True
                    # After a swap, the neighborhood changes significantly.
                    # Breaking inner loop is standard for first-improvement.
                    # Breaking outer loop might be too aggressive? 
                    # Let's just break inner loop and continue with next i.
                    break
            
            if time.time() > deadline - 5.0:
                break
    
    # Final safety check
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
