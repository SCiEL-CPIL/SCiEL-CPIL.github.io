"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s
    deadline = start_time + 145.0
    
    # Ensure int64
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # 1. Initialization: Rank Matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    def get_cost(p):
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # b_perm[i, j] = b[p[i], p[j]]
        # This creates a new array of size n x n.
        # b[np.ix_(p, p)] is efficient.
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
        
    current_cost = get_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute for 2-swap delta: 
    # We maintain c[i] = sum_j a[i,j] * b[p[i], p[j]]
    # This is the contribution of facility i to the total cost.
    # Total cost = sum_i c[i] (Note: this counts each pair twice if a is symmetric? 
    # No, c[i] is sum over j. The total cost is sum_i sum_j a[i,j]b[p[i],p[j]].
    # So total cost = sum_i c[i].
    
    def compute_c(p, a, b):
        c = np.zeros(n, dtype=np.int64)
        # c[i] = dot(a[i], b[p[i], p])
        # b[p[i], p] is the row of b at index p[i], permuted by p?
        # No. b[p[i], p[j]] for all j.
        # This is simply b[p[i], p]
        for i in range(n):
            c[i] = int(np.dot(a[i], b[p[i], p]))
        return c
        
    c = compute_c(p, a, b)
    
    # 2. Local Search: 2-Swap
    # Delta for swapping i and j:
    # Let u = p[i], v = p[j]
    # New cost - Old cost = 
    # (a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u]))  # Interaction between i,j
    # + sum_{k != i,j} a[i,k] * (b[v, p[k]] - b[u, p[k]])    # i with others
    # + sum_{k != i,j} a[j,k] * (b[u, p[k]] - b[v, p[k]])    # j with others
    
    # We can vectorize the sums over k.
    # Let Bp = b[:, p] (n x n). Bp[x, k] = b[x, p[k]]
    # Term 2: sum_{k != i,j} a[i,k] * (Bp[v, k] - Bp[u, k])
    # Term 3: sum_{k != i,j} a[j,k] * (Bp[u, k] - Bp[v, k])
    
    # We will iterate i, and vectorize j.
    
    while time.time() < deadline:
        improved = False
        # One pass of 2-swap local search
        # To speed up, we can randomize the order of i or use a limited number of passes
        # Let's do a full pass but break if no improvement in a pass? 
        # Or just keep going until no improvement.
        
        # Precompute Bp = b[:, p] for this configuration
        Bp = b[:, p]
        
        # We will check all pairs (i, j) with i < j
        # For each i, compute deltas for all j > i
        # delta_ij = (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]]) 
        #            + np.dot(a[i] - a[j], Bp[p[j]] - Bp[p[i]])
        # Note: The dot product includes k=i and k=j. 
        # Let's check the terms for k=i and k=j in the dot product.
        # Dot = sum_k (a[i,k] - a[j,k]) * (Bp[v,k] - Bp[u,k])
        # For k=i: (a[i,i] - a[j,i]) * (Bp[v,i] - Bp[u,i]) = (a[i,i] - a[j,i]) * (b[v, p[i]] - b[u, p[i]])
        #          = (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
        # For k=j: (a[i,j] - a[j,j]) * (Bp[v,j] - Bp[u,j]) = (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
        
        # The manual formula for 2-swap delta usually excludes i,j from the sum and handles them separately.
        # Let's stick to the safe, slightly slower but correct loop for i, vectorized for j.
        
        any_improvement_in_pass = False
        # Randomize order of i to avoid bias
        order = np.random.permutation(n)
        
        for i in order:
            if time.time() > deadline:
                break
            u = p[i]
            ai = a[i]
            # We want to find best j > i (or any j != i)
            # To ensure we don't miss, we check all j != i.
            # But checking all j for each i is O(n^2).
            # Let's vectorize for all j at once.
            
            # Deltas for swapping i with all j
            # j_indices = np.arange(n)
            # mask = j_indices != i
            # v_vals = p[j_indices[mask]]
            
            # Let's just loop j for simplicity and safety, n=150 is small.
            # 150 * 149 / 2 ~ 11k pairs. 11k dot products of length 150.
            # This is fast in NumPy if vectorized, but loop in Python is slow.
            
            # Vectorized approach for fixed i:
            # j_idx = np.arange(n)
            # j_idx = j_idx[j_idx != i]
            # v = p[j_idx]
            # diff_B = Bp[v] - Bp[u]  # shape (num_j, n)
            # diff_A = ai - a[j_idx]  # shape (num_j, n)
            # deltas = np.sum(diff_A * diff_B, axis=1)
            # + (a[i, j_idx] - a[j_idx, i]) * (b[v, u] - b[u, v])
            
            j_idx = np.arange(n)
            j_idx = j_idx[j_idx != i]
            if len(j_idx) == 0: continue
            
            v = p[j_idx]
            # Bp[v] gives rows of Bp for indices v. Bp[x, k] = b[x, p[k]].
            # We need sum_k a[i,k] * (b[v, p[k]] - b[u, p[k]]) - sum_k a[j,k] * ...
            # This is exactly dot(a[i] - a[j], Bp[v] - Bp[u])
            
            # Bp[v] is (num_j, n)
            # Bp[u] is (n,) -> broadcast to (num_j, n)
            diff_B = Bp[v] - Bp[u]
            
            # a[j_idx] is (num_j, n)
            diff_A = ai[None, :] - a[j_idx]
            
            dot_terms = np.sum(diff_A * diff_B, axis=1)
            
            # Interaction term: (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
            # b[v,u] is b[p[j], p[i]]
            # b[u,v] is b[p[i], p[j]]
            # We have v = p[j_idx], u = p[i]
            # b_v_u = b[v, u] -> b[np.ix_(v, [u])] ? No, b[v, u] is just b[v, u] element?
            # No, u and v are indices. b[v, u] is a scalar for each j.
            # b_mat = b[v, u] -> This is not valid numpy for array v and scalar u? 
            # b[v, u] works if u is scalar? Yes, b[v, u] gives array of b[v_i, u].
            # b[u, v] gives array of b[u, v_i].
            
            b_v_u = b[v, u]
            b_u_v = b[u, v]
            
            inter_terms = (a[i, j_idx] - a[j_idx, i]) * (b_v_u - b_u_v)
            
            deltas = dot_terms + inter_terms
            
            # Find best delta
            min_idx_local = np.argmin(deltas)
            min_delta = deltas[min_idx_local]
            
            if min_delta < 0:
                j = j_idx[min_idx_local]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                current_cost += int(min_delta)
                
                # Update c? 
                # We don't strictly need c if we recompute Bp? 
                # But we reused Bp. If we swap, Bp changes.
                # So we must break and recompute Bp? Or update Bp?
                # Updating Bp is tricky. 
                # Given n=150, recomputing Bp (150x150 slice) is fast.
                # So we break the i-loop and restart the pass?
                # Or just update p and continue? 
                # If we continue, Bp is stale.
                # So we must restart the pass or recompute Bp.
                # Let's recompute Bp and c immediately.
                Bp = b[:, p]
                c = compute_c(p, a, b) # Actually we don't use c in the vectorized delta above, we use a and Bp directly.
                # Wait, my vectorized delta above uses `a` and `Bp`. It does NOT use `c`.
                # So I don't need to update c for the delta calculation.
                # I only need to update Bp.
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                
                any_improvement_in_pass = True
                # Break i-loop to recompute Bp for next i? 
                # Or just continue with stale Bp? No, stale Bp gives wrong deltas.
                # So we must break and restart the pass? 
                # That is expensive.
                # Alternative: Don't vectorize over j for fixed i. 
                # Just loop j as well.
                # 11k pairs * (dot product len 150) = 1.6M ops. 
                # Python loop overhead is high.
                
                # Let's stick to the loop over i, but if we swap, we break and restart the outer while loop?
                # No, that's inefficient.
                
                # Better: Use the `c` array approach for O(n) delta per pair?
                # Delta = (a[i,j]-a[j,i])*(b[v,u]-b[u,v]) 
                #         + (c[i] - c[j] - a[i,j]*b[u,v] - a[j,i]*b[v,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v] ... ) 
                # The c-based delta is:
                # delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v]) 
                #         + sum_{k!=i,j} a[i,k](b[v,p[k]] - b[u,p[k]]) 
                #         + sum_{k!=i,j} a[j,k](b[u,p[k]] - b[v,p[k]])
                
                # Let's revert to the simple loop for i and j, but optimize the inner dot.
                # Actually, the vectorized approach above is good IF we don't swap.
                # If we swap, we update p. The next i in the loop will use the NEW p?
                # No, Bp is fixed at the start of the i-loop.
                # So if we swap, Bp is invalid for subsequent j in the same i, and for subsequent i.
                
                # Solution: 
                # 1. Find best move in the current Bp.
                # 2. If improvement, apply it.
                # 3. Recompute Bp.
                # 4. Restart the i-loop? Or continue with the new Bp?
                # If we continue, we might miss moves involving the new i?
                # Let's just restart the pass (break i-loop) and start a new pass.
                # Since n=150, a pass is ~11k checks. 
                # If we improve, we restart. This is a steepest descent with restarts.
                
                break # Break i-loop, restart pass
        
        if not any_improvement_in_pass:
            # Local minimum reached for 2-swaps
            # Structural Change: 3-Swap Perturbation
            if time.time() < deadline - 5.0: # Need time for next LS
                # Pick 3 random distinct indices
                idxs = np.random.choice(n, 3, replace=False)
                i, j, k = idxs
                # Rotate p[i], p[j], p[k]
                # i gets j's loc, j gets k's loc, k gets i's loc
                tmp = p[i]
                p[i] = p[j]
                p[j] = p[k]
                p[k] = tmp
                
                current_cost = get_cost(p)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                
                # Update Bp and c for next LS
                Bp = b[:, p]
                c = compute_c(p, a, b)
                
            else:
                break # No time for perturbation
        
        # End while
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
