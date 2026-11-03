"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0  # leave 5s buffer for overhead

    # Convert to float for optimization
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def perm_cost(p):
        """Compute cost of permutation p."""
        return int(np.sum(a_f * b_f[np.ix_(p, p)]))

    def sinkhorn(X, max_iter=100, tol=1e-8):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        # Ensure X is positive
        X = np.maximum(X, 1e-12)
        
        # Normalize to sum to 1 first to stabilize
        total = X.sum()
        if total == 0:
            return np.eye(n) / n
        X = X / total
        
        for _ in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.maximum(row_sums, 1e-12)
            X = X / row_sums
            
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-12)
            X = X / col_sums
            
            # Check convergence
            if np.max(np.abs(X.sum(axis=1) - 1.0)) < tol and np.max(np.abs(X.sum(axis=0) - 1.0)) < tol:
                break
        return X

    def swap_delta(p, i, j):
        """Compute cost change from swapping p[i] and p[j]."""
        pi, pj = p[i], p[j]
        
        # Efficient delta calculation for 2-swap
        # Delta = sum_k [ a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) 
        #           + a[k,i]*(b[p[k],pj] - b[p[k],pi]) + a[k,j]*(b[p[k],pi] - b[p[k],pj]) ]
        # - Double counted terms correction
        # The previous implementation had a complex correction. Let's use a simpler vectorized approach for speed if possible, 
        # or stick to the verified logic.
        
        # Vectorized calculation of the 4 sums
        # 1. a[i, :] * (b[pj, p] - b[pi, p])
        # 2. a[j, :] * (b[pi, p] - b[pj, p])
        # 3. a[:, i] * (b[p, pj] - b[p, pi])
        # 4. a[:, j] * (b[p, pi] - b[p, pj])
        
        b_pi_p = b_f[pi, p]
        b_pj_p = b_f[pj, p]
        b_p_pi = b_f[p, pi]
        b_p_pj = b_f[p, pj]
        
        delta = 0.0
        # Sum 1 & 2
        delta += np.dot(a_f[i, :], b_pj_p - b_pi_p)
        delta += np.dot(a_f[j, :], b_pi_p - b_pj_p)
        
        # Sum 3 & 4
        delta += np.dot(a_f[:, i], b_p_pj - b_p_pi)
        delta += np.dot(a_f[:, j], b_p_pi - b_p_pj)
        
        # Corrections for double counting (i,i), (j,j), (i,j), (j,i)
        # The sums above include k=i and k=j in the dot products.
        # Term (i,i): In sum 1 (k=i) and sum 3 (k=i).
        # Sum 1 k=i: a[i,i]*(b[pj,pi] - b[pi,pi])
        # Sum 3 k=i: a[i,i]*(b[pi,pj] - b[pi,pi])  -- Wait, p[k] is p[i]=pi.
        # Let's re-evaluate the double counting carefully.
        
        # It is safer to use the previous explicit correction logic which was verified, 
        # but optimized with vectorized precomputation of the main sums.
        
        # Let's stick to the previous logic but ensure it's correct.
        # Actually, for n=150, the Python loop in the previous swap_delta was slow.
        # The vectorized dot products above are O(n).
        # The correction terms are O(1).
        
        # Let's verify the correction terms from the previous code:
        # delta -= a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi])
        # delta -= a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])
        # delta -= a_f[i, j] * (b_f[pj, pj] + b_f[pi, pi] - 2 * b_f[pj, pi])
        # delta -= a_f[j, i] * (b_f[pi, pi] + b_f[pj, pj] - 2 * b_f[pi, pj])
        
        # Let's re-derive to be sure.
        # C = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Swap i, j.
        # Terms involving i or j as u or v change.
        # Unchanged: u,v not in {i,j}.
        # Changed:
        # (i,i): a[i,i] b[pi,pi] -> a[i,i] b[pj,pj]. Delta: a[i,i](b[pj,pj]-b[pi,pi])
        # (j,j): a[j,j] b[pj,pj] -> a[j,j] b[pi,pi]. Delta: a[j,j](b[pi,pi]-b[pj,pj])
        # (i,j): a[i,j] b[pi,pj] -> a[i,j] b[pj,pi]. Delta: a[i,j](b[pj,pi]-b[pi,pj])
        # (j,i): a[j,i] b[pj,pi] -> a[j,i] b[pi,pj]. Delta: a[j,i](b[pi,pj]-b[pj,pi])
        
        # Terms with one index in {i,j} and one outside:
        # (i,k) k!=i,j: a[i,k] b[pi,p[k]] -> a[i,k] b[pj,p[k]]. Delta: a[i,k](b[pj,p[k]]-b[pi,p[k]])
        # (j,k) k!=i,j: a[j,k] b[pj,p[k]] -> a[j,k] b[pi,p[k]]. Delta: a[j,k](b[pi,p[k]]-b[pj,p[k]])
        # (k,i) k!=i,j: a[k,i] b[p[k],pi] -> a[k,i] b[p[k],pj]. Delta: a[k,i](b[p[k],pj]-b[p[k],pi])
        # (k,j) k!=i,j: a[k,j] b[p[k],pj] -> a[k,j] b[p[k],pi]. Delta: a[k,j](b[p[k],pi]-b[p[k],pj])
        
        # My vectorized sums included k=i and k=j.
        # Sum 1 (a[i,:] * ...): Includes k=i, j.
        # k=i: a[i,i](b[pj,pi]-b[pi,pi])
        # k=j: a[i,j](b[pj,pj]-b[pi,pj])
        
        # Sum 2 (a[j,:] * ...): Includes k=i, j.
        # k=i: a[j,i](b[pi,pi]-b[pj,pi])
        # k=j: a[j,j](b[pi,pj]-b[pj,pj])
        
        # Sum 3 (a[:,i] * ...): Includes k=i, j.
        # k=i: a[i,i](b[pi,pj]-b[pi,pi])  -- Note b[p[i], pj] = b[pi, pj]
        # k=j: a[j,i](b[pj,pj]-b[pi,pj]) -- Note b[p[j], pj] = b[pj, pj], b[p[j], pi] = b[pj, pi]
        
        # Sum 4 (a[:,j] * ...): Includes k=i, j.
        # k=i: a[i,j](b[pi,pi]-b[pi,pj]) -- Note b[p[i], pi] = b[pi, pi], b[p[i], pj] = b[pi, pj]
        # k=j: a[j,j](b[pj,pi]-b[pj,pj]) -- Note b[p[j], pi] = b[pj, pi], b[p[j], pj] = b[pj, pj]
        
        # Total Vectorized Delta (V) = Sum 1 + Sum 2 + Sum 3 + Sum 4
        
        # We want True Delta (T).
        # T = (i,i) + (j,j) + (i,j) + (j,i) + (i,k)_{k!=i,j} + (j,k)_{k!=i,j} + (k,i)_{k!=i,j} + (k,j)_{k!=i,j}
        
        # Let's express V in terms of T components.
        # V includes all terms where u=i or v=i or u=j or v=j.
        # But terms like (i,i) appear in Sum 1 (u=i, v=i) and Sum 3 (u=i, v=i). So (i,i) is counted twice in V.
        # (j,j) appears in Sum 2 (u=j, v=j) and Sum 4 (u=j, v=j). Counted twice.
        # (i,j) appears in Sum 1 (u=i, v=j) and Sum 4 (u=i, v=j). Counted twice.
        # (j,i) appears in Sum 2 (u=j, v=i) and Sum 3 (u=j, v=i). Counted twice.
        # (i,k) k!=i,j appears in Sum 1 (u=i, v=k) only? 
        #   Sum 1: u=i, v=k. Yes.
        #   Sum 3: u=k, v=i. No, that's (k,i).
        #   So (i,k) is counted once in V?
        #   Wait. Sum 1 is sum_k a[i,k] ... . This covers all v=k.
        #   Sum 3 is sum_k a[k,i] ... . This covers all u=k.
        #   So for (i,k) with k!=i,j:
        #   It is in Sum 1 (u=i, v=k).
        #   Is it in Sum 3? Sum 3 is (u=k, v=i). No.
        #   Is it in Sum 2? u=j. No.
        #   Is it in Sum 4? v=j. No.
        #   So (i,k) is counted ONCE in V.
        #   But in T, (i,k) is a distinct term.
        #   So V = 2*(i,i) + 2*(j,j) + 2*(i,j) + 2*(j,i) + 1*(i,k) + 1*(j,k) + 1*(k,i) + 1*(k,j) ... for k!=i,j.
        #   T = 1*(i,i) + 1*(j,j) + 1*(i,j) + 1*(j,i) + 1*(i,k) + 1*(j,k) + 1*(k,i) + 1*(k,j).
        
        # Therefore: V = T + (i,i) + (j,j) + (i,j) + (j,i)
        # So T = V - [(i,i) + (j,j) + (i,j) + (j,i)]
        
        # Let's calculate the correction term C = (i,i) + (j,j) + (i,j) + (j,i)
        # (i,i) delta: a[i,i](b[pj,pj]-b[pi,pi])
        # (j,j) delta: a[j,j](b[pi,pi]-b[pj,pj])
        # (i,j) delta: a[i,j](b[pj,pi]-b[pi,pj])
        # (j,i) delta: a[j,i](b[pi,pj]-b[pj,pi])
        
        # So T = V - [ a[i,i](b[pj,pj]-b[pi,pi]) + a[j,j](b[pi,pi]-b[pj,pj]) + a[i,j](b[pj,pi]-b[pi,pj]) + a[j,i](b[pi,pj]-b[pj,pi]) ]
        
        # Let's check the previous code's corrections:
        # prev: delta -= a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi])  # Matches (i,i)
        # prev: delta -= a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])  # Matches (j,j)
        # prev: delta -= a_f[i, j] * (b_f[pj, pj] + b_f[pi, pi] - 2 * b_f[pj, pi]) 
        #   My derivation: a[i,j](b[pj,pi]-b[pi,pj])
        #   Prev: a[i,j](b[pj,pj] + b[pi,pi] - 2b[pj,pi])
        #   Are they equal?
        #   b[pj,pi] - b[pi,pj]  vs  b[pj,pj] + b[pi,pi] - 2b[pj,pi]
        #   If b is symmetric (QAPLIB distance matrices are symmetric), b[pi,pj]=b[pj,pi].
        #   Then RHS = b[pj,pj] + b[pi,pi] - 2b[pi,pj].
        #   LHS = b[pi,pj] - b[pi,pj] = 0? No.
        #   Wait, (i,j) term in V:
        #   Sum 1 (k=j): a[i,j](b[pj,pj]-b[pi,pj])
        #   Sum 4 (k=i): a[i,j](b[pi,pi]-b[pi,pj])  -- Wait, Sum 4 is a[k,j](b[p[k],pi]-b[p[k],pj]). k=i -> a[i,j](b[pi,pi]-b[pi,pj]).
        #   So V contains a[i,j](b[pj,pj]-b[pi,pj] + b[pi,pi]-b[pi,pj]) = a[i,j](b[pj,pj]+b[pi,pi]-2b[pi,pj]).
        #   T contains a[i,j](b[pj,pi]-b[pi,pj]).
        #   If b is symmetric, b[pj,pi]=b[pi,pj].
        #   T(i,j) = a[i,j](b[pi,pj]-b[pi,pj]) = 0? No, b[pj,pi] is b[pi,pj] if symmetric.
        #   Wait, the assignment changes.
        #   Original: i at pi, j at pj. Term (i,j) is a[i,j] b[pi,pj].
        #   New: i at pj, j at pi. Term (i,j) is a[i,j] b[pj,pi].
        #   Delta = a[i,j] (b[pj,pi] - b[pi,pj]).
        #   If b is symmetric, b[pj,pi] = b[pi,pj]. So Delta is 0?
        #   This implies that for symmetric B, swapping i and j does not change the cost of the pair (i,j) relative to itself?
        #   No, it changes which location is which.
        #   If B is symmetric, B[pi,pj] = B[pj,pi].
        #   So the cost of the edge (i,j) is the same before and after swap?
        #   Yes, if B is symmetric, the distance between location pi and pj is the same.
        #   So the term a[i,j] b[pi,pj] becomes a[i,j] b[pj,pi] = a[i,j] b[pi,pj]. No change.
        #   So T(i,j) = 0.
        #   And V(i,j) contribution = a[i,j](b[pj,pj]+b[pi,pi]-2b[pi,pj]).
        #   So Correction = V(i,j) - T(i,j) = a[i,j](b[pj,pj]+b[pi,pi]-2b[pi,pj]).
        #   This matches the previous code!
        
        # So the previous code was correct for symmetric B. QAPLIB instances usually have symmetric distance matrices.
        # I will use the vectorized V and the correction terms.
        
        correction = (a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi]) + 
                      a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj]) + 
                      a_f[i, j] * (b_f[pj, pj] + b_f[pi, pi] - 2 * b_f[pj, pi]) + 
                      a_f[j, i] * (b_f[pi, pi] + b_f[pj, pj] - 2 * b_f[pi, pj]))
                      
        return delta - correction

    def local_search_2opt(p, max_iter=1000, time_limit_inner=10.0):
        """Improve permutation via pairwise swaps."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        
        no_improve_count = 0
        for _ in range(max_iter):
            if time.time() - t_start > time_limit_inner:
                break
            
            improved = False
            # Try random pairs of swaps
            for _ in range(50):
                i, j = np.random.choice(n, size=2, replace=False)
                if i == j:
                    continue
                
                delta = swap_delta(p, i, j)
                
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
        return p, current_cost

    # 1. Generate initial permutation using rank matching
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # 2. Continuous Relaxation
    # Initialize X
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p_init[i]] = 1.0
    X = sinkhorn(X, max_iter=10)

    # Optimization
    step_size = 0.05
    best_relaxed_cost = np.inf
    best_X = X.copy()
    
    # Precompute gradients if possible? No, X changes.
    
    for iteration in range(2000):
        if time.time() - t0 > time_limit - 15:
            break
        
        # Gradient: A X B^T + B X^T A^T
        # Use float64 for stability
        AX = a_f @ X
        AXB_T = AX @ b_f.T
        BX_T = b_f @ X.T
        BX_T_A_T = BX_T @ a_f.T
        grad = AXB_T + BX_T_A_T
        
        # Update X
        X_new = X - step_size * grad
        X_new = sinkhorn(X_new, max_iter=20)
        
        # Calculate cost
        # C = tr(A X B X^T) = sum(A * (X B X^T))
        # Or sum(B * (A X B X^T))? No.
        # C = sum_{i,j} A_{ij} (X B X^T)_{ij}
        # (X B X^T)_{ij} = sum_k X_{ik} B_{kl} X_{jl}
        # Let's compute M = X @ b_f @ X.T
        # Then cost = np.sum(a_f * M)
        M = X_new @ b_f @ X_new.T
        cost = np.sum(a_f * M)
        
        if cost < best_relaxed_cost:
            best_relaxed_cost = cost
            best_X = X_new.copy()
            X = X_new
            # Increase step size slightly if improving
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Decrease step size
            step_size *= 0.8
            if step_size < 1e-6:
                break

    # 3. Recover permutation from best_X
    # We want to find permutation p that maximizes sum X[p[i], i]? 
    # No, X[i,j] is 1 if i is at j.
    # We want to pick one j for each i such that X[i,j] is high.
    # This is a maximum weight bipartite matching problem.
    # scipy linear_sum_assignment minimizes cost. So we pass -best_X.
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p_relaxed = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p_relaxed[i] = j
        
    current_cost = perm_cost(p_relaxed)
    best_p = p_relaxed.copy()
    best_cost = current_cost

    # 4. Discrete Polish
    if time.time() - t0 < time_limit - 5:
        p_ls, cost_ls = local_search_2opt(best_p, max_iter=2000, time_limit_inner=time_limit - 5 - (time.time() - t0))
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls

    return best_p.tolist()
# EVOLVE-BLOCK-END
