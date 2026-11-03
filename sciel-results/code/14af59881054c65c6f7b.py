"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Calculate initial cost
    cost = np.sum(a * b[p])
    
    best_cost = cost
    best_p = p.copy()

    # Precompute constant terms for the linearized cost matrix
    # The cost of assignment p is sum_{i,j} a[i,j] b[p[i], p[j]]
    # We want to find a new permutation p' that minimizes this.
    # We linearize this around the current p.
    
    # Let's define the "reduced cost" matrix C for the LAP.
    # If we move facility i to location j, the change in cost involves interactions with all other facilities k.
    # However, a simpler and very effective LAP-based approach is to minimize a linear approximation 
    # of the QAP objective. 
    # 
    # A common strategy is to use the "linearization" based on the current solution:
    # Objective: sum_{i,j} a_{ij} b_{p(i), p(j)}
    # We can rewrite this as sum_i (sum_j a_{ij} b_{p(i), p(j)})
    # For a fixed assignment of other facilities, the cost contribution of facility i at location l is:
    # Cost_i(l) = sum_j a_{ij} b_{l, p(j)}
    # Note: This includes the self-term a_{ii} b_{l,l} and cross terms.
    # 
    # If we solve the LAP with matrix M where M[i, l] = sum_j a_{ij} b_{l, p(j)}, 
    # we are finding the assignment that minimizes the sum of these individual costs.
    # This is equivalent to minimizing sum_i sum_j a_{ij} b_{p'(i), p'(j)} 
    # ONLY IF the assignment of j is fixed. But p' changes for all i.
    # So this is an iterative improvement: 
    # 1. Compute M[i, l] = sum_j a_{ij} b_{l, p_current(j)}
    # 2. Solve LAP for M to get new permutation p_new
    # 3. If cost(p_new) < cost(p_current), accept.
    # This is a form of block coordinate descent / successive linear approximation.
    
    # To make it more powerful, we can also add a diagonal penalty or use a different linearization.
    # But let's stick to the standard "linear QAP" step first.
    
    # Optimization: 
    # M[i, l] = sum_j a[i, j] * b[l, p[j]]
    # Let B_p be the matrix where B_p[k, l] = b[l, p[k]]? No.
    # b[l, p[j]] is the element at row l, column p[j] of b.
    # So M[i, :] = a[i, :] @ B_p.T ?
    # Let B_p be defined such that (B_p)[j, l] = b[l, p[j]].
    # Then sum_j a[i, j] * (B_p)[j, l] = (a[i, :] @ B_p)[l].
    # So M = a @ B_p.
    
    # Constructing B_p:
    # B_p[j, l] = b[l, p[j]]
    # This is equivalent to: B_p = b[:, p].T ?
    # b[:, p] has shape (n, n). Row l of b[:, p] is b[l, p].
    # So b[:, p][l, j] = b[l, p[j]].
    # We want B_p[j, l] = b[l, p[j]].
    # So B_p = (b[:, p]).T
    
    # Let's verify:
    # M[i, l] = sum_j a[i, j] * B_p[j, l]
    # B_p = b.T[:, p] ? No.
    # b is (n, n).
    # b[:, p] is (n, n). Element [l, j] is b[l, p[j]].
    # We need B_p such that B_p[j, l] = b[l, p[j]].
    # So B_p = (b[:, p]).T.
    
    # So M = a @ ((b[:, p]).T)
    # M[i, l] = sum_j a[i, j] b[l, p[j]]
    
    # Solving the LAP on M gives a permutation p_new that minimizes sum_i M[i, p_new[i]]
    # = sum_i sum_j a[i, j] b[p_new[i], p[j]]
    # This is NOT the same as the QAP cost sum_i sum_j a[i, j] b[p_new[i], p_new[j]].
    # It is a linear approximation where the "destination" of j is fixed to its current location p[j].
    # This is a valid local search move: it treats the positions of other facilities as fixed anchors 
    # and optimizes the position of each facility relative to those anchors.
    # Since the anchors change when we update p, this is an iterative process.
    
    # This method is known to be quite strong for QAPs because it allows for non-local moves 
    # (a facility can move to any location, and multiple facilities can swap in a cycle) 
    # in a single step, provided the linear approximation is good.
    
    # To improve the quality of the linearization, we can use a "weighted" linearization or 
    # simply iterate this process.
    
    # Let's implement the loop:
    # 1. Compute M
    # 2. Solve LAP
    # 3. Evaluate true cost
    # 4. If improved, update p and best.
    # 5. If not improved, try a perturbation or stop.
    
    # To avoid getting stuck immediately, we can also try to minimize a different linearization.
    # Another common one is: M[i, l] = sum_j a[i, j] b[p[i], l] ? No, that fixes source.
    # The one above (fixing destination of j) is standard.
    
    # Let's also consider the "symmetric" linearization if a and b are symmetric? 
    # tho150 is likely symmetric.
    
    # We will run this LAP-based local search.
    # To escape local optima of this specific linearization, we can add random noise to M 
    # or use a tabu-like mechanism, but for simplicity and speed, we'll just iterate.
    # If it stalls, we can perturb p by swapping a random pair and restart the LAP loop.
    
    max_lap_iters = 50
    lap_iter = 0
    
    # Precompute b.T for faster access? b[:, p] is a gather.
    # b[:, p] creates a new array. 
    # b.T is a view.
    # (b[:, p]).T is a copy.
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Compute linearization matrix M
        # M[i, l] = sum_j a[i, j] b[l, p[j]]
        # b[:, p] has shape (n, n). 
        # Let B_gathered = b[:, p]  # Row l is b[l, p]
        # M = a @ B_gathered.T
        B_gathered = b[:, p]  # (n, n) where [l, j] = b[l, p[j]]
        M = a @ B_gathered.T  # (n, n) where [i, l] = sum_j a[i, j] b[l, p[j]]
        
        # Solve LAP
        # linear_sum_assignment returns row_ind, col_ind such that sum M[row_ind, col_ind] is minimal
        row_ind, col_ind = linear_sum_assignment(M)
        
        # row_ind is 0..n-1 (facilities)
        # col_ind is the new location for each facility
        # So p_new[i] = col_ind[i]
        p_new = col_ind.astype(np.int64)
        
        # Calculate new true cost
        new_cost = np.sum(a * b[p_new])
        
        if new_cost < cost:
            # Accept
            cost = new_cost
            p = p_new
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
            
            lap_iter = 0
        else:
            # No improvement from this linearization step
            # Try a perturbation to escape
            lap_iter += 1
            if lap_iter > max_lap_iters:
                # Stuck. Perturb.
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Random swap
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
                    cost = np.sum(a * b[p])
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    lap_iter = 0
                    # Continue loop to try LAP from new position
                else:
                    # Unlikely, but if it happens, just break
                    break
            else:
                # Try a different linearization? 
                # Or just break and rely on perturbation later?
                # For now, if LAP doesn't improve, it's likely at a local optimum of the linearization.
                # We can try swapping the roles or adding noise.
                # Let's just perturb immediately if no improvement to save time.
                if time.time() - start_time > time_limit - 5:
                    break
                
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
                    cost = np.sum(a * b[p])
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    lap_iter = 0
                else:
                    break

    return best_p.tolist()

# EVOLVE-BLOCK-END
