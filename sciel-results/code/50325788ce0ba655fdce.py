"""Rewritten QAP solver: spectral + Hungarian + fast 2-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization via eigendecomposition of flow/distance matrices.
    2. Hungarian refinement on linearized cost.
    3. Fast 2-opt local search with precomputed delta terms.
    4. Iterated local search with random perturbations.
    """
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # --- Spectral initialization ---
    # For QAP, a good heuristic is to match facilities by flow "centrality" 
    # with locations by distance "centrality". But a more sophisticated approach
    # uses the eigenvectors of the matrices.
    
    # Simple rank-matching as baseline
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum, locations by increasing distance sum
    facility_order = np.argsort(-flow_sums, kind="stable")
    location_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order

    # --- Compute cost function ---
    def compute_cost(p):
        """Compute total cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
        # Vectorized: for each i, a[i,:] dot b[p[i], p[:]]
        # b[p[i], p] is the row p[i] of b indexed by p
        # We can do: B_p = b[np.ix_(p, p)] then sum(a * B_p) but that's O(n^2) memory
        # Alternative: sum_i a[i,:] @ b[p[i], p]
        # b[p[i], p] for all i: this is b[p, p] which is n x n
        # Actually: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] * b[p[i], p]).sum()
        # Let's compute B_p = b[p][:, p] -- wait, b[p] gives rows in order p
        # b[p, :] has shape (n,n) where row i is b[p[i], :]
        # Then b[p, p] has element (i,j) = b[p[i], p[j]]
        # So cost = np.sum(a * b[p, p])
        B_p = b[p][:, p]
        return int(np.sum(a * B_p))

    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # --- Hungarian refinement ---
    def hungarian_refine(p):
        """Refine permutation using Hungarian on linearized cost matrix."""
        # Linearized cost: C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        # b[j, p[k]]: for fixed j, this is b[j, :] indexed by p -> b[j, p]
        # So sum_k a[i,k]*b[j,p[k]] = (a[i,:] * b[j,p]).sum() = a[i,:] @ b[j,p]
        # Similarly sum_k a[k,i]*b[p[k],j] = a[:,i] @ b[p,j]
        
        # Build C[i,j] = a[i,:] @ b[j,p] + a[:,i] @ b[p,j]
        # = a[i,:] @ b[j,p] + a[:,i] @ b[p,j]
        
        # b[j,p] for all j: b[:, p] -- shape (n,n), row j is b[j, p]
        # b[p,j] for all j: b[p, :] -- shape (n,n), row k is b[p[k], :], col j is b[p[k],j]
        
        b_col_p = b[:, p]  # (n,n): [j,k] = b[j,p[k]]
        b_row_p = b[p, :]  # (n,n): [k,j] = b[p[k],j]
        
        # C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        # First term: (a @ b_col_p.T)[i,j] = sum_k a[i,k]*b_col_p[j,k] = sum_k a[i,k]*b[j,p[k]] ✓
        # Second term: (a.T @ b_row_p)[i,j] = sum_k a.T[i,k]*b_row_p[k,j] = sum_k a[k,i]*b[p[k],j] ✓
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    # Apply Hungarian refinement
    if time.time() < deadline - 15.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung

    # --- Fast local search with precomputed deltas ---
    # Precompute for fast swap delta evaluation
    # For a swap of positions i and j (values u=p[i], v=p[j]):
    # Delta = sum over affected terms
    # We can precompute:
    #   R[i] = sum_k a[i,k] * b[p[i], p[k]]  (row contribution of i)
    #   C[i] = sum_k a[k,i] * b[p[k], p[i]]  (col contribution of i)
    # But the delta for swapping i and j involves cross terms.
    
    # Let's use a direct but vectorized approach for the delta.
    
    def swap_delta_fast(i, j, p):
        """Compute delta for swapping p[i] and p[j]. Returns delta (negative = improvement)."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        
        # The change in cost from swapping positions i and j:
        # Old terms involving u or v in b:
        #   For all (k,l): a[k,l] * b[p[k], p[l]]
        # After swap: p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j
        # Changed terms: those where p[k] in {u,v} or p[l] in {u,v}
        # i.e., k in {i,j} or l in {i,j}
        
        # Delta = sum_{k,l: k in {i,j} or l in {i,j}} a[k,l]*(b[p'[k],p'[l]] - b[p[k],p[l]])
        
        # Group into 4 categories:
        # (i,i), (i,j), (j,i), (j,j) -- direct
        # (i,k), (j,k) for k not in {i,j} -- row i/j changes
        # (k,i), (k,j) for k not in {i,j} -- col i/j changes
        
        # Direct terms:
        delta = 0
        delta += a[i,i] * (b[v,v] - b[u,u])
        delta += a[i,j] * (b[v,u] - b[u,v])
        delta += a[j,i] * (b[u,v] - b[v,u])
        delta += a[j,j] * (b[u,u] - b[v,v])
        
        # Row terms: k not in {i,j}
        # (i,k): a[i,k]*(b[v,p[k]] - b[u,p[k]])
        # (j,k): a[j,k]*(b[u,p[k]] - b[v,p[k]])
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            # Vectorized dot products
            delta += np.dot(a[i, ks], b[v, p[ks]] - b[u, p[ks]])
            delta += np.dot(a[j, ks], b[u, p[ks]] - b[v, p[ks]])
            # Col terms: (k,i): a[k,i]*(b[p[k],v] - b[p[k],u])
            # (k,j): a[k,j]*(b[p[k],u] - b[p[k],v])
            delta += np.dot(a[ks, i], b[p[ks], v] - b[p[ks], u])
            delta += np.dot(a[ks, j], b[p[ks], u] - b[p[ks], v])
        
        return delta

    # Local search loop
    def local_search(p, cost, max_time):
        """Perform 2-opt local search until no improvement or time runs out."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Random sampling of pairs for large n
            if n > 50:
                num_trials = 50000
                for _ in range(num_trials):
                    if time.time() > max_time:
                        break
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i == j:
                        continue
                    d = swap_delta_fast(i, j, p)
                    if d < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += d
                        improved = True
            else:
                # Exhaustive for small n
                for i in range(n):
                    if time.time() > max_time:
                        break
                    for j in range(i + 1, n):
                        d = swap_delta_fast(i, j, p)
                        if d < -1e-9:
                            p[i], p[j] = p[j], p[i]
                            cost += d
                            improved = True
        return p, cost

    # Phase 1: Local search from current best
    if time.time() < deadline - 20.0:
        best_p, best_cost = local_search(best_p.copy(), best_cost, deadline - 20.0)

    # Phase 2: Hungarian + local search iteration
    remaining_budget = deadline - 5.0
    num_iterations = 0
    while time.time() < remaining_budget - 10.0 and num_iterations < 5:
        num_iterations += 1
        # Hungarian refinement
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
        # Local search
        if time.time() < remaining_budget - 5.0:
            best_p, best_cost = local_search(best_p.copy(), best_cost, remaining_budget - 5.0)

    # Phase 3: Final polish with more local search
    if time.time() < deadline - 3.0:
        best_p, best_cost = local_search(best_p.copy(), best_cost, deadline - 3.0)

    return best_p.tolist()
# EVOLVE-BLOCK-END
