# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Ensure arrays are int64 for exact integer arithmetic where possible
    # But for vectorized ops, float64 is often faster. We'll cast back or use int64 carefully.
    # The task says a and b are int64.
    
    # 1. Initial Solution: Rank Matching (Strong Heuristic)
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Helper to compute cost
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()

    # Precompute structures for fast delta calculation
    # We maintain:
    # b_p: matrix where b_p[i] = b[p[i]] (rows of b indexed by p)
    # This allows fast access to b[u, p] which is b_p[u]
    # And b[p, u] which is b[:, u][p] -> can be precomputed as b_col_p? 
    # Actually, let's stick to the vectorized delta formula from inspiration 1/2 which is very fast.
    
    # delta = sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]]) 
    #       + sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    
    # To make this O(n) per swap check, we can precompute:
    # For a fixed i, we need to check all j.
    # Let's use the "incremental" approach or just efficient vectorized loops.
    # Given n=150, O(n^3) total search time is roughly 150^3 = 3.3M ops per full pass?
    # No, a full pass of 2-swap is O(n^2) pairs. Each pair check is O(n) dot product.
    # So one full pass is O(n^3) = 3.3M ops. 
    # In Python, this might be slow if not vectorized.
    # Using NumPy, we can vectorize the inner loop for a fixed i.
    
    # Let's implement a vectorized local search step.
    # For a fixed i, we can compute the delta for all j > i at once?
    # delta(i, j) depends on p[i], p[j].
    
    # Let's try a simpler, robust approach:
    # 1. Generate multiple initial solutions using GRASP-like construction.
    # 2. For each, run a fast vectorized 2-swap local search.
    # 3. Keep the best.

    def grasp_construct(alpha=0.5):
        """
        Construct a solution using GRASP.
        At each step, pick a facility and assign it to a location from a restricted candidate list.
        """
        p = np.empty(n, dtype=np.int64)
        assigned_facilities = np.zeros(n, dtype=bool)
        assigned_locations = np.zeros(n, dtype=bool)
        
        # Precompute static parts? Hard because cost depends on current assignment.
        # Simplified GRASP: 
        # 1. Sort facilities by some metric (e.g., row sum of A).
        # 2. For each facility, compute cost of assigning to each free location.
        # 3. Pick from top alpha fraction randomly.
        
        # Metric for ordering facilities: row sum of A
        facility_order = np.argsort(-a.sum(axis=1), kind="stable")
        
        for idx in facility_order:
            # Compute incremental cost for assigning facility idx to each free location l
            # Cost contribution of i->l: sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
            # We need to know p[j] for already assigned j.
            
            # Initialize cost vector for all locations
            costs = np.zeros(n, dtype=np.float64)
            
            # Only consider unassigned locations
            free_locs = np.where(~assigned_locations)[0]
            
            if len(free_locs) == 0:
                continue
                
            # Vectorized calculation
            # For each free location l in free_locs:
            # cost[l] = a[idx] @ b[l, p_assigned] + a[:, idx] @ b[p_assigned, l]
            
            # p_assigned is p[assigned_facilities]
            # b[l, p_assigned] is a vector of length num_assigned
            # a[idx] is a vector of length n. We need a[idx, assigned_facilities]
            
            a_row_idx = a[idx]
            a_col_idx = a[:, idx]
            
            # Get assigned facilities
            assigned_idx = np.where(assigned_facilities)[0]
            if len(assigned_idx) == 0:
                # First assignment, cost is 0 (or just random)
                # Pick random from free
                l = free_locs[random.randint(0, len(free_locs)-1)]
                p[idx] = l
                assigned_facilities[idx] = True
                assigned_locations[l] = True
                continue
            
            p_assigned = p[assigned_idx]
            
            # b[l, p_assigned] for all l in free_locs
            # b[free_locs, p_assigned] is a matrix of size (len(free_locs), len(assigned_idx))
            b_rows = b[free_locs, p_assigned] # shape: (F, A)
            # a_row_idx[assigned_idx] is shape (A,)
            # Dot product: b_rows @ a_row_idx[assigned_idx] -> shape (F,)
            term1 = b_rows @ a_row_idx[assigned_idx]
            
            # b[p_assigned, l] for all l in free_locs
            # b[p_assigned, free_locs] is a matrix of size (A, F)
            b_cols = b[p_assigned, free_locs] # shape: (A, F)
            # a_col_idx[assigned_idx] is shape (A,)
            # Dot product: b_cols.T @ a_col_idx[assigned_idx] -> shape (F,)
            # Wait, b[p_assigned, free_locs] is (A, F). We want sum_k a[k, idx] * b[p[k], l]
            # This is a_col_idx[assigned_idx] @ b[p_assigned, free_locs]
            term2 = a_col_idx[assigned_idx] @ b_cols
            
            costs[free_locs] = term1 + term2
            
            # Restricted Candidate List: pick from top alpha fraction
            # Sort costs for free locations
            sorted_costs = np.sort(costs[free_locs])
            num_candidates = max(1, int(len(free_locs) * alpha))
            
            # Get the indices of the best candidates
            # We need the locations, not just the costs
            # Find the threshold
            threshold = sorted_costs[num_candidates - 1]
            
            # Candidates are free_locs where costs[free_locs] <= threshold
            # Note: costs is 0 for non-free, so this is safe
            candidate_mask = costs[free_locs] <= threshold
            candidates = free_locs[candidate_mask]
            
            # Random pick from candidates
            # To ensure diversity, we can use a weighted random or just uniform.
            # Uniform is standard GRASP.
            l = candidates[random.randint(0, len(candidates)-1)]
            
            p[idx] = l
            assigned_facilities[idx] = True
            assigned_locations[l] = True
            
        return p

    def local_search(p, current_cost):
        """
        Vectorized 2-swap local search.
        """
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # To avoid bias, we can shuffle the order of i, but O(n^2) is fixed.
            # We'll just iterate i from 0 to n-1, j from i+1 to n-1.
            # For efficiency, we can vectorize the j-loop for a fixed i.
            
            for i in range(n):
                if time.time() > deadline - 2.0:
                    return p, current_cost, False
                
                u = p[i]
                # Precompute vectors for i
                # We want delta(i, j) for all j > i
                # delta = (a[i] - a[j]) @ (b[p[j], p] - b[u, p]) + (a[:, i] - a[:, j]) @ (b[p, p[j]] - b[p, u])
                
                # Let's compute this vectorized over j.
                # Let J = range(i+1, n)
                # p_j = p[J]
                # b_u_p = b[u, p] (shape n)
                # b_p_u = b[p, u] (shape n)
                
                b_u_p = b[u, p]
                b_p_u = b[p, u]
                
                # For each j in J:
                # term1_j = (a[i] - a[j]) @ (b[p[j], p] - b_u_p)
                # term2_j = (a[:, i] - a[:, j]) @ (b[p, p[j]] - b_p_u)
                
                # b[p[j], p] for all j in J is b[p[J], p] (shape |J|, n)
                # b[p, p[j]] for all j in J is b[p, p[J]] (shape n, |J|)
                
                p_J = p[i+1:n]
                if len(p_J) == 0:
                    continue
                
                b_pJ_p = b[p_J, p] # shape (|J|, n)
                b_p_pJ = b[p, p_J] # shape (n, |J|)
                
                # diff_b_row_j = b[p[j], p] - b_u_p
                # diff_b_row = b_pJ_p - b_u_p[None, :] # shape (|J|, n)
                
                # diff_b_col_j = b[p, p[j]] - b_p_u
                # diff_b_col = b_p_pJ - b_p_u[:, None] # shape (n, |J|)
                
                # diff_a_row_j = a[i] - a[j]
                # diff_a_row = a[i][None, :] - a[i+1:n] # shape (|J|, n)
                
                # diff_a_col_j = a[:, i] - a[:, j]
                # diff_a_col = a[:, i][:, None] - a[:, i+1:n] # shape (n, |J|)
                
                # term1_j = diff_a_row_j @ diff_b_row_j
                # This is sum over k.
                # diff_a_row has shape (|J|, n), diff_b_row has shape (|J|, n)
                # term1 = np.sum(diff_a_row * diff_b_row, axis=1)
                
                # term2_j = diff_a_col_j @ diff_b_col_j
                # diff_a_col has shape (n, |J|), diff_b_col has shape (n, |J|)
                # term2 = np.sum(diff_a_col * diff_b_col, axis=0)
                
                diff_a_row = a[i][None, :] - a[i+1:n]
                diff_b_row = b_pJ_p - b_u_p[None, :]
                term1 = np.sum(diff_a_row * diff_b_row, axis=1)
                
                diff_a_col = a[:, i][:, None] - a[:, i+1:n]
                diff_b_col = b_p_pJ - b_p_u[:, None]
                term2 = np.sum(diff_a_col * diff_b_col, axis=0)
                
                deltas = term1 + term2
                
                # Find the best improving swap
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < 0:
                    j = i + 1 + min_idx
                    v = p[j]
                    
                    # Perform swap
                    p[i], p[j] = v, u
                    current_cost += int(min_delta)
                    improved = True
                    
                    # Update best if better
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
                    # Restart the i-loop? 
                    # Standard first-improvement restarts to ensure we don't miss better moves.
                    # But vectorized search over all j for a fixed i is efficient.
                    # We can just continue to next i, or restart from i=0.
                    # Restarting from i=0 is safer for convergence but slower.
                    # Let's restart from i=0 for robustness.
                    break # Break inner loop, restart outer
                    
            # If no improvement found in full pass, break
            # The 'improved' flag handles this.
            
        return p, current_cost, True

    # Main Loop: Multi-start with GRASP and Local Search
    # We will try several GRASP instances with different alphas.
    # Also include the initial rank-matching solution.
    
    # 1. Rank Matching Solution
    p_curr = best_p.copy()
    cost_curr = best_cost
    p_curr, cost_curr, _ = local_search(p_curr, cost_curr)
    if cost_curr < best_cost:
        best_cost = cost_curr
        best_p = p_curr.copy()
        
    # 2. GRASP Solutions
    # We can vary alpha to get different qualities of starting points.
    # Smaller alpha -> more greedy, higher quality start, less diversity.
    # Larger alpha -> more random, lower quality start, more diversity.
    
    alphas = [0.2, 0.3, 0.5, 0.7, 1.0]
    random.shuffle(alphas)
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
            
        p_grasp = grasp_construct(alpha)
        cost_grasp = compute_cost(p_grasp)
        
        # If GRASP solution is already good, it's a good start.
        # Run local search on it.
        p_ls, cost_ls, _ = local_search(p_grasp, cost_grasp)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # 3. If time remains, do some random perturbations of the best solution
    while time.time() < deadline - 2.0:
        # Perturb best_p
        p_pert = best_p.copy()
        # Swap 2 random elements
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        cost_pert = compute_cost(p_pert)
        
        # Run local search
        p_ls, cost_ls, _ = local_search(p_pert, cost_pert)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
