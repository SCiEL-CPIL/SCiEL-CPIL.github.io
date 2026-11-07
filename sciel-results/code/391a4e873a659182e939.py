import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _solve_knapsack(dp_cap, weights, profits, fixed_cost):
    """
    Solves 0/1 knapsack to maximize profit - fixed_cost.
    Returns (max_profit_minus_fixed, selected_indices).
    If max is negative, returns (0, []).
    """
    n = len(weights)
    if n == 0:
        return 0, []
    
    # DP array
    dp = np.zeros(dp_cap + 1, dtype=float)
    # To track items, we can store the previous state or use a 2D array (too memory heavy if cap is large, but cap is ~2000, n=100, 200k floats is fine)
    # Or we can just store the best profit and backtrack.
    # Backtracking requires knowing which item was taken.
    # Let's use a 2D array for backtracking: taken[i][c] = True if item i was taken to achieve dp[i][c].
    # But 100 * 2000 = 200,000 booleans. This is small.
    
    # Optimization: Only consider items with positive profit
    items = [(w, p) for w, p in zip(weights, profits) if p > 0]
    if not items:
        return 0, []
        
    items.sort(key=lambda x: x[1]/x[0], reverse=True) # Greedy sort helps heuristic, but DP is exact.
    
    n_items = len(items)
    # dp[i][c] = max profit using first i items with capacity c
    # We only need previous row to compute current, but for backtracking we need all.
    # Let's just use a 1D dp and store the choice in a separate structure or recompute?
    # Recomputing is O(N*C). Backtracking is O(N*C).
    
    dp = np.zeros(dp_cap + 1, dtype=float)
    # Store the item index that was added to reach state c from c-w
    # prev[c] = index of item added, or -1
    # This doesn't work for full backtracking path easily with 1D.
    
    # Let's use the standard 2D boolean table for backtracking, it's small enough.
    # taken[i][c] = True if item i is included in the optimal solution for capacity c using items 0..i
    taken = np.zeros((n_items, dp_cap + 1), dtype=bool)
    
    # Initialize dp
    # dp[c] for i=0 (first item)
    w0, p0 = items[0]
    for c in range(w0, dp_cap + 1):
        dp[c] = p0
        taken[0][c] = True
        
    for i in range(1, n_items):
        w, p = items[i]
        # Copy previous row
        prev_dp = dp.copy()
        for c in range(w, dp_cap + 1):
            # Option 1: Don't take item i
            # Option 2: Take item i
            if prev_dp[c - w] + p > dp[c]:
                dp[c] = prev_dp[c - w] + p
                taken[i][c] = True
            else:
                taken[i][c] = False
        # Note: if we didn't take item i, taken[i][c] is False, and we rely on taken[i-1][c]
        # Actually, standard backtracking:
        # If taken[i][c] is True, item i is in solution. Go to c-w, i-1.
        # If taken[i][c] is False, item i is NOT in solution. Go to c, i-1.
        # BUT, the logic above `taken[i][c] = False` doesn't distinguish between "explicitly not taken" and "inherited".
        # We need to be careful.
        # Let's refine:
        # We can just store the decision.
        # If we update dp[c], taken[i][c] = True.
        # If we don't update, taken[i][c] = False.
        # This works.

    best_cap = np.argmax(dp)
    best_profit = dp[best_cap]
    
    if best_profit - fixed_cost <= 0:
        return 0, []
        
    # Backtrack
    selected = []
    c = best_cap
    for i in range(n_items - 1, -1, -1):
        if taken[i][c]:
            selected.append(i)
            c -= items[i][0]
            
    # Map back to original indices
    # We sorted items, so we need to map indices back.
    # This is getting complex. Let's simplify by not sorting for DP, just filter positive profits.
    # The sorting was for heuristic speed, but DP is exact.
    # Let's redo without sorting to keep index mapping simple.
    
    # RE-IMPLEMENTATION SIMPLER
    # Filter positive profits
    pos_indices = [i for i, p in enumerate(profits) if p > 0]
    if not pos_indices:
        return 0, []
        
    w_sub = weights[pos_indices]
    p_sub = profits[pos_indices]
    n_sub = len(pos_indices)
    
    dp = np.zeros(dp_cap + 1, dtype=float)
    taken = np.zeros((n_sub, dp_cap + 1), dtype=bool)
    
    w0, p0 = w_sub[0], p_sub[0]
    for c in range(w0, dp_cap + 1):
        dp[c] = p0
        taken[0][c] = True
        
    for i in range(1, n_sub):
        w, p = w_sub[i], p_sub[i]
        prev_dp = dp.copy()
        for c in range(w, dp_cap + 1):
            if prev_dp[c - w] + p > dp[c]:
                dp[c] = prev_dp[c - w] + p
                taken[i][c] = True
            else:
                taken[i][c] = False
                
    best_cap = np.argmax(dp)
    best_profit = dp[best_cap]
    
    if best_profit - fixed_cost <= 0:
        return 0, []
        
    selected_sub = []
    c = best_cap
    for i in range(n_sub - 1, -1, -1):
        if taken[i][c]:
            selected_sub.append(i)
            c -= w_sub[i]
            
    selected_orig = [pos_indices[idx] for idx in selected_sub]
    return best_profit - fixed_cost, selected_orig

def _lagrangian(demands, capacities, open_cost, assign_cost, lambda_vec, max_iter=100):
    """
    Performs subgradient optimization of the Lagrangian dual.
    Returns best LB and the assignment from the last iteration.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_lb = -np.inf
    best_assignment = None
    
    # Initial lambda
    if lambda_vec is None:
        lambda_vec = np.zeros(n_cust)
    else:
        lambda_vec = lambda_vec.copy()
        
    # Upper bound for step size (current best feasible solution)
    # We don't have one yet, so use a large number or estimate
    # Estimate: sum of min assign costs + min open cost
    ub = np.sum(np.min(assign_cost, axis=1)) + np.min(open_cost)
    
    for it in range(max_iter):
        # Solve sub-problems
        profits = assign_cost - lambda_vec[:, np.newaxis] # (N, M)
        
        total_lagrangian = np.sum(lambda_vec)
        current_assignment = np.full(n_cust, -1, dtype=int)
        
        for j in range(n_fac):
            # Facility j sub-problem
            p_j = profits[:, j]
            w_j = demands
            c_j = capacities[j]
            f_j = open_cost[j]
            
            val, selected = _solve_knapsack(c_j, w_j, p_j, f_j)
            total_lagrangian += val
            
            # Record assignment
            for i in selected:
                # If customer i is already assigned to another facility, it's a conflict
                # For now, we just record the "wish"
                # We'll handle conflicts in the feasible solution construction
                current_assignment[i] = j # Overwrite, we need a better way to track conflicts
                
        # Calculate LB
        lb = total_lagrangian
        
        if lb > best_lb:
            best_lb = lb
            # Save assignment? It's not feasible.
            # We need to construct a feasible solution from this.
            
        # Subgradient
        # g_i = 1 - sum_j x_ij
        # x_ij is 1 if i assigned to j in sub-problem.
        # We need to count how many facilities want each customer.
        counts = np.zeros(n_cust)
        # Re-run sub-problems to get counts? Or store them?
        # Let's store the selection in the loop above.
        
        # Re-do sub-problem solving to get counts and selection properly
        counts = np.zeros(n_cust)
        selections = [None] * n_fac
        for j in range(n_fac):
            p_j = profits[:, j]
            w_j = demands
            c_j = capacities[j]
            f_j = open_cost[j]
            val, selected = _solve_knapsack(c_j, w_j, p_j, f_j)
            selections[j] = selected
            for i in selected:
                counts[i] += 1
                
        # g_i = 1 - counts[i]
        g = 1 - counts
        
        # Step size
        # alpha = (UB - LB) / (||g||^2 + epsilon)
        norm_g = np.sum(g**2)
        if norm_g < 1e-6:
            break
        alpha = (ub - lb) / norm_g
        # Decay step size
        alpha *= 0.9**it
        
        lambda_vec = np.maximum(0, lambda_vec + alpha * g)
        
        # Construct a feasible solution from this iteration
        # Strategy: Assign customers to their most preferred facility that has capacity.
        # Sort customers by "conflict" or random.
        # For each customer, find the facility j that maximized profit (or was selected).
        # If multiple, pick one.
        
        # Let's build a preference list for each customer based on the Lagrangian profits
        # Actually, the sub-problem solution gives the optimal set for each facility.
        # A simple heuristic:
        # 1. For each customer i, identify facilities that selected them.
        # 2. If exactly one, assign there.
        # 3. If multiple, assign to the one with highest profit (c_ij - lambda_i).
        # 4. If none, assign to the facility with min assign cost (c_ij).
        # 5. Check capacities. If full, move to next best.
        
        # This construction is O(N*M*logM).
        
        # We'll do this in the main function to keep this clean.
        
    return best_lb, lambda_vec, counts, selections

def _construct_feasible(demands, capacities, open_cost, assign_cost, selections, lambda_vec):
    """
    Constructs a feasible solution from Lagrangian selections.
    selections[j] is list of customers assigned to j in sub-problem.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    # Calculate profit for each customer-facility pair
    profits = assign_cost - lambda_vec[:, np.newaxis]
    
    # For each customer, get the list of facilities that selected them, sorted by profit descending
    cust_fac_prefs = [[] for _ in range(n_cust)]
    for j in range(n_fac):
        for i in selections[j]:
            cust_fac_prefs[i].append((profits[i, j], j))
            
    for i in range(n_cust):
        if not cust_fac_prefs[i]:
            # No facility selected this customer in Lagrangian.
            # Assign to facility with min assign cost
            min_cost_fac = np.argmin(assign_cost[i])
            cust_fac_prefs[i].append((assign_cost[i, min_cost_fac], min_cost_fac))
            
        # Sort by profit descending
        cust_fac_prefs[i].sort(reverse=True)
        
        assigned = False
        for p, j in cust_fac_prefs[i]:
            if load[j] + demands[i] <= capacities[j]:
                result[i] = j
                load[j] += demands[i]
                assigned = True
                break
                
        if not assigned:
            # Fallback: find any facility with space
            for j in range(n_fac):
                if load[j] + demands[i] <= capacities[j]:
                    result[i] = j
                    load[j] += demands[i]
                    assigned = True
                    break
                    
        if not assigned:
            # Infeasible? Should not happen if total capacity > total demand
            # Assign to first facility (will be invalid)
            result[i] = 0
            load[0] += demands[i]
            
    return result, load

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    n_cust = len(result)
    n_fac = len(capacities)
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        
    # Check validity
    for j in range(n_fac):
        if load[j] > capacities[j]:
            return float('inf'), load
            
    total = 0.0
    open_fac = set()
    for i in range(n_cust):
        f = result[i]
        total += assign_cost[i, f]
        open_fac.add(f)
        
    for f in open_fac:
        total += open_cost[f]
        
    return total, load

def _local_search(demands, capacities, open_cost, assign_cost, init_result, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    result = init_result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        
    current_cost, _ = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if current_cost == float('inf'):
        return result, current_cost
        
    best_cost = current_cost
    best_result = result.copy()
    
    start_time = time.time()
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    # Precompute assign_cost for fast access
    ac = assign_cost
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count = 0
        
        # Try single moves
        # Shuffle customer order for diversity
        order = np.random.permutation(n_cust)
        
        for i in order:
            if time.time() > deadline:
                break
                
            cur_f = result[i]
            d_i = demands[i]
            
            # Try moving to all other facilities
            # Sort facilities by assign cost for this customer
            fac_order = np.argsort(ac[i])
            
            for new_f in fac_order:
                if new_f == cur_f:
                    continue
                    
                if load[new_f] + d_i <= capacities[new_f]:
                    # Calculate delta cost
                    delta_assign = ac[i, new_f] - ac[i, cur_f]
                    delta_open = 0.0
                    
                    # Check open cost changes
                    # cur_f becomes empty?
                    if load[cur_f] - d_i == 0:
                        delta_open -= open_cost[cur_f]
                        
                    # new_f becomes non-empty?
                    if load[new_f] == 0:
                        delta_open += open_cost[new_f]
                        
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Accept move
                        result[i] = new_f
                        load[cur_f] -= d_i
                        load[new_f] += d_i
                        current_cost += total_delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = result.copy()
                        improved = True
                        break # Move to next customer
                    
        # Try pair swaps
        # This is more expensive. Only do if single moves stopped improving?
        # Or do it every iteration?
        # Let's do it every iteration but limit the number of swaps tried.
        
        if not improved:
            # Try swaps
            swap_improved = False
            # Pick random pairs
            for _ in range(100): # Limit swaps per iteration
                if time.time() > deadline:
                    break
                i, k = np.random.choice(n_cust, 2, replace=False)
                f_i = result[i]
                f_k = result[k]
                
                if f_i == f_k:
                    continue
                    
                d_i = demands[i]
                d_k = demands[k]
                
                # Check capacities
                new_load_i = load[f_i] - d_i + d_k
                new_load_k = load[f_k] - d_k + d_i
                
                if new_load_i <= capacities[f_i] and new_load_k <= capacities[f_k]:
                    # Calculate delta
                    delta_assign = (ac[i, f_k] + ac[k, f_i]) - (ac[i, f_i] + ac[k, f_k])
                    # Open cost doesn't change for swaps between non-empty facilities
                    # (as argued before)
                    # Exception: if one facility becomes empty?
                    # If load[f_i] == d_i, then new_load_i = d_k. Since d_k > 0, f_i is not empty.
                    # So open cost is constant.
                    
                    if delta_assign < -1e-9:
                        # Accept swap
                        result[i] = f_k
                        result[k] = f_i
                        load[f_i] = new_load_i
                        load[f_k] = new_load_k
                        current_cost += delta_assign
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = result.copy()
                        swap_improved = True
                        improved = True
                        break
                        
            if not swap_improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
    return best_result, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Lagrangian Relaxation to get initial solution
    # Run a few iterations of subgradient
    lambda_vec = None
    best_lb = -np.inf
    best_selections = None
    
    # We need an initial feasible solution for UB
    # Greedy: assign to cheapest facility with capacity
    greedy_result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands) # Largest first
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                greedy_result[i] = f
                load[f] += demands[i]
                break
        if greedy_result[i] == -1:
            # Fallback
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    greedy_result[i] = f
                    load[f] += demands[i]
                    break
                    
    greedy_cost, _ = _compute_cost(greedy_result, demands, capacities, open_cost, assign_cost)
    
    # Run Lagrangian
    max_lag_iter = 50
    for it in range(max_lag_iter):
        if time.time() > deadline - 20: # Leave time for local search
            break
            
        # Subgradient step
        profits = assign_cost - (lambda_vec if lambda_vec is not None else np.zeros(n_cust))[:, np.newaxis]
        
        total_lagrangian = np.sum(lambda_vec) if lambda_vec is not None else 0
        selections = [None] * n_fac
        counts = np.zeros(n_cust)
        
        for j in range(n_fac):
            p_j = profits[:, j]
            w_j = demands
            c_j = capacities[j]
            f_j = open_cost[j]
            
            val, selected = _solve_knapsack(c_j, w_j, p_j, f_j)
            total_lagrangian += val
            selections[j] = selected
            for i in selected:
                counts[i] += 1
                
        lb = total_lagrangian
        
        if lb > best_lb:
            best_lb = lb
            best_selections = selections
            
        g = 1 - counts
        norm_g = np.sum(g**2)
        if norm_g < 1e-6:
            break
            
        alpha = (greedy_cost - lb) / norm_g
        alpha *= 0.9**it
        
        if lambda_vec is None:
            lambda_vec = alpha * g
        else:
            lambda_vec = np.maximum(0, lambda_vec + alpha * g)
            
    # Construct feasible solution from best selections
    if best_selections is not None:
        lag_result, _ = _construct_feasible(demands, capacities, open_cost, assign_cost, best_selections, lambda_vec)
    else:
        lag_result = greedy_result
        
    # 2. Local Search
    # Run local search on both greedy and lag_result
    ls_deadline = deadline - 5.0
    
    best_overall_cost = float('inf')
    best_overall_result = greedy_result
    
    # Polish Greedy
    res_g, cost_g = _local_search(demands, capacities, open_cost, assign_cost, greedy_result, ls_deadline)
    if cost_g < best_overall_cost:
        best_overall_cost = cost_g
        best_overall_result = res_g
        
    # Polish Lagrangian
    if time.time() < ls_deadline:
        res_l, cost_l = _local_search(demands, capacities, open_cost, assign_cost, lag_result, ls_deadline)
        if cost_l < best_overall_cost:
            best_overall_cost = cost_l
            best_overall_result = res_l
            
    return best_overall_result
