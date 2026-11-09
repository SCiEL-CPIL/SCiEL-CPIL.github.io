"""Capacitated facility location: Lagrangian Relaxation Initialization + Vectorized Iterated Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 145s max
    deadline = time.time() + 145.0
    
    d_arr = np.array(demands)
    cap_arr = np.array(capacities)
    oc_arr = np.array(open_cost)
    ac_arr = np.array(assign_cost)
    
    # 1. Lagrangian Relaxation Initialization
    # We relax the capacity constraints: sum_i d_i x_ij <= cap_j
    # Lagrangian: L(x, mu) = sum_i,j c_ij x_ij + sum_j f_j * max(0, sum_i d_i x_ij) + sum_j mu_j (sum_i d_i x_ij - cap_j)
    # For a fixed mu, the problem decomposes by customer i.
    # For customer i, we minimize over j: c_ij + f_j * (1 if x_ij=1 and load_j>0? No, in Lagrangian the fixed cost is tricky if load is fractional or if we treat it as part of the relaxation).
    # Actually, standard Lagrangian for CFL usually relaxes capacity.
    # Let's use a simpler heuristic initialization that mimics Lagrangian dual:
    # Minimize c_ij + mu_j * d_i.
    # We can find good mu values by subgradient method or just random sampling.
    # Given time constraints, we will generate multiple initial solutions using different "price" vectors mu.
    
    def generate_initial_solution(mu_vec):
        # Effective cost for customer i to facility j: c_ij + mu_j * d_i
        # Note: We ignore open_cost in the initial assignment logic of this step, 
        # or we can approximate it. Let's add a small penalty for opening.
        # Actually, to get a diverse set of starting points, we can vary mu.
        
        # If mu is large, we penalize loading facilities heavily, leading to spread.
        # If mu is 0, we just pick cheapest c_ij.
        
        eff_costs = ac_arr + np.outer(d_arr, mu_vec)
        
        # Greedy assignment with capacity check
        # Sort customers by demand descending (large first is safer for feasibility)
        order = np.argsort(-d_arr)
        
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        for i in order:
            # Find feasible facility with min effective cost
            feasible = np.where(load + d_arr[i] <= cap_arr)[0]
            
            if len(feasible) == 0:
                # Fallback: pick facility with max remaining capacity
                # This shouldn't happen if total cap >= total demand
                f = np.argmax(cap_arr - load)
                res[i] = f
                load[f] += d_arr[i]
                continue
            
            # Among feasible, pick min cost
            min_idx = np.argmin(eff_costs[i, feasible])
            f_new = feasible[min_idx]
            
            res[i] = f_new
            load[f_new] += d_arr[i]
            
        return res

    # Generate a pool of initial solutions using different mu vectors
    # mu_j represents the "shadow price" of capacity at facility j.
    # We try:
    # 1. mu = 0 (pure cost)
    # 2. mu = uniform random
    # 3. mu proportional to open_cost (encourages using cheap-to-open facilities? No, open cost is fixed. Maybe inverse capacity?)
    # 4. mu based on density of demand?
    
    initial_solutions = []
    
    # Solution 1: Pure cost greedy
    initial_solutions.append(generate_initial_solution(np.zeros(n_fac)))
    
    # Solution 2: Random mu
    np.random.seed(42) # Fixed seed for reproducibility of the *algorithm structure*, though instance is fixed
    for _ in range(5):
        mu = np.random.rand(n_fac) * 10.0
        initial_solutions.append(generate_initial_solution(mu))
        
    # Solution 3: Heuristic mu based on open cost (high open cost -> high mu to avoid opening unless necessary? 
    # Actually, if mu is high, we avoid assigning there. If a facility has high open cost, we *want* to avoid it if possible, 
    # but if we must open it, we want to fill it. 
    # Let's try mu proportional to open_cost to discourage usage of expensive facilities initially?
    # Or maybe mu = 0 for all, but sort facilities by open_cost?
    # Let's try mu = open_cost / 100
    initial_solutions.append(generate_initial_solution(oc_arr / 100.0))
    
    # Solution 4: mu = 1/capacity (discourage filling small facilities?)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_cap = 1.0 / cap_arr
        mu_cap[~np.isfinite(mu_cap)] = 0
    initial_solutions.append(generate_initial_solution(mu_cap))
    
    # Evaluate and keep the best initial solution
    def calc_cost(res):
        if not np.all(np.isin(res, range(n_fac))):
            return np.inf
        # Check capacities
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += d_arr[i]
        if np.any(load > cap_arr):
            return np.inf
            
        used = np.unique(res)
        return np.sum(oc_arr[used]) + np.sum(ac_arr[np.arange(n_cust), res])
    
    best_init_res = None
    best_init_score = np.inf
    
    for sol in initial_solutions:
        score = calc_cost(sol)
        if score < best_init_score:
            best_init_score = score
            best_init_res = sol.copy()
            
    if best_init_res is None:
        # Fallback: simple greedy
        best_init_res = generate_initial_solution(np.zeros(n_fac))
        best_init_score = calc_cost(best_init_res)
        
    # 2. Vectorized Iterated Local Search (ILS)
    # We will run ILS starting from best_init_res.
    # Moves:
    # 1. Single customer move: Move customer i from f_old to f_new.
    # 2. Swap: Swap customers i and j.
    # 3. Close Facility: Try to close facility k by moving all its customers to others. (Expensive, do sparingly)
    
    current_res = best_init_res.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_res[i]] += d_arr[i]
    current_score = best_init_score
    
    best_res = current_res.copy()
    best_score = current_score
    
    # Precompute for speed
    d_float = d_arr.astype(float)
    
    def try_single_moves(res, load, score):
        improved = False
        # Vectorized check for all customers
        # For each customer i, we want to find best f_new != f_old such that:
        # load[f_new] + d_i <= cap[f_new]
        # Delta = (ac[i, f_new] - ac[i, f_old]) + (oc[f_new] if load[f_new]==0 else 0) - (oc[f_old] if load[f_old]==d_i else 0)
        
        # This is hard to fully vectorize due to the "if load==0" and "if load==d_i" conditions.
        # We will use a semi-vectorized approach: iterate customers, but vectorize the facility selection.
        
        for i in range(n_cust):
            if time.time() > deadline - 2.0:
                return res, load, score, False
                
            f_old = res[i]
            d_i = d_arr[i]
            
            # Feasible targets: load + d_i <= cap, and f != f_old
            feasible_mask = (load + d_i <= cap_arr) & (np.arange(n_fac) != f_old)
            
            if not np.any(feasible_mask):
                continue
                
            feasible_indices = np.where(feasible_mask)[0]
            
            # Calculate delta costs
            # Base delta: ac[i, feasible] - ac[i, f_old]
            delta_base = ac_arr[i, feasible_indices] - ac_arr[i, f_old]
            
            # Open cost adjustments
            delta_open = np.zeros(len(feasible_indices))
            
            # If f_old becomes empty, we save oc[f_old]
            if load[f_old] == d_i:
                delta_open -= oc_arr[f_old]
                
            # If target is empty, we pay oc[target]
            # Identify empty targets in feasible_indices
            empty_targets_mask = (load[feasible_indices] == 0)
            if np.any(empty_targets_mask):
                empty_target_indices = feasible_indices[empty_targets_mask]
                delta_open[empty_targets_mask] += oc_arr[empty_target_indices]
                
            total_delta = delta_base + delta_open
            
            min_idx = np.argmin(total_delta)
            best_delta = total_delta[min_idx]
            
            if best_delta < -1e-9:
                f_new = feasible_indices[min_idx]
                # Apply move
                res[i] = f_new
                load[f_old] -= d_i
                load[f_new] += d_i
                score += best_delta
                improved = True
                
        return res, load, score, improved

    def try_swaps(res, load, score, n_samples=200):
        improved = False
        for _ in range(n_samples):
            if time.time() > deadline - 2.0:
                return res, load, score, False
                
            i, j = np.random.randint(0, n_cust, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            f_i = res[i]
            f_j = res[j]
            if f_i == f_j:
                continue
                
            d_i = d_arr[i]
            d_j = d_arr[j]
            
            # Check capacity
            if load[f_j] - d_j + d_i > cap_arr[f_j]:
                continue
            if load[f_i] - d_i + d_j > cap_arr[f_i]:
                continue
                
            # Calculate delta
            # Swap i and j
            delta_assign = (ac_arr[i, f_j] - ac_arr[i, f_i]) + (ac_arr[j, f_i] - ac_arr[j, f_j])
            
            # Open cost change:
            # If f_i becomes empty (load[f_i] == d_i) AND f_j doesn't become empty (load[f_j] - d_j + d_i > 0? No, j moves out, i moves in.
            # load_new_f_i = load[f_i] - d_i + d_j.
            # load_new_f_j = load[f_j] - d_j + d_i.
            
            # Old open cost contribution: oc[f_i] if load[f_i]>0 else 0, oc[f_j] if load[f_j]>0 else 0.
            # New open cost contribution: oc[f_i] if load_new_f_i>0 else 0, oc[f_j] if load_new_f_j>0 else 0.
            
            old_open_i = oc_arr[f_i] if load[f_i] > 0 else 0.0
            old_open_j = oc_arr[f_j] if load[f_j] > 0 else 0.0
            
            new_load_i = load[f_i] - d_i + d_j
            new_load_j = load[f_j] - d_j + d_i
            
            new_open_i = oc_arr[f_i] if new_load_i > 0 else 0.0
            new_open_j = oc_arr[f_j] if new_load_j > 0 else 0.0
            
            delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
            
            total_delta = delta_assign + delta_open
            
            if total_delta < -1e-9:
                # Apply swap
                res[i], res[j] = res[j], res[i]
                load[f_i] = new_load_i
                load[f_j] = new_load_j
                score += total_delta
                improved = True
                
        return res, load, score, improved

    def try_close_facility(res, load, score):
        # Try to close one facility to save open cost
        # Pick a facility with low load relative to capacity? Or just random?
        # Let's pick the facility with the smallest open_cost among those that can be closed?
        # Or the one with fewest customers?
        
        # We will sample a facility that is currently open.
        open_facs = np.where(load > 0)[0]
        if len(open_facs) <= 1:
            return res, load, score, False
            
        # Try to close one
        # Strategy: Pick the facility with the smallest "savings per customer moved" ?
        # Let's just try to close the facility with the smallest load (easiest to move customers)
        # Or smallest open cost (least penalty to lose)?
        # Let's try the one with smallest load.
        
        # Sort open facilities by load
        open_fac_loads = load[open_facs]
        min_load_idx = np.argmin(open_fac_loads)
        f_to_close = open_facs[min_load_idx]
        
        # If load is 0, it's already closed.
        if load[f_to_close] == 0:
            return res, load, score, False
            
        # Get customers in f_to_close
        custs_in_fac = np.where(res == f_to_close)[0]
        
        if len(custs_in_fac) == 0:
            return res, load, score, False
            
        # For each customer in f_to_close, find best alternative
        # We must move ALL of them.
        # This is a sub-assignment problem.
        # Since the set is small (hopefully), we can use a simple greedy or even brute force if very small.
        # Let's use a greedy approach with backtracking if needed? 
        # Or just simple greedy: sort by demand, assign to cheapest feasible.
        
        new_res = res.copy()
        new_load = load.copy()
        
        # Sort customers by demand descending to pack
        order = np.argsort(-d_arr[custs_in_fac])
        
        feasible_moves = True
        temp_load = new_load.copy()
        temp_res = new_res.copy()
        
        # First, remove the customers from f_to_close
        temp_load[f_to_close] = 0
        
        for idx in order:
            i = custs_in_fac[idx]
            d_i = d_arr[i]
            
            # Feasible facilities: temp_load + d_i <= cap, and f != f_to_close
            feasible = (temp_load + d_i <= cap_arr) & (np.arange(n_fac) != f_to_close)
            
            if not np.any(feasible):
                feasible_moves = False
                break
                
            # Pick min cost
            # Note: We need to account for open costs of *new* facilities opened.
            # But for simplicity in this heuristic step, we just look at assign cost + open cost if empty.
            costs = ac_arr[i, feasible]
            open_penalties = np.zeros(np.sum(feasible))
            fac_indices = np.where(feasible)[0]
            
            # If a facility is empty in temp_load, add open cost
            empty_mask = temp_load[fac_indices] == 0
            if np.any(empty_mask):
                open_penalties[empty_mask] = oc_arr[fac_indices[empty_mask]]
                
            total_costs = costs + open_penalties
            min_idx = np.argmin(total_costs)
            f_new = fac_indices[min_idx]
            
            temp_res[i] = f_new
            temp_load[f_new] += d_i
            
        if feasible_moves:
            # Calculate the score change
            # Old score contribution from f_to_close: oc[f_to_close] + sum_{i in f_to_close} ac[i, f_to_close]
            old_contrib = oc_arr[f_to_close] + np.sum(ac_arr[custs_in_fac, f_to_close])
            
            # New score contribution:
            # Sum of ac[i, new_f] for these customers + sum of oc[new_f] for newly opened facilities
            new_contrib = np.sum(ac_arr[custs_in_fac, temp_res[custs_in_fac]])
            
            # Identify newly opened facilities
            # A facility is newly opened if it was empty in 'load' but has >0 in 'temp_load'
            # And it's not f_to_close.
            newly_opened = np.where((load == 0) & (temp_load > 0))[0]
            if len(newly_opened) > 0:
                new_contrib += np.sum(oc_arr[newly_opened])
                
            delta = new_contrib - old_contrib
            
            if delta < -1e-9:
                res[:] = temp_res
                load[:] = temp_load
                score += delta
                return res, load, score, True
                
        return res, load, score, False

    # ILS Loop
    # We run ILS until time runs out.
    # At each step, we try to improve via local search.
    # If no improvement, we apply a perturbation (random move/swap) and continue.
    
    iter_count = 0
    max_iters = 5000 # Safety cap
    
    while time.time() < deadline - 5.0 and iter_count < max_iters:
        iter_count += 1
        
        # 1. Local Search Phase
        # Alternate between single moves and swaps
        improved = True
        while improved:
            improved = False
            # Try single moves
            current_res, current_load, current_score, imp1 = try_single_moves(current_res, current_load, current_score)
            improved |= imp1
            
            if time.time() > deadline - 5.0:
                break
                
            # Try swaps
            current_res, current_load, current_score, imp2 = try_swaps(current_res, current_load, current_score)
            improved |= imp2
            
            if time.time() > deadline - 5.0:
                break
                
            # Occasionally try close facility (expensive)
            if iter_count % 10 == 0:
                current_res, current_load, current_score, imp3 = try_close_facility(current_res, current_load, current_score)
                improved |= imp3
                
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_res = current_res.copy()
            
        # 2. Perturbation Phase (if no improvement or always to escape local optima)
        # Apply a random perturbation: k random swaps or moves
        k = np.random.randint(1, 4) # 1 to 3 perturbations
        for _ in range(k):
            if time.time() > deadline - 5.0:
                break
                
            if np.random.rand() < 0.5:
                # Random swap
                i, j = np.random.randint(0, n_cust, size=2)
                if i != j:
                    f_i = current_res[i]
                    f_j = current_res[j]
                    if f_i != f_j:
                        d_i = d_arr[i]
                        d_j = d_arr[j]
                        if current_load[f_j] - d_j + d_i <= cap_arr[f_j] and \
                           current_load[f_i] - d_i + d_j <= cap_arr[f_i]:
                            current_res[i], current_res[j] = current_res[j], current_res[i]
                            current_load[f_i] -= d_i
                            current_load[f_i] += d_j
                            current_load[f_j] -= d_j
                            current_load[f_j] += d_i
            else:
                # Random move
                i = np.random.randint(0, n_cust)
                f_old = current_res[i]
                d_i = d_arr[i]
                feasible = (current_load + d_i <= cap_arr) & (np.arange(n_fac) != f_old)
                if np.any(feasible):
                    f_new = feasible[np.random.choice(np.sum(feasible))]
                    current_res[i] = f_new
                    current_load[f_old] -= d_i
                    current_load[f_new] += d_i
                    
        # Recalculate score after perturbation (it might be worse)
        current_score = calc_cost(current_res)
        if not np.isfinite(current_score):
            # If perturbation made it invalid (shouldn't happen with checks, but just in case)
            # Restore from best
            current_res = best_res.copy()
            current_load = np.zeros(n_fac)
            for i in range(n_cust):
                current_load[current_res[i]] += d_arr[i]
            current_score = best_score

    # Final check
    final_score = calc_cost(best_res)
    if final_score < best_score:
        best_score = final_score
        # best_res is already the one associated with best_score if we updated it correctly
        # But let's ensure consistency
        # Actually, best_res is updated only when current_score < best_score.
        # So best_res corresponds to best_score.
        
    return best_res
# EVOLVE-BLOCK-END
