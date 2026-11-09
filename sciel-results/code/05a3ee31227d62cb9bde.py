import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using MILP formulation.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def greedy_init(seed):
        """Greedy initialization with randomization."""
        random.seed(seed)
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        order = np.argsort(-demands)
        
        for i in order:
            fac_costs = assign_cost[i]
            
            noise = np.random.uniform(0, 1, n_fac) * np.std(fac_costs)
            noisy_costs = fac_costs + noise
            fac_order_noisy = np.argsort(noisy_costs)
            
            assigned = False
            for f in fac_order_noisy:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                space = capacities - load
                max_space = np.max(space)
                candidates = np.where(space == max_space)[0]
                f = random.choice(candidates.tolist())
                res[i] = f
                load[f] += demands[i]
                
        return res
    
    def local_search(res, load, max_iter=5000):
        """Local search with relocate and swap moves."""
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            # --- Phase 1: Relocate Moves ---
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_delta = 1e-9
                best_f = -1
                
                curr_cost_i = assign_cost[i, current_f]
                
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    
                    delta = assign_cost[i, f_new] - curr_cost_i
                    
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
            
            # --- Phase 2: Swap Moves ---
            if it % 5 == 0:
                swapped = False
                pairs = []
                for i in range(n_cust):
                    for j in range(i+1, n_cust):
                        pairs.append((i, j))
                random.shuffle(pairs)
                
                for i, j in pairs[:200]:
                    if time.time() > deadline:
                        break
                        
                    f_i = res[i]
                    f_j = res[j]
                    
                    if f_i == f_j:
                        continue
                    
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < -1e-9:
                        load[f_i] += demands[j] - demands[i]
                        load[f_j] += demands[i] - demands[j]
                        res[i], res[j] = f_j, f_i
                        improved = True
                        swapped = True
                        break
        
        return res
    
    def solve_milp():
        """
        Solve the capacitated facility location problem as a MILP.
        
        Variables:
        - x[i, f] = 1 if customer i is assigned to facility f (binary)
        - y[f] = 1 if facility f is open (binary)
        
        Objective: minimize sum_f open_cost[f] * y[f] + sum_{i,f} assign_cost[i,f] * x[i,f]
        
        Constraints:
        1. Each customer assigned to exactly one facility: sum_f x[i,f] = 1 for all i
        2. Capacity constraint: sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
        3. Facility opening: x[i,f] <= y[f] for all i, f
        """
        # Total number of variables: n_cust * n_fac (x) + n_fac (y)
        n_x = n_cust * n_fac
        n_y = n_fac
        n_vars = n_x + n_y
        
        # Objective coefficients
        c = np.zeros(n_vars)
        # x[i,f] coefficient: assign_cost[i,f]
        for i in range(n_cust):
            for f in range(n_fac):
                c[i * n_fac + f] = assign_cost[i, f]
        # y[f] coefficient: open_cost[f]
        for f in range(n_fac):
            c[n_x + f] = open_cost[f]
        
        # Variable bounds: all binary
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        # Constraints
        A_rows = []
        lb_list = []
        ub_list = []
        
        # Constraint 1: sum_f x[i,f] = 1 for each customer i
        for i in range(n_cust):
            row = np.zeros(n_vars)
            for f in range(n_fac):
                row[i * n_fac + f] = 1.0
            A_rows.append(row)
            lb_list.append(1.0)
            ub_list.append(1.0)
        
        # Constraint 2: sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for each facility f
        # Rearranged: sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
        for f in range(n_fac):
            row = np.zeros(n_vars)
            for i in range(n_cust):
                row[i * n_fac + f] = demands[i]
            row[n_x + f] = -capacities[f]
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
        
        # Constraint 3: x[i,f] <= y[f] for all i, f
        # Rearranged: x[i,f] - y[f] <= 0
        for i in range(n_cust):
            for f in range(n_fac):
                row = np.zeros(n_vars)
                row[i * n_fac + f] = 1.0
                row[n_x + f] = -1.0
                A_rows.append(row)
                lb_list.append(-np.inf)
                ub_list.append(0.0)
        
        A = np.array(A_rows)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        constraint = LinearConstraint(A, lb, ub)
        
        # Solve MILP
        try:
            result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if result.status == 0:  # Success
                x_flat = result.x[:n_x]
                y = result.x[n_x:]
                
                # Extract assignment
                res = np.zeros(n_cust, dtype=int)
                for i in range(n_cust):
                    for f in range(n_fac):
                        if x_flat[i * n_fac + f] > 0.5:
                            res[i] = f
                            break
                
                # Verify feasibility
                load = np.zeros(n_fac)
                np.add.at(load, res, demands)
                if np.any(load > capacities + 1e-9):
                    return None
                
                # Verify each customer is assigned
                if np.any(res < 0) or np.any(res >= n_fac):
                    return None
                
                return res
            else:
                return None
        except Exception as e:
            return None
    
    # Try MILP first
    milp_result = solve_milp()
    
    if milp_result is not None and time.time() < deadline:
        score = compute_score(milp_result)
        if np.isfinite(score):
            return milp_result.astype(int)
    
    # Fallback to heuristic if MILP fails or times out
    best_assign = None
    best_score = np.inf
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        res = greedy_init(r)
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        # Ensure feasibility
        if np.any(load > capacities + 1e-9):
            for i in range(n_cust):
                if load[res[i]] > capacities[res[i]] + 1e-9:
                    for f in range(n_fac):
                        if load[f] + demands[i] <= capacities[f] + 1e-9:
                            load[res[i]] -= demands[i]
                            load[f] += demands[i]
                            res[i] = f
                            break
        
        local_search(res, load, max_iter=100)
        
        score = compute_score(res)
        if score < best_score:
            best_score = score
            best_assign = res.copy()
            
    if best_assign is not None:
        return best_assign.astype(int)
    else:
        # Last resort: simple greedy
        res = greedy_init(0)
        return res.astype(int)
