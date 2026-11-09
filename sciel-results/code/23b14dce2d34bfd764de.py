"""Program using Mixed Integer Linear Programming (MILP) for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using MILP.
    
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
    
    def greedy_init():
        """Greedy initialization: assign customers in order of demand (largest first)."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res
    
    def local_search(res, load, max_iter=2000):
        """Local search with relocate moves."""
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
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
    
    # Get initial feasible solution
    res_greedy = greedy_init()
    load_greedy = np.zeros(n_fac)
    np.add.at(load_greedy, res_greedy, demands)
    local_search(res_greedy, load_greedy, max_iter=1000)
    best_assign = res_greedy.copy()
    best_score = compute_score(best_assign)
    
    # Build MILP model
    # Variables:
    # x[i, j] = 1 if customer i is assigned to facility j (binary)
    # y[j] = 1 if facility j is open (binary)
    # Total variables: n_cust * n_fac + n_fac
    
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective: minimize sum of open_cost * y[j] + assign_cost[i,j] * x[i,j]
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c[n_x + j] = open_cost[j]
    
    # Variable bounds: all binary [0, 1]
    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_j x[i,j] = 1 for all i
    # 2. Capacity: sum_i demands[i] * x[i,j] <= capacities[j] for all j
    # 3. Linking: x[i,j] <= y[j] for all i, j
    
    # Build constraint matrix A and bounds
    # Number of constraints: n_cust + n_fac + n_cust * n_fac
    n_constraints = n_cust + n_fac + n_cust * n_fac
    
    A = np.zeros((n_constraints, n_vars))
    lb_c = np.zeros(n_constraints)
    ub_c = np.zeros(n_constraints)
    
    # Constraint 1: Each customer assigned to exactly one facility
    for i in range(n_cust):
        row = i
        for j in range(n_fac):
            A[row, i * n_fac + j] = 1.0
        lb_c[row] = 1.0
        ub_c[row] = 1.0
    
    # Constraint 2: Capacity constraints
    for j in range(n_fac):
        row = n_cust + j
        for i in range(n_cust):
            A[row, i * n_fac + j] = demands[i]
        lb_c[row] = 0.0
        ub_c[row] = capacities[j]
    
    # Constraint 3: Linking constraints x[i,j] <= y[j]
    for i in range(n_cust):
        for j in range(n_fac):
            row = n_cust + n_fac + i * n_fac + j
            A[row, i * n_fac + j] = 1.0
            A[row, n_x + j] = -1.0
            lb_c[row] = -np.inf
            ub_c[row] = 0.0
    
    linear_constraint = LinearConstraint(A, lb_c, ub_c)
    bounds = Bounds(lb, ub)
    
    # Solve MILP
    try:
        result = milp(
            c=c,
            constraints=linear_constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0 and result.x is not None:
            x_sol = result.x[:n_x].reshape(n_cust, n_fac)
            
            # Extract assignment
            res_milp = np.zeros(n_cust, dtype=int)
            for i in range(n_cust):
                # Find the facility with x[i,j] closest to 1
                res_milp[i] = np.argmax(x_sol[i])
            
            # Verify feasibility
            load_milp = np.zeros(n_fac)
            np.add.at(load_milp, res_milp, demands)
            
            if not np.any(load_milp > capacities + 1e-9):
                score_milp = compute_score(res_milp)
                if score_milp < best_score:
                    best_score = score_milp
                    best_assign = res_milp.copy()
        
        # Run local search on the best solution
        load_best = np.zeros(n_fac)
        np.add.at(load_best, best_assign, demands)
        local_search(best_assign, load_best, max_iter=3000)
        
    except Exception:
        # If MILP fails, use the greedy solution
        pass
    
    # Final verification
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        best_assign = res_greedy
    
    return best_assign.astype(int)

# EVOLVE-BLOCK-END
