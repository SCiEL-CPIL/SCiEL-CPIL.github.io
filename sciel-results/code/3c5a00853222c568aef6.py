"""Capacitated facility location: Lagrangian Relaxation with Subgradient Method and Greedy Repair."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation.
    
    Strategy:
    1. Relax capacity constraints with multipliers (lambda).
    2. Solve the decomposed problem: min sum(c_ij + lambda_j * d_i) x_ij + fixed costs.
       Note: Fixed costs (open_cost) are tricky in Lagrangian. 
       Instead, we use a "penalty" approach where we add lambda_j * (load_j - cap_j) to the cost.
       Actually, standard Lagrangian for CFL:
       L(lambda) = min_x sum(c_ij x_ij) + sum(f_j y_j) + sum_j lambda_j (sum_i d_i x_ij - cap_j)
       s.t. sum_j x_ij = 1, x_ij in {0,1}, y_j in {0,1}, x_ij <= y_j.
       
       If we fix y_j (or handle it via thresholding), the inner problem decomposes by customer:
       For each customer i, choose j minimizing c_ij + lambda_j * d_i.
       However, this ignores the fixed cost f_j. 
       
       Alternative: Use a "Cost Modification" heuristic.
       We maintain dual variables lambda_j >= 0.
       Effective cost for assigning i to j is c_ij + lambda_j * d_i.
       We assign each i to argmin_j (c_ij + lambda_j * d_i).
       This gives a tentative assignment.
       
       Then we update lambda_j based on capacity violation:
       if load_j > cap_j: lambda_j += step * (load_j - cap_j)
       if load_j < cap_j: lambda_j = max(0, lambda_j - step * (cap_j - load_j))
       
       This minimizes the Lagrangian dual. The solution to the subproblem is often infeasible 
       (violates capacities) or has "wrong" fixed costs (opens too many/too few).
       
       To get a feasible solution, we use a Greedy Repair:
       1. Take the assignment from the subproblem.
       2. If infeasible, move customers from overloaded facilities to underloaded ones 
          based on marginal cost + penalty.
       3. Also, we need to account for open_cost. 
          The Lagrangian dual provides a lower bound, but the actual solution quality depends on 
          how well we reconstruct a feasible solution.
          
       Better Approach for this specific task structure:
       The problem is small (100 customers, 20 facilities).
       We can use the Lagrangian multipliers to guide a **Greedy Construction** or **Local Search**.
       
       Let's stick to the Subgradient method on the Lagrangian, but with a robust repair.
       Key Insight: The subproblem solution x*(lambda) is a "relaxed" assignment.
       We can use it as a starting point for a local search that *only* moves customers 
       to fix capacity violations, minimizing the increase in assignment cost.
       
       However, the prompt asks for a STRUCTURAL change. Lagrangian Relaxation is structurally 
       different from pure local search (which starts from a feasible solution and moves).
       Lagrangian starts from infeasible (relaxed) solutions and projects back.
       
       Implementation Plan:
       1. Initialize lambda = 0.
       2. Loop until time limit:
          a. Solve subproblem: For each i, j_i = argmin_j (assign_cost[i,j] + lambda[j] * demands[i]).
          b. Calculate loads.
          c. If feasible, calculate true cost (with open_cost). Update best.
          d. Update lambda using subgradient: 
             grad_j = load_j - cap_j
             step = (2 * (UB - LB)) / (sum(grad^2) + epsilon)  # Polyak step size
             lambda = max(0, lambda + step * grad)
          e. To improve UB (Upper Bound / Feasible Solution):
             - Use the current x* as a guide.
             - Run a "Feasibility Repair": 
               * If overloaded, move customers with smallest "penalty" (cost increase) to feasible slots.
               * If we have spare capacity, we might be able to close facilities? 
                 The subproblem doesn't explicitly handle y_j (open/close).
                 We can infer y_j = 1 if load_j > 0.
                 The true cost is sum(open_cost[used]) + sum(assign_cost).
                 
       Refinement: The subproblem ignores open_cost. This is a weakness.
       To incorporate open_cost, we can use a "Surrogate" cost:
       c_ij_eff = assign_cost[i,j] + open_cost[j] * (demands[i] / cap_j)  # Proportional sharing?
       This is heuristic. 
       
       Let's use a **Lagrangian with Fixed Cost Approximation**:
       We treat open_cost as a fixed charge. In the subproblem, we can't easily decompose fixed costs.
       However, we can use the Lagrangian multipliers to bias towards facilities with "cheaper" 
       effective capacity.
       
       Alternative Structural Approach: **Iterative Rounding / Greedy with Dual Guidance**
       1. Solve LP Relaxation? (Too slow in pure Python/NumPy without CVXPY, but SciPy has linprog).
          LP Relaxation of CFL is NP-hard to solve exactly, but the relaxation (x_ij in [0,1]) is an LP.
          100*20 variables = 2000 vars. Constraints: 100 (assignment) + 20 (capacity) + 2000 (x<=y) + 20 (y<=1).
          This is solvable by linprog in reasonable time? Maybe.
          
       Let's try **Linprog (LP Relaxation) + Rounding**.
       1. Formulate LP:
          Minimize sum(c_ij x_ij) + sum(f_j y_j)
          Subject to:
          sum_j x_ij = 1 for all i
          sum_i d_i x_ij <= cap_j * y_j for all j
          x_ij <= y_j for all i,j
          x_ij >= 0, y_j >= 0, y_j <= 1
          
       2. Solve LP.
       3. Round the solution:
          - Sort facilities by y_j descending. Open top K.
          - Assign customers to open facilities greedily based on x_ij or cost.
          - Repair capacity.
       
       This is a fundamentally different approach: **Continuous Relaxation + Rounding**.
       
       Let's implement Linprog.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    deadline = time.time() + 140.0
    
    # Variables: x[0..N-1, 0..M-1] flattened, then y[0..M-1]
    # Total vars: N*M + M
    # Objective: c_flat + f
    # Constraints:
    # 1. sum_j x_ij = 1 (N constraints, equality)
    # 2. sum_i d_i x_ij - cap_j y_j <= 0 (M constraints, inequality)
    # 3. x_ij - y_j <= 0 (N*M constraints, inequality) -- This is too many for standard A_ub?
    #    N*M = 2000. M=20. N=100. Total 2000 + 20 = 2020 inequality constraints.
    #    Plus 100 equality.
    #    Variables: 2020.
    #    This is a dense LP. SciPy linprog might be slow but let's try.
    
    # Actually, we can simplify.
    # If y_j is binary, x_ij <= y_j is redundant if we enforce capacity? No.
    # But in LP, y_j can be fractional.
    
    # Let's construct the matrices.
    c_flat = assign_cost.flatten()
    f_vec = open_cost
    c_obj = np.concatenate([c_flat, f_vec])
    
    # A_ub: 
    # Rows 1..M: Capacity constraints.
    # For each facility j: sum_i d_i * x_ij - cap_j * y_j <= 0
    # Row j (0-indexed j):
    #   Coeffs for x_ij: demands[i]
    #   Coeff for y_j: -capacities[j]
    
    # Rows M+1..M+N*M: x_ij - y_j <= 0
    # This is 2000 rows.
    
    # Total A_ub rows: 20 + 2000 = 2020.
    # A_ub cols: 2020.
    
    # This is a 2020x2020 matrix. Dense.
    # Construction time might be high. 
    # Let's see if we can avoid the x<=y constraints by using a different formulation?
    # If we just use capacity constraints, a facility with y_j=0 could still have x_ij>0 if cap_j is large?
    # No, if y_j=0, cap_j*y_j = 0, so sum d_i x_ij <= 0 => x_ij=0.
    # So x_ij <= y_j is implied by the capacity constraint IF we assume d_i > 0.
    # Wait, if y_j=0, RHS is 0. LHS is sum d_i x_ij. Since d_i>0 and x_ij>=0, x_ij must be 0.
    # So x_ij <= y_j is redundant given the capacity constraint and non-negativity!
    # 
    # So we only need:
    # 1. sum_j x_ij = 1 (N equality)
    # 2. sum_i d_i x_ij - cap_j y_j <= 0 (M inequality)
    # 3. 0 <= x_ij, 0 <= y_j <= 1
    
    # This reduces A_ub size to M (20) rows.
    # A_eq size to N (100) rows.
    # This is MUCH faster.
    
    # Construct A_eq (100 x 2020)
    A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        # x_ij are in indices i*n_fac to (i+1)*n_fac - 1
        A_eq[i, i*n_fac : (i+1)*n_fac] = 1.0
        
    # Construct A_ub (20 x 2020)
    A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
    b_ub = np.zeros(n_fac)
    for j in range(n_fac):
        # Capacity constraint for facility j:
        # sum_i d_i * x_ij <= cap_j * y_j
        # => sum_i d_i * x_ij - cap_j * y_j <= 0
        for i in range(n_cust):
            idx_x = i * n_fac + j
            A_ub[j, idx_x] = demands[i]
        idx_y = n_cust * n_fac + j
        A_ub[j, idx_y] = -capacities[j]
        
    # Bounds
    # x_ij: [0, 1]
    # y_j: [0, 1]
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
    
    # Solve LP
    try:
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if not res_lp.success:
            raise Exception("LP failed")
        
        x_lp = res_lp.x[:n_cust * n_fac].reshape(n_cust, n_fac)
        y_lp = res_lp.x[n_cust * n_fac : n_cust * n_fac + n_fac]
        
    except Exception as e:
        # Fallback to greedy if LP fails or is too slow
        # But let's assume it works. If it takes too long, we need a timeout.
        # Linprog with HiGHS should be fast for 2000 vars.
        pass

    # If LP didn't run (e.g. exception), fallback to a simple greedy
    if 'res_lp' not in locals() or not res_lp.success:
        # Greedy fallback
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                f = np.argmax(capacities - load)
            else:
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            result[i] = f
            load[f] += demands[i]
        return result

    # Rounding Strategy:
    # 1. Identify "core" facilities: those with y_lp > 0.5 or high assignment probability.
    # 2. We need to select a subset of facilities to open.
    # 3. Let's try a few rounding schemes.
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Scheme 1: Greedy Rounding based on y_lp
    # Sort facilities by y_lp descending.
    # Try opening top K facilities for K in [1..n_fac] or based on y_lp threshold.
    
    # To speed up, we can just take the top K facilities where y_lp is significant.
    # Let's try K from 1 to n_fac, but prioritize those with high y_lp.
    
    # Actually, a better rounding:
    # For each customer, assign to facility with highest x_lp[i,j] among open facilities.
    # But which facilities are open?
    # We can estimate the number of open facilities from the LP objective or y_lp sum.
    
    # Let's iterate over potential sets of open facilities.
    # Since n_fac=20, we can't try all subsets.
    # But we can try "Threshold" based sets.
    
    # Sort facilities by y_lp descending.
    fac_order = np.argsort(-y_lp)
    
    # Try opening the first K facilities in this order, for K in a range.
    # Also try "Top K by x_lp sum".
    
    x_sum_per_fac = np.sum(x_lp, axis=0)
    fac_order_x = np.argsort(-x_sum_per_fac)
    
    candidates = []
    for K in range(1, n_fac + 1):
        candidates.append(set(fac_order[:K]))
        candidates.append(set(fac_order_x[:K]))
        
    # Also add the set of facilities where y_lp > 0.1, etc.
    for thresh in [0.1, 0.2, 0.5, 0.8]:
        candidates.append(set(np.where(y_lp > thresh)[0]))
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # For each candidate set of open facilities, solve the assignment subproblem.
    # Since the set is fixed, we just need to assign customers to minimize cost subject to capacity.
    # This is a Transportation Problem.
    # We can solve it with Linprog again, or a greedy heuristic.
    # Given 100 customers, greedy is fast.
    
    for open_set in unique_candidates:
        if time.time() > deadline - 5.0:
            break
            
        if len(open_set) == 0:
            continue
            
        open_list = np.array(list(open_set))
        
        # Greedy assignment to open facilities
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        # Sort customers by demand descending to fill large capacities first?
        # Or sort by "regret"?
        # Let's sort by demand descending.
        order = np.argsort(-demands)
        
        feasible_all = True
        for i in order:
            # Find feasible open facilities
            feasible = open_list[load[open_list] + demands[i] <= capacities[open_list]]
            if len(feasible) == 0:
                feasible_all = False
                break
            # Choose cheapest
            costs = assign_cost[i, feasible]
            best_f = feasible[np.argmin(costs)]
            res[i] = best_f
            load[best_f] += demands[i]
            
        if not feasible_all:
            continue
            
        # Calculate score
        used_fac = np.unique(res)
        # Ensure only open_set facilities are used (should be true by construction)
        # But what if a facility in open_set has 0 load? It's still "open" in our candidate set, 
        # but does it incur cost? 
        # The problem says "open_cost over facilities with at least one customer".
        # So if we open a facility but assign no one, we don't pay.
        # Our greedy only assigns to open_set. If a facility in open_set gets 0 load, it's not in used_fac.
        
        score = np.sum(open_cost[used_fac]) + np.sum(assign_cost[np.arange(n_cust), res])
        
        if score < best_score:
            best_score = score
            best_result = res.copy()
            
    # If no feasible solution found (shouldn't happen), fallback to greedy
    if best_score == np.inf:
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                f = np.argmax(capacities - load)
            else:
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            result[i] = f
            load[f] += demands[i]
        return result

    # Local Search Polish on the best rounded solution
    # Use a simple local search to improve the rounded solution.
    current_res = best_result.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_res[i]] += demands[i]
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n_cust):
            f_old = current_res[i]
            d_i = demands[i]
            # Try moving to any feasible facility
            feasible = (current_load + d_i <= capacities) & (np.arange(n_fac) != f_old)
            if not np.any(feasible):
                continue
            fac_indices = np.where(feasible)[0]
            costs = assign_cost[i, fac_indices]
            best_f = fac_indices[np.argmin(costs)]
            
            # Calculate delta
            delta_assign = assign_cost[i, best_f] - assign_cost[i, f_old]
            
            # Open cost delta
            delta_open = 0.0
            # If f_old becomes empty
            if current_load[f_old] == d_i:
                delta_open -= open_cost[f_old]
            # If best_f was empty
            if current_load[best_f] == 0:
                delta_open += open_cost[best_f]
                
            total_delta = delta_assign + delta_open
            if total_delta < -1e-9:
                current_res[i] = best_f
                current_load[f_old] -= d_i
                current_load[best_f] += d_i
                improved = True
                best_score = np.sum(open_cost[np.unique(current_res)]) + np.sum(assign_cost[np.arange(n_cust), current_res])
                best_result = current_res.copy()
                
    return best_result
# EVOLVE-BLOCK-END
