"""Capacitated facility location: Branch-and-bound over facility subsets with min-cost flow."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    deadline = time.time() + 142.0
    total_demand = float(np.sum(demands))

    # 1. Greedy Fallback (Always valid)
    def get_greedy():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                # Should not happen if total capacity >= total demand, but safe fallback
                f = int(np.argmax(capacities - load))
                res[i] = f
                load[f] += d_i
            else:
                costs = assign_cost[i, feasible]
                f_idx = int(feasible[np.argmin(costs)])
                res[i] = f_idx
                load[f_idx] += d_i
        return res

    best_res = get_greedy()
    best_score = float(np.sum(open_cost[np.unique(best_res)]) + np.sum(assign_cost[np.arange(n_cust), best_res]))

    # 2. Exact Solver for a fixed set of facilities
    # Returns (total_cost, assignment_array) or (inf, None) if infeasible
    def solve_subset(fac_indices):
        m = len(fac_indices)
        if m == 0:
            return np.inf, None
        
        # Build cost matrix for these facilities
        # Shape: (n_cust, m)
        sub_costs = assign_cost[:, fac_indices]
        
        # We need to handle capacities.
        # Min-cost flow formulation:
        # Source -> Customers (cap=demand, cost=0)
        # Customers -> Facilities (cap=inf, cost=cost)
        # Facilities -> Sink (cap=capacity, cost=0)
        
        # Since scipy doesn't have a direct min-cost flow solver that returns assignment easily 
        # without a graph library (like networkx which might not be available or slow),
        # we can use the Hungarian algorithm if capacities were infinite.
        # With capacities, we can "split" facilities into multiple slots.
        # However, splitting is inefficient if capacities are large.
        
        # Alternative: Use scipy.optimize.linear_sum_assignment on an expanded matrix?
        # No, linear_sum_assignment requires square matrix (or rectangular) and assigns 1-to-1.
        # Here, one facility can serve multiple customers.
        
        # We must implement a simple min-cost flow or use a trick.
        # Trick: If we duplicate each facility index 'k' times where k = capacity // min_demand? 
        # No, demands are variable.
        
        # Let's implement a simple successive shortest path algorithm for min-cost flow.
        # Or, given the small size (100 customers, 20 facilities), we can use a linear programming approach?
        # scipy.optimize.linprog is available.
        
        # Let's try linprog.
        # Variables x_ij for i in 0..99, j in 0..m-1
        # Minimize sum c_ij x_ij
        # s.t. sum_j x_ij = 1 for all i
        #      sum_i d_i x_ij <= cap_j for all j
        #      x_ij >= 0
        # 
        # This is a fractional assignment. The constraint matrix is not necessarily totally unimodular 
        # because of the capacity constraints on the right side (sum d_i x_ij <= cap_j).
        # Wait, the assignment constraint is sum x_ij = 1.
        # If we relax x_ij to be continuous, is the optimal solution integer?
        # The problem is: Min c^T x s.t. A x = b, G x <= h, x >= 0.
        # A is the identity-like matrix for customers (rows sum to 1).
        # G is the capacity matrix.
        # This is a transportation problem variant.
        # Actually, if we treat each customer as a supply of 1 unit, and facilities as demands?
        # No, customer demand is d_i.
        # Let's scale: Customer i supplies d_i units. Facility j demands cap_j units?
        # No, facility j has capacity cap_j.
        # Total supply = sum d_i. Total capacity = sum cap_j.
        # We need to send flow from customers to facilities.
        # Flow from i to j is f_ij.
        # Sum_j f_ij = d_i (for all i)
        # Sum_i f_ij <= cap_j (for all j)
        # Cost is (assign_cost[i,j] / d_i) * f_ij ? 
        # If we send f_ij units from i to j, the cost is f_ij * (assign_cost[i,j] / d_i).
        # This is a standard min-cost flow problem.
        # Min-cost flow has integer optimal solutions if supplies/demands/capacities are integers.
        # Here d_i and cap_j are integers.
        # So we can solve this as a min-cost flow.
        
        # scipy.optimize.linprog might be slow for 100*20 = 2000 variables? 
        # Actually, 2000 variables is small for linprog.
        # But we need to call it many times.
        
        # Is there a faster way?
        # NetworkX has min_cost_flow. Is NetworkX available? "Python standard library, NumPy and SciPy are available."
        # NetworkX is NOT standard library. So we cannot use it.
        
        # We must implement Min-Cost Flow or use LinProg.
        # Let's try to implement a fast Min-Cost Flow using Successive Shortest Paths with Potentials.
        # Nodes: Source(0), Customers(1..100), Facilities(101..100+m), Sink(100+m+1)
        # Edges:
        # Source -> i: cap=d_i, cost=0
        # i -> j: cap=inf, cost=assign_cost[i,j]/d_i  (Wait, if we send 1 unit of "customer i", it costs assign_cost[i,j]. 
        #        If we send d_i units, total cost is d_i * (assign_cost[i,j]/d_i) = assign_cost[i,j]. Correct.)
        # j -> Sink: cap=cap_j, cost=0
        
        # Graph size: ~122 nodes. Edges: 100 + 100*m + m.
        # For m=20, edges ~ 100 + 2000 + 20 = 2120.
        # Successive Shortest Path: We augment 1 unit at a time? No, we augment as much as possible along the shortest path.
        # Since d_i are integers (5-35), the number of augmentations is bounded by total demand / min_step?
        # Actually, the number of augmentations in SSP is O(F * E log V) where F is total flow?
        # Total flow is ~2000. E is ~2000. 2000 * 2000 * log(120) ~ 4M * 7 ~ 28M operations.
        # This might be too slow in Python for many B&B nodes.
        
        # Alternative: Use scipy.optimize.linprog with method='highs'.
        # HiGHS is very fast.
        # Let's try LinProg first. If it's too slow, we'll switch.
        
        from scipy.optimize import linprog
        
        # Variables: x_ij for i=0..n-1, j=0..m-1
        # Total variables: n * m
        # Objective: min sum c_ij x_ij
        # Constraints:
        # 1. sum_j x_ij = 1 for each i  (Equality)
        # 2. sum_i d_i x_ij <= cap_j for each j (Inequality)
        
        # c vector: flatten(sub_costs)
        c = sub_costs.flatten()
        
        # A_eq: n rows, n*m cols
        A_eq = np.zeros((n_cust, n_cust * m))
        b_eq = np.ones(n_cust)
        for i in range(n_cust):
            for j in range(m):
                A_eq[i, i * m + j] = 1.0
        
        # A_ub: m rows, n*m cols
        A_ub = np.zeros((m, n_cust * m))
        b_ub = np.array(capacities[fac_indices])
        for j in range(m):
            for i in range(n_cust):
                A_ub[j, i * m + j] = demands[i]
        
        # Bounds: 0 <= x_ij <= 1
        bounds = [(0, 1)] * (n_cust * m)
        
        res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res_lp.success:
            return np.inf, None
            
        x = res_lp.x.reshape(n_cust, m)
        
        # The solution should be integer (or close to it) due to total unimodularity of the assignment part?
        # Actually, the capacity constraints break total unimodularity of the combined matrix?
        # No, the constraint matrix of the transportation problem is TU.
        # Here, the constraints are:
        # Sum_j x_ij = 1
        # Sum_i d_i x_ij <= cap_j
        # This is NOT a standard transportation problem matrix because of the d_i coefficients.
        # However, if we scale variables y_ij = d_i x_ij, then:
        # Sum_j y_ij = d_i
        # Sum_i y_ij <= cap_j
        # Min sum (c_ij/d_i) y_ij
        # This IS a standard transportation problem (with supplies d_i and capacities cap_j).
        # The constraint matrix for transportation is TU.
        # So the optimal solution for y is integer.
        # Thus x_ij = y_ij / d_i might be fractional.
        # BUT, we need x_ij to be 0 or 1 (assignment).
        # Wait, the original problem is "Assign each customer to ONE facility".
        # So x_ij must be 0 or 1.
        # The relaxation allows splitting a customer's demand across facilities.
        # Is the optimal solution of the relaxation always integral in x?
        # No. Consider 2 customers, 2 facilities.
        # d1=1, d2=1. cap1=1, cap2=1.
        # c11=1, c12=2, c21=2, c22=1.
        # Optimal: x11=1, x22=1. Cost 2.
        # Relaxation: same.
        # Consider: d1=2, d2=2. cap1=2, cap2=2.
        # c11=1, c12=10, c21=10, c22=1.
        # Optimal: x11=1, x22=1. Cost 2.
        # What if d1=1, d2=1. cap1=1, cap2=1.
        # c11=1, c12=1, c21=100, c22=100.
        # Optimal: x11=1, x22=1? No, x12=1, x21=1 is better? Cost 101.
        # x11=1, x22=1 cost 2.
        
        # The key question: Does the LP relaxation of the CFFL (Capacitated Facility Location) have integer optimal solutions?
        # Generally, NO. The LP relaxation can have fractional solutions.
        # However, for small instances, it might be close.
        # But we need a VALID integer assignment.
        # If we use the LP solution, we might get x11=0.5, x12=0.5.
        # We must round.
        # Rounding might violate capacity or optimality.
        
        # Given the time limit, maybe we should stick to the heuristic?
        # Or use the LP solution to guide a heuristic?
        
        # Let's try a different approach:
        # Since n=100, m<=20.
        # We can use a simple greedy assignment on the subset, but optimized.
        # Or, we can use the LP solution to identify the "most likely" facilities and then do local search.
        
        # Actually, let's look at the "Branch and Bound" aspect.
        # If we can't solve the subproblem exactly, B&B is hard.
        
        # Let's reconsider the Min-Cost Flow implementation.
        # If we implement SSP correctly, it will find the integer optimal solution for the flow problem.
        # The flow problem variables y_ij are integers.
        # But y_ij represents the amount of demand from i sent to j.
        # We need x_ij in {0,1}.
        # The flow solution allows y_ij to be any integer between 0 and d_i.
        # This means a customer's demand can be split.
        # The task requires "assigns each of 100 customers to ONE facility".
        # So we CANNOT split.
        # Therefore, Min-Cost Flow does NOT solve the problem directly.
        
        # We must solve the Integer Program.
        # With 100 customers and 20 facilities, this is NP-Hard.
        # So we MUST use heuristics or bounds.
        
        # Strategy Pivot:
        # Since exact solving is hard, we will use the Branch and Bound structure to explore subsets,
        # but for each subset, we will use a FAST HEURISTIC to find a good assignment.
        # The heuristic:
        # 1. Sort customers by demand descending.
        # 2. Assign each to the cheapest facility in the subset that has remaining capacity.
        # 3. Improve with local search (single moves) restricted to the subset.
        
        # This is essentially what the current program does, but the B&B structure ensures we explore
        # subsets systematically and can prune based on lower bounds.
        
        # Lower Bound for a subset S:
        # LB = sum(open_cost[S]) + sum(min_j in S assign_cost[i,j] for i in 1..n)
        # This ignores capacity. It's a valid LB.
        
        # If LB >= best_score, prune.
        
        # This is a different strategy: "Systematic Subset Exploration with Capacity-Aware Heuristic".
        
        # Let's implement this.
        
        # We need to enumerate subsets.
        # 2^20 is too many.
        # We can use a priority queue (Best-First Search) over subsets.
        # Start with all facilities open.
        # Pop the subset with the smallest LB.
        # Solve the heuristic for this subset.
        # If it improves best_score, update.
        # Generate children by removing one facility from the current subset.
        # Push children with their LBs.
        
        # This is a standard B&B for facility location.
        
        # Let's implement this.
        
        import heapq
        
        # Precompute min_assign_cost for each customer for any subset?
        # No, LB depends on the subset.
        # LB(S) = sum(open_cost[S]) + sum_i min_{j in S} assign_cost[i,j]
        
        # We can compute min_{j in S} assign_cost[i,j] quickly if we precompute sorted costs?
        # Or just compute it on the fly. n=100, m<=20. 100*20 = 2000 ops per LB. Fast.
        
        def calc_lb(fac_indices):
            if len(fac_indices) == 0:
                return np.inf
            open_sum = np.sum(open_cost[fac_indices])
            # min cost per customer
            sub_costs = assign_cost[:, fac_indices]
            min_costs = np.min(sub_costs, axis=1)
            assign_sum = np.sum(min_costs)
            return open_sum + assign_sum
        
        # Heuristic Solver for a subset
        def solve_heuristic(fac_indices):
            m = len(fac_indices)
            if m == 0:
                return None, np.inf
            
            # 1. Greedy Assign
            res = np.full(n_cust, -1, dtype=int)
            load = np.zeros(m)
            order = np.argsort(-demands)
            
            for i in order:
                d_i = demands[i]
                # Find feasible facilities in this subset
                feasible = []
                for k, f in enumerate(fac_indices):
                    if load[k] + d_i <= capacities[f]:
                        feasible.append(k)
                
                if len(feasible) == 0:
                    return None, np.inf # Infeasible subset (capacity too low)
                
                # Pick cheapest
                costs = assign_cost[i, fac_indices[feasible]]
                best_k = feasible[np.argmin(costs)]
                res[i] = fac_indices[best_k]
                load[best_k] += d_i
            
            # 2. Local Search (Single Moves) restricted to this subset
            # Map global facility index to local index
            # We will work with global indices for simplicity, but only consider moves to fac_indices
            
            current_load = np.zeros(n_fac)
            for i in range(n_cust):
                current_load[res[i]] += demands[i]
                
            improved = True
            max_iter = 50
            for _ in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(n_cust):
                    f_old = res[i]
                    d_i = demands[i]
                    # Find best new facility in fac_indices
                    best_f_new = -1
                    best_delta = 0
                    base_cost = assign_cost[i, f_old]
                    
                    # Calculate potential savings from closing f_old (if it becomes empty)
                    # This is complex in local search. Let's just look at assignment cost change 
                    # plus open cost change if facility becomes empty or new one becomes non-empty.
                    
                    # For simplicity, let's just optimize assignment cost first, then check open cost.
                    # Actually, the current program's single_moves_pass is good.
                    # Let's adapt it to be restricted to fac_indices.
                    
                    feasible_mask = np.zeros(n_fac, dtype=bool)
                    for f in fac_indices:
                        if current_load[f] + d_i <= capacities[f]:
                            feasible_mask[f] = True
                            
                    if not np.any(feasible_mask):
                        continue
                        
                    # delta_assign
                    delta_assign = assign_cost[i, :] - base_cost
                    
                    # delta_open
                    delta_open = np.zeros(n_fac)
                    if current_load[f_old] == d_i:
                        # f_old becomes empty
                        delta_open[f_old] = -open_cost[f_old]
                        
                    # New facilities opened
                    for f in fac_indices:
                        if f != f_old and current_load[f] == 0 and feasible_mask[f]:
                            delta_open[f] = open_cost[f]
                            
                    total_delta = delta_assign + delta_open
                    total_delta[~feasible_mask] = np.inf
                    total_delta[f_old] = np.inf # Don't move to self
                    
                    best_f_new = int(np.argmin(total_delta))
                    best_delta = total_delta[best_f_new]
                    
                    if best_delta < -1e-9:
                        res[i] = best_f_new
                        current_load[f_old] -= d_i
                        current_load[best_f_new] += d_i
                        improved = True
                        
                if not improved:
                    break
            
            # Calculate final score
            used = np.unique(res)
            score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
            
            # Verify validity
            if np.any(current_load > capacities):
                return None, np.inf
                
            return res, score

        # Best-First Search over Subsets
        # State: (lb, tuple_of_fac_indices)
        # We start with all facilities.
        # We want to find a good subset.
        
        # To avoid exploring too many subsets, we limit the number of nodes popped.
        max_nodes = 5000
        
        # Initialize heap
        heap = []
        initial_subset = tuple(range(n_fac))
        lb_init = calc_lb(initial_subset)
        heapq.heappush(heap, (lb_init, initial_subset))
        
        visited = set()
        
        while heap and time.time() < deadline - 10.0:
            lb, subset = heapq.heappop(heap)
            if subset in visited:
                continue
            visited.add(subset)
            
            if lb >= best_score:
                continue # Prune
            
            # Solve heuristic for this subset
            res, score = solve_heuristic(subset)
            
            if res is not None and score < best_score:
                best_score = score
                best_res = res.copy()
                
            # Generate children (remove one facility)
            # To keep diversity, we might want to remove facilities that are "least used" or "most expensive"?
            # For now, remove all.
            # But 2^20 is huge. We need to be smart.
            # Only generate children if the current subset is "promising".
            # Or limit the number of children.
            
            # Let's generate children by removing each facility in the subset.
            # To avoid explosion, we only do this if the subset size is > 1.
            if len(subset) > 1:
                # To reduce branching factor, we can randomly sample a few facilities to remove?
                # Or remove the one with highest open cost?
                # Let's remove the one with highest open cost first, then next, etc.
                # Sort facilities in subset by open cost descending.
                facs_sorted = sorted(subset, key=lambda x: -open_cost[x])
                
                # Only generate a few children to keep it tractable
                for f_remove in facs_sorted[:5]: # Top 5 most expensive to close
                    new_subset = tuple(f for f in subset if f != f_remove)
                    if new_subset not in visited:
                        lb_new = calc_lb(new_subset)
                        if lb_new < best_score:
                            heapq.heappush(heap, (lb_new, new_subset))
                            
            # Limit heap size to prevent memory issues
            if len(heap) > 10000:
                # Pop the worst ones? No, heap is min-heap.
                # We can just stop adding?
                pass

    # Final Polish: Run a global local search on the best result found
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[best_res[i]] += demands[i]
        
    # Re-use the logic from the original program for a final polish
    # But restricted to the time limit
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n_cust):
            f_old = best_res[i]
            d_i = demands[i]
            feasible_mask = (current_load + d_i <= capacities)
            feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
            if not np.any(feasible_mask):
                continue
            delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
            delta_open = np.zeros(n_fac)
            if current_load[f_old] == d_i:
                delta_open[f_old] = -open_cost[f_old]
            open_new = (current_load == 0) & (np.arange(n_fac) != f_old)
            if np.any(open_new):
                delta_open[open_new] = open_cost[open_new]
            total_delta = delta_assign + delta_open
            total_delta[~feasible_mask] = np.inf
            total_delta[f_old] = np.inf
            best_f_new = int(np.argmin(total_delta))
            best_delta = total_delta[best_f_new]
            if best_delta < -1e-9:
                best_res[i] = best_f_new
                current_load[f_old] -= d_i
                current_load[best_f_new] += d_i
                improved = True

    return best_res
# EVOLVE-BLOCK-END
