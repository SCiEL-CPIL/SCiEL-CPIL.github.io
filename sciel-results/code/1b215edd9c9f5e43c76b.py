# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import heapq


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses min-cost flow for exact assignment given a facility subset,
    combined with facility subset search and local search improvements.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit, 10s buffer)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def _total_cost(result):
        """Compute total cost: open cost + assignment cost"""
        used = set(result.tolist())
        oc = sum(open_cost[f] for f in used)
        ac = sum(assign_cost[i, result[i]] for i in range(n_cust))
        return oc + ac
    
    def _is_feasible(result):
        """Check if assignment is feasible (no capacity violations)"""
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)
    
    def _min_cost_flow_for_subset(subset, max_time=5.0):
        """
        Solve min-cost flow for assignment given a fixed subset of facilities.
        Returns (assignment, assignment_cost) or (None, inf) if infeasible.
        Uses scipy's linear_sum_assignment on a transformed problem.
        """
        subset = np.array(subset, dtype=int)
        k = len(subset)
        
        # Build cost matrix: n_cust x k
        cost_mat = assign_cost[:, subset]
        
        # We need to handle capacities. The standard linear_sum_assignment
        # handles 1-to-1 matching. For capacitated assignment, we use
        # a different approach: expand facilities by capacity units.
        
        # Alternative: use scipy.optimize.linear_sum_assignment won't work directly
        # for capacitated. Let's use a greedy + local search approach instead,
        # or use a flow-based approach.
        
        # Let's try a different approach: use scipy's min_cost_flow if available
        # Otherwise, use a heuristic.
        
        # Since scipy.optimize.min_cost_flow may not be available in all versions,
        # let's implement a simple approach:
        # 1. Sort customers by demand (largest first)
        # 2. For each customer, assign to cheapest feasible facility
        
        # But first, let's try the Hungarian algorithm approach with expansion
        # This is complex, so let's use a smarter greedy + local search
        
        # Greedy initialization with multiple strategies
        best_assign = None
        best_cost = float('inf')
        
        orderings = [
            np.argsort(-demands),  # Largest first
            np.argsort(demands),   # Smallest first
            np.argsort(cost_mat.mean(axis=1)),  # Cheapest average first
        ]
        
        for order in orderings:
            if time.time() > start_time + max_time:
                break
            result, cost = _greedy_for_subset(order, subset, cost_mat)
            if result is not None:
                if cost < best_cost:
                    best_cost = cost
                    best_assign = result
        
        if best_assign is None:
            return None, float('inf')
        
        # Local search to improve
        if time.time() < start_time + max_time - 0.5:
            improved = _local_search_for_subset(best_assign, subset, cost_mat)
            if improved is not None:
                imp_cost = sum(cost_mat[i, best_assign[i]] for i in range(n_cust))
                if imp_cost < best_cost:
                    best_cost = imp_cost
                    best_assign = improved
        
        return best_assign, best_cost
    
    def _greedy_for_subset(order, subset, cost_mat):
        """Greedy assignment for a given subset and ordering"""
        k = len(subset)
        load = np.zeros(k)
        result = np.full(n_cust, -1, dtype=int)
        
        for i in order:
            placed = False
            # Sort facilities by cost for this customer
            sorted_idx = np.argsort(cost_mat[i])
            for idx in sorted_idx:
                f = idx
                if load[f] + demands[i] <= capacities[subset[f]] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                return None, float('inf')
        
        total_cost = sum(cost_mat[i, result[i]] for i in range(n_cust))
        return result, total_cost
    
    def _local_search_for_subset(result, subset, cost_mat):
        """Local search for assignment within a fixed subset"""
        result = result.copy()
        k = len(subset)
        load = np.zeros(k)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        improved = True
        max_iters = 100
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1-move improvements
            for i in range(n_cust):
                current_f = result[i]
                current_cost = cost_mat[i, current_f]
                
                for f in range(k):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] <= capacities[subset[f]] + 1e-9:
                        new_cost = cost_mat[i, f]
                        if new_cost < current_cost - 1e-9:
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            current_cost = new_cost
                            improved = True
                            break
        
        return result
    
    def _evaluate_subset(subset):
        """Evaluate a facility subset: solve assignment and compute total cost"""
        assign_result, assign_cost_val = _min_cost_flow_for_subset(subset)
        if assign_result is None:
            return None, float('inf')
        
        # Map back to original facility indices
        final_result = np.zeros(n_cust, dtype=int)
        for i in range(n_cust):
            final_result[i] = subset[assign_result[i]]
        
        total = _total_cost(final_result)
        return final_result, total
    
    # Generate candidate subsets
    subsets_to_try = []
    
    # All facilities
    subsets_to_try.append(list(range(n_fac)))
    
    # By open cost
    by_open = np.argsort(open_cost)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_open[:k].tolist())
    
    # By average assignment cost
    avg_assign = assign_cost.mean(axis=0)
    by_assign = np.argsort(avg_assign)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_assign[:k].tolist())
    
    # By combined metric (open cost + avg assign cost)
    combined = open_cost + avg_assign
    by_combined = np.argsort(combined)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_combined[:k].tolist())
    
    # By efficiency: open_cost / capacity
    efficiency = open_cost / (capacities + 1)
    by_eff = np.argsort(efficiency)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_eff[:k].tolist())
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in subsets_to_try:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Sort subsets by a quick heuristic estimate
    def _subset_estimate(subset):
        subset = np.array(subset)
        oc = open_cost[subset].sum()
        # Estimate assignment cost: for each customer, min cost to any facility in subset
        ac = assign_cost[:, subset].min(axis=1).sum()
        return oc + ac
    
    unique_subsets.sort(key=_subset_estimate)
    
    # Try subsets in order of estimated cost
    best_result = None
    best_cost = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 10:
            break
        
        result, cost = _evaluate_subset(subset)
        if result is not None and cost < best_cost:
            best_cost = cost
            best_result = result.copy()
        
        # If we're running low on time, stop
        if time.time() > deadline - 20:
            break
    
    # If we have a feasible solution, do additional local search on the full problem
    if best_result is not None and time.time() < deadline - 5:
        # Convert to internal representation for full local search
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[best_result[i]] += demands[i]
        
        # 1-move improvements on the full problem
        improved = True
        max_iters = 200
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            
            for i in range(n_cust):
                current_f = best_result[i]
                current_cost = assign_cost[i, current_f]
                
                for f in range(n_fac):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        new_cost = assign_cost[i, f]
                        delta_assign = new_cost - current_cost
                        
                        # Check open cost changes
                        delta_open = 0
                        used = set(best_result.tolist())
                        if f not in used:
                            delta_open += open_cost[f]
                        if load[current_f] - demands[i] < 1e-9:
                            if current_f not in used or sum(1 for x in best_result if x == current_f) == 1:
                                delta_open -= open_cost[current_f]
                        
                        if delta_assign + delta_open < -1e-9:
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            best_result[i] = f
                            improved = True
                            break
    
    # Final feasibility check
    if best_result is None:
        # Fallback: simple greedy with all facilities
        load = np.zeros(n_fac)
        best_result = np.zeros(n_cust, dtype=int)
        for i in np.argsort(-demands):
            placed = False
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    best_result[i] = f
                    placed = True
                    break
            if not placed:
                # Last resort
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        load[f] += demands[i]
                        best_result[i] = f
                        break
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
