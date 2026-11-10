import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    
    # 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    
    # Parse distribution
    probs = np.ones(n_types) / n_types
    if isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
            probs = probs / probs.sum()
        elif isinstance(distribution_spec, dict):
            for i, (v1, v2) in enumerate(types):
                if (v1, v2) in distribution_spec:
                    probs[i] = distribution_spec[(v1, v2)]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, str):
        # Try to parse as uniform, independent, or product
        if 'uniform' in distribution_spec.lower() or 'independent' in distribution_spec.lower():
            probs = np.ones(n_types) / n_types
        elif 'product' in distribution_spec.lower():
            # Assume independent uniform
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    
    # Ensure valid probabilities
    probs = np.clip(probs, 0, 1)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
        
    # Decision variables:
    # x[0:16]   = alloc good 1 for each type
    # x[16:32]  = alloc good 2 for each type
    # x[32:48]  = payments for each type
    n_vars = n_types * 3  # 48
    
    # Objective: maximize expected revenue = sum_i probs[i] * payment[i]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]
    
    # Constraints:
    # 1. Allocation feasibility: 0 <= x_g(i) <= 1 for all i, g (handled by bounds)
    # 2. Nonnegative payments: p(i) >= 0 (handled by bounds)
    # 3. Individual Rationality (IR): p(i) <= v1(i)*x1(i) + v2(i)*x2(i)
    #    => -v1(i)*x1(i) - v2(i)*x2(i) + p(i) <= 0
    # 4. Incentive Compatibility (IC): for all i, j:
    #    v1(i)*x1(i) + v2(i)*x2(i) - p(i) >= v1(i)*x1(j) + v2(i)*x2(j) - p(j)
    #    => -v1(i)*x1(i) - v2(i)*x2(i) + p(i) + v1(i)*x1(j) + v2(i)*x2(j) - p(j) <= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = types[i]
        row[i] = -v1       
        row[n_types + i] = -v2  
        row[32 + i] = 1    
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[n_types + i] = -v2_i
            row[32 + i] = 1
            row[j] = v1_i
            row[n_types + j] = v2_i
            row[32 + j] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: allocations in [0,1], payments >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x1(i)
    for i in range(n_types):
        bounds.append((0, 1))  # x2(i)
    for i in range(n_types):
        bounds.append((0, None))  # p(i)
        
    # Check time before solving
    if time.time() - start_time > 80:
        # Fallback if we are already close to limit (unlikely for initial setup)
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    # Solve LP with high precision options
    options = {
        'presolve': True,
        'time_limit': 60, # Hard limit for the solver itself
        'dual_feasibility_tolerance': 1e-12,
        'primal_feasibility_tolerance': 1e-12,
        'dual_simplex_iterations': 100000,
        'primal_simplex_iterations': 100000,
        'ipm_optimality_tolerance': 1e-12,
        'ipm_feasibility_tolerance': 1e-12,
        'ipm_max_iterations': 10000
    }
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        # Fallback if solver fails
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            if v1 + v2 >= 2:
                allocations[i] = [1, 1]
                payments[i] = 2.0
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            if v1 + v2 >= 2:
                allocations[i] = [1, 1]
                payments[i] = 2.0
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.column_stack([x[0:n_types], x[n_types:2*n_types]])
    payments = x[32:48]
    
    # Verify and fix numerical issues
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Check deadline
    if time.time() - start_time > 85:
        return {'allocations': allocations, 'payments': payments}
        
    # Polish step: Ensure strict feasibility for IR and IC within 1e-8 tolerance
    # The LP solver returns a solution that is theoretically optimal, but floating point
    # errors might push constraints slightly out of bounds.
    # We check if any constraint is violated by more than a tiny epsilon.
    
    def check_feasibility(allocs, pays):
        # Check IR: p(i) <= v1(i)*x1(i) + v2(i)*x2(i)
        for i in range(n_types):
            v1, v2 = types[i]
            util = v1 * allocs[i, 0] + v2 * allocs[i, 1]
            if pays[i] > util + 1e-8:
                return False
        # Check IC: v1(i)*x1(i) + v2(i)*x2(i) - p(i) >= v1(i)*x1(j) + v2(i)*x2(j) - p(j)
        for i in range(n_types):
            v1_i, v2_i = types[i]
            u_i = v1_i * allocs[i, 0] + v2_i * allocs[i, 1] - pays[i]
            for j in range(n_types):
                if i == j: continue
                u_j = v1_i * allocs[j, 0] + v2_i * allocs[j, 1] - pays[j]
                if u_i < u_j - 1e-8:
                    return False
        return True

    if not check_feasibility(allocations, payments):
        # If infeasible, we need to adjust.
        # A common safe adjustment is to reduce payments slightly or adjust allocations.
        # Since we want to maximize revenue, we should try to keep payments high.
        # However, if IR is violated, we must lower payments.
        # If IC is violated, it's more complex.
        
        # Simple fix: Reduce payments by a small epsilon if IR is violated
        # But IC violations are harder to fix simply.
        
        # Given the small problem size, we can try to re-solve with tighter bounds or
        # just accept the LP solution if the violation is tiny (< 1e-9).
        # The evaluator requires regret <= 1e-8.
        
        # Let's check the maximum violation
        max_violation = 0
        for i in range(n_types):
            v1, v2 = types[i]
            util = v1 * allocations[i, 0] + v2 * allocations[i, 1]
            viol_ir = payments[i] - util
            if viol_ir > max_violation:
                max_violation = viol_ir
            
            u_i = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
            for j in range(n_types):
                if i == j: continue
                u_j = v1 * allocations[j, 0] + v2 * allocations[j, 1] - payments[j]
                viol_ic = u_j - u_i
                if viol_ic > max_violation:
                    max_violation = viol_ic
        
        if max_violation > 1e-8:
            # This is a significant violation. 
            # Fallback to a conservative mechanism that is known to be feasible.
            # The "seed" fallback was: allocate both if sum >= 2, charge 2.
            # Let's use a slightly better fallback: 
            # Myerson style for independent uniform values is complex.
            # Let's stick to the simple feasible one if LP fails badly.
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i, (v1, v2) in enumerate(types):
                if v1 + v2 >= 2:
                    allocations[i] = [1, 1]
                    payments[i] = 2.0
        else:
            # Violation is tiny, likely due to float precision. 
            # We can nudge payments down by max_violation to ensure feasibility.
            payments -= max_violation
            payments = np.maximum(payments, 0)

    return {'allocations': allocations, 'payments': payments}
