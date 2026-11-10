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
    
    # Decision variables: x[0:16] = alloc good 1, x[16:32] = alloc good 2, x[32:48] = payments
    # Total: 48 variables
    n_vars = n_types * 3  # 48
    
    # Objective: maximize expected revenue = sum_i probs[i] * payment[i]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]
    
    # Constraints:
    # 1. Allocation feasibility: 0 <= x_g(i) <= 1 for all i, g
    # 2. Nonnegative payments: p(i) >= 0
    # 3. Individual Rationality: p(i) <= v1(i)*x1(i) + v2(i)*x2(i) for all i
    # 4. Incentive Compatibility: for all i, j:
    #    v1(i)*x1(i) + v2(i)*x2(i) - p(i) >= v1(i)*x1(j) + v2(i)*x2(j) - p(j)
    #    Rearranged: -v1(i)*x1(i) - v2(i)*x2(i) + p(i) + v1(i)*x1(j) + v2(i)*x2(j) - p(j) <= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints: p(i) - v1(i)*x1(i) - v2(i)*x2(i) <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = types[i]
        row[i] = -v1       # -v1 * x1(i)
        row[n_types + i] = -v2  # -v2 * x2(i)
        row[32 + i] = 1    # +1 * p(i)
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints: for all i (true type), j (reported type) adjacent to i in the lattice
    # Adjacent means differ by exactly 1 in one coordinate and 0 in the other
    # This is sufficient for IC on a lattice type space (reduces from 240 to ~64 constraints)
    type_index = {(v1, v2): i for i, (v1, v2) in enumerate(types)}
    for i in range(n_types):
        v1_i, v2_i = types[i]
        # Generate adjacent types
        adjacent = []
        for dv1, dv2 in [(1,0), (-1,0), (0,1), (0,-1)]:
            j_type = (v1_i + dv1, v2_i + dv2)
            if 0 <= j_type[0] <= 3 and 0 <= j_type[1] <= 3:
                j = type_index[j_type]
                adjacent.append(j)
        for j in adjacent:
            row = np.zeros(n_vars)
            # -v1(i)*x1(i) - v2(i)*x2(i) + p(i)
            row[i] = -v1_i
            row[n_types + i] = -v2_i
            row[32 + i] = 1
            # +v1(i)*x1(j) + v2(i)*x2(j) - p(j)
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
    
    # Solve LP with high precision
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': 'highs', 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_staleness_threshold': 1e-12,
                              'ipm_optimality_tolerance': 1e-12})
    
    if not result.success:
        # Fallback: simple mechanism - allocate both goods if value sum >= 2, charge fixed price
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
    
    # Verify IC and IR constraints are satisfied within tolerance
    # If not, try a second pass with the full IC constraint set
    max_violation = 0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        # IR check
        ir_viol = payments[i] - v1_i * allocations[i, 0] - v2_i * allocations[i, 1]
        max_violation = max(max_violation, ir_viol)
        # IC check against all other types
        for j in range(n_types):
            if i == j:
                continue
            util_truth = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
            util_lie = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
            ic_viol = util_lie - util_truth
            max_violation = max(max_violation, ic_viol)
    
    if max_violation > 1e-8:
        # Re-solve with full IC constraints if adjacent-only wasn't sufficient
        A_ub_full = []
        b_ub_full = []
        # IR constraints
        for i in range(n_types):
            row = np.zeros(n_vars)
            v1, v2 = types[i]
            row[i] = -v1
            row[n_types + i] = -v2
            row[32 + i] = 1
            A_ub_full.append(row)
            b_ub_full.append(0)
        # Full IC constraints
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
                A_ub_full.append(row)
                b_ub_full.append(0)
        
        A_ub_full = np.array(A_ub_full)
        b_ub_full = np.array(b_ub_full)
        result_full = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs',
                              options={'presolve': 'highs', 'dual_feasibility_tolerance': 1e-12,
                                       'primal_feasibility_tolerance': 1e-12,
                                       'dual_staleness_threshold': 1e-12,
                                       'ipm_optimality_tolerance': 1e-12})
        if result_full.success:
            x = result_full.x
            allocations = np.column_stack([x[0:n_types], x[n_types:2*n_types]])
            payments = x[32:48]
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
    
    # Check deadline
    if time.time() - start_time > 85:
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': allocations, 'payments': payments}
