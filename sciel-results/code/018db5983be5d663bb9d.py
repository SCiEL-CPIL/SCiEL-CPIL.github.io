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
    
    # IC constraints: for all i (true type), j (reported type), i != j
    # Utility from truth: v1(i)*x1(i) + v2(i)*x2(i) - p(i)
    # Utility from lying as j: v1(i)*x1(j) + v2(i)*x2(j) - p(j)
    # IC: truth >= lie
    # v1(i)*x1(i) + v2(i)*x2(i) - p(i) - v1(i)*x1(j) - v2(i)*x2(j) + p(j) >= 0
    # Rearranged for linprog (<= 0):
    # -v1(i)*x1(i) - v2(i)*x2(i) + p(i) + v1(i)*x1(j) + v2(i)*x2(j) - p(j) <= 0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
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
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
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
    
    # Check deadline
    if time.time() - start_time > 85:
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': allocations, 'payments': payments}
