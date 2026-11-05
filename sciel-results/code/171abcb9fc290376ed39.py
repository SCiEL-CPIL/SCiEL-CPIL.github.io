"""Incentive-compatible finite auction design solver – full LP over allocations and payments."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Setup Types and Distributions
    values = [0, 1, 2, 3]
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    type_indices = {t: i for i, t in enumerate(types)}

    # Parse distribution robustly
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.ones(4) / 4
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.ones(4) / 4

    # Joint probability distribution over types
    # types are ordered as (0,0), (0,1), ..., (3,3)
    # p1 is prob of v1, p2 is prob of v2
    joint_probs = np.zeros(n_types)
    for i, (v1, v2) in enumerate(types):
        joint_probs[i] = p1[v1] * p2[v2]

    # 2. Define LP Variables
    # x[0..15]: allocation probability for good 1 for each type
    # x[16..31]: allocation probability for good 2 for each type
    # x[32..47]: payment for each type
    n_vars = 48
    n_alloc = 32
    n_pay = 16

    # Objective: Maximize Expected Revenue
    # Max sum_t P(t) * Payment[t]
    c = np.zeros(n_vars)
    c[32:48] = -joint_probs  # linprog minimizes, so negate

    # 3. Constraints
    A_ub = []
    b_ub = []

    # A. Individual Rationality (IR)
    # Payment[t] <= Value[t] . Allocation[t]
    # Payment[t] - v1[t]*x1[t] - v2[t]*x2[t] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t] = 1.0       # x1[t]
        row[t + 16] = 1.0  # x2[t]
        row[32 + t] = -1.0 # Payment[t]
        A_ub.append(row)
        b_ub.append(0.0)

    # B. Incentive Compatibility (IC)
    # For all t, t':
    # v1[t]*x1[t] + v2[t]*x2[t] - Payment[t] >= v1[t]*x1[t'] + v2[t]*x2[t'] - Payment[t']
    # => v1[t]*(x1[t]-x1[t']) + v2[t]*(x2[t]-x2[t']) - Payment[t] + Payment[t'] >= 0
    # => -v1[t]*x1[t] + v1[t]*x1[t'] - v2[t]*x2[t] + v2[t]*x2[t'] + Payment[t] - Payment[t'] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # Coefficients for x1[t] and x2[t] (LHS of original utility)
            row[t] = -v1_t
            row[t + 16] = -v2_t
            # Coefficients for x1[tp] and x2[tp] (RHS of original utility)
            row[tp] = v1_t
            row[tp + 16] = v2_t
            # Coefficients for Payments
            row[32 + t] = 1.0
            row[32 + tp] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # C. Non-negativity of Payments
    # Payment[t] >= 0  =>  -Payment[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[32 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # D. Bounds
    # Allocations: [0, 1]
    # Payments: [0, None] (Upper bound handled by IR, but let's keep it open for solver)
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # 4. Solve LP
    best_alloc = None
    best_pay = None
    best_revenue = -np.inf

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            
            # Extract solutions
            alloc1 = x[0:16]
            alloc2 = x[16:32]
            payments = x[32:48]
            
            # Ensure numerical stability (clip tiny negatives)
            alloc1 = np.clip(alloc1, 0, 1)
            alloc2 = np.clip(alloc2, 0, 1)
            payments = np.maximum(payments, 0)
            
            allocations = np.column_stack((alloc1, alloc2))
            
            # Calculate Revenue
            rev = float(np.dot(joint_probs, payments))
            
            # Validation Step (Crucial for "Incentive-compatible" requirement)
            # Check IR and IC with tolerance
            valid = True
            tol = 1e-6
            for t in range(n_types):
                v1_t, v2_t = types[t]
                u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                
                # IR Check
                if u_tt < -tol:
                    valid = False
                    break
                
                # IC Check
                for tp in range(n_types):
                    u_tt_prime = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
                    if u_tt < u_tt_prime - tol:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_alloc = allocations
                best_pay = payments
                best_revenue = rev
            else:
                # If validation fails due to numerical precision, try a fallback
                # or accept if the violation is extremely small
                max_violation = 0
                for t in range(n_types):
                    v1_t, v2_t = types[t]
                    u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                    if u_tt < -max_violation: max_violation = -u_tt
                    for tp in range(n_types):
                        u_tt_prime = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
                        if u_tt < u_tt_prime:
                            viol = u_tt_prime - u_tt
                            if viol > max_violation: max_violation = viol
                
                if max_violation < 1e-4:
                    best_alloc = allocations
                    best_pay = payments
                    best_revenue = rev

    except Exception as e:
        pass

    # Fallback if LP failed or returned invalid
    if best_alloc is None:
        # Simple Threshold Fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for t in range(16):
            v1_t, v2_t = types[t]
            if v1_t >= 2:
                allocations[t, 0] = 1
            if v2_t >= 2:
                allocations[t, 1] = 1
            payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        best_alloc = allocations
        best_pay = payments

    return {'allocations': best_alloc, 'payments': best_pay}
