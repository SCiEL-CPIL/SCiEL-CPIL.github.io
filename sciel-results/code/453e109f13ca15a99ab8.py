"""Rewritten program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # The 16 reports: all combinations of values in {0,1,2,3} for two goods
    all_reports = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Parse support to determine which reports are used
    if support is not None and len(support) > 0:
        reports = []
        for s in support:
            if isinstance(s, (tuple, list)):
                reports.append(tuple(s))
            elif isinstance(s, (int, np.integer)):
                # Could be an index
                reports.append(all_reports[int(s) % 16])
            else:
                reports.append(all_reports[0])
        # Ensure we have exactly 16 reports
        if len(reports) > 16:
            reports = reports[:16]
        while len(reports) < 16:
            reports.append(all_reports[len(reports)])
    else:
        reports = all_reports
    
    # Parse distribution_spec to get probabilities for each report
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            probs = np.zeros(16)
            for i, r in enumerate(reports):
                if r in distribution_spec:
                    probs[i] = float(distribution_spec[r])
                else:
                    # Try tuple matching
                    for k, v in distribution_spec.items():
                        try:
                            if tuple(k) == r:
                                probs[i] = float(v)
                                break
                        except:
                            pass
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.array(distribution_spec, dtype=float).flatten()
            if len(arr) >= 16:
                probs = arr[:16].copy()
            else:
                probs = np.zeros(16)
                probs[:len(arr)] = arr
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    
    n = 16  # number of reports
    
    # LP formulation
    # Variables: for each report i, x1_i, x2_i, p_i
    # Total variables: 3 * 16 = 48
    # Index: i*3 + 0 = x1_i, i*3 + 1 = x2_i, i*3 + 2 = p_i
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(3 * n)
    for i in range(n):
        c[i * 3 + 2] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # IC constraints: for each true type i and report j:
    #   v1_i*x1_j + v2_i*x2_j - v1_i*x1_i - v2_i*x2_i - p_j + p_i <= 0
    for i in range(n):  # true type
        v1_i, v2_i = reports[i]
        for j in range(n):  # report
            row = np.zeros(3 * n)
            row[j * 3 + 0] = v1_i
            row[j * 3 + 1] = v2_i
            row[i * 3 + 0] = -v1_i
            row[i * 3 + 1] = -v2_i
            row[j * 3 + 2] = -1.0
            row[i * 3 + 2] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    # IR constraints: p_i - v1_i*x1_i - v2_i*x2_i <= 0
    for i in range(n):
        v1_i, v2_i = reports[i]
        row = np.zeros(3 * n)
        row[i * 3 + 0] = -v1_i
        row[i * 3 + 1] = -v2_i
        row[i * 3 + 2] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds: 0 <= x1_i, x2_i <= 1, 0 <= p_i
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple threshold mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(n):
            v1, v2 = reports[i]
            if v1 >= 2:
                allocations[i, 0] = 1.0
                payments[i] += 2.0
            if v2 >= 2:
                allocations[i, 1] = 1.0
                payments[i] += 2.0
        return {"allocations": allocations, "payments": payments}
    
    x = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((n, 2))
    payments = np.zeros(n)
    for i in range(n):
        allocations[i, 0] = x[i * 3 + 0]
        allocations[i, 1] = x[i * 3 + 1]
        payments[i] = x[i * 3 + 2]
    
    # Clip to valid range due to numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
