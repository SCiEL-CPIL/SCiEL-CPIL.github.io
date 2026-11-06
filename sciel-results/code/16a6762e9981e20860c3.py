# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    One additive bidder, two goods, values in {0,1,2,3} per good.
    Returns dict with 'allocations': (16,2) and 'payments': (16,).
    """
    start_time = time.time()
    deadline = start_time + 75
    
    reports = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_reports = 16
    
    probs = _extract_probs(distribution_spec, reports)
    
    if time.time() > deadline - 10:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    n_vars = 48
    c = np.zeros(n_vars)
    for r in range(n_reports):
        c[32 + r] = -probs[r]
    
    A_rows = []
    b_rows = []
    
    for r in range(n_reports):
        v1, v2 = reports[r]
        row = np.zeros(n_vars)
        row[r*2 + 0] = -v1
        row[r*2 + 1] = -v2
        row[32 + r] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    for r in range(n_reports):
        v1_r, v2_r = reports[r]
        for r2 in range(n_reports):
            if r == r2:
                continue
            row = np.zeros(n_vars)
            row[r2*2 + 0] = v1_r
            row[r2*2 + 1] = v2_r
            row[r*2 + 0] = -v1_r
            row[r*2 + 1] = -v2_r
            row[32 + r] = 1
            row[32 + r2] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    if time.time() > deadline - 15:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x = result.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for r in range(n_reports):
                allocations[r, 0] = x[r*2 + 0]
                allocations[r, 1] = x[r*2 + 1]
                payments[r] = x[32 + r]
            
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            max_viol = _check_feasibility(reports, allocations, payments)
            if max_viol <= 1e-6:
                return {'allocations': allocations, 'payments': payments}
    except Exception:
        pass
    
    return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

def _extract_probs(distribution_spec, reports):
    n = len(reports)
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.size == n:
            arr = arr.flatten()
            if arr.sum() > 0:
                return arr / arr.sum()
    elif isinstance(distribution_spec, dict):
        probs = np.zeros(n)
        for i, r in enumerate(reports):
            key = tuple(r)
            if key in distribution_spec:
                probs[i] = float(distribution_spec[key])
            elif r in distribution_spec:
                probs[i] = float(distribution_spec[r])
        if probs.sum() > 0:
            return probs / probs.sum()
    return np.ones(n) / n

def _check_feasibility(reports, allocations, payments):
    n = len(reports)
    max_viol = 0
    for r in range(n):
        v1, v2 = reports[r]
        util_self = v1 * allocations[r, 0] + v2 * allocations[r, 1] - payments[r]
        if util_self < 0:
            max_viol = max(max_viol, -util_self)
        for r2 in range(n):
            util_other = v1 * allocations[r2, 0] + v2 * allocations[r2, 1] - payments[r2]
            diff = util_other - util_self
            if diff > max_viol:
                max_viol = diff
    return max_viol
# EVOLVE-BLOCK-END
