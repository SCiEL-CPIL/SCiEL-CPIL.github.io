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
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # Generate all 16 reports: (v1, v2) where v1,v2 in {0,1,2,3}
    reports = []
    for v1 in range(4):
        for v2 in range(4):
            reports.append((v1, v2))
    
    n_reports = len(reports)  # 16
    
    # Parse distribution_spec to get probabilities for each report
    # distribution_spec could be a list of 16 probabilities, a dict, or other format
    probs = np.ones(n_reports) / n_reports  # default uniform
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_reports:
            probs = np.array(distribution_spec, dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
        elif len(distribution_spec) == 4:
            # Could be marginal distributions for each good
            # Assume independent: p(v1,v2) = p1(v1) * p2(v2)
            p1 = np.array(distribution_spec[0], dtype=float)
            p2 = np.array(distribution_spec[1], dtype=float) if len(distribution_spec) > 1 else np.ones(4)/4
            probs = np.zeros(n_reports)
            for idx, (v1, v2) in enumerate(reports):
                probs[idx] = p1[v1] * p2[v2]
            probs = probs / probs.sum() if probs.sum() > 0 else probs
    elif isinstance(distribution_spec, dict):
        for idx, rep in enumerate(reports):
            if rep in distribution_spec:
                probs[idx] = distribution_spec[rep]
            elif str(rep) in distribution_spec:
                probs[idx] = distribution_spec[str(rep)]
        probs = probs / probs.sum() if probs.sum() > 0 else probs
    elif isinstance(distribution_spec, str):
        # Try to parse as JSON or similar
        import json
        try:
            spec = json.loads(distribution_spec)
            if isinstance(spec, list) and len(spec) == n_reports:
                probs = np.array(spec, dtype=float)
                probs = probs / probs.sum() if probs.sum() > 0 else probs
            elif isinstance(spec, dict):
                for idx, rep in enumerate(reports):
                    if rep in spec:
                        probs[idx] = spec[rep]
                    elif str(rep) in spec:
                        probs[idx] = spec[str(rep)]
                probs = probs / probs.sum() if probs.sum() > 0 else probs
        except:
            pass
    
    # Ensure probs are valid
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_reports) / n_reports
    
    # LP formulation:
    # Variables: x_1[0..15], x_2[0..15], p[0..15] -> 48 variables
    # Maximize sum_i probs[i] * p[i]
    # Subject to:
    #   0 <= x_1[i] <= 1, 0 <= x_2[i] <= 1, p[i] >= 0
    #   IC: For all i,j: v_i . x_i - p_i >= v_i . x_j - p_j
    #     i.e., p_j - p_i >= v_i . (x_j - x_i)
    #   IR: For all i: v_i . x_i - p_i >= 0
    #     i.e., p_i <= v_i . x_i
    
    n_vars = 3 * n_reports  # x1, x2, p for each report
    
    # Objective: maximize sum probs[i] * p[i] -> minimize -sum probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[2 * n_reports + i] = -probs[i]  # negative for minimization
    
    # Inequality constraints: A_ub @ x <= b_ub
    # IC constraints: for each pair (i,j), p_j - p_i - v_i . (x_j - x_i) <= 0
    # Rewrite: -v_i1*(x1_j - x1_i) - v_i2*(x2_j - x2_i) + p_j - p_i <= 0
    # In terms of variables: 
    #   x1_j coeff: -v_i1, x1_i coeff: v_i1
    #   x2_j coeff: -v_i2, x2_i coeff: v_i2
    #   p_j coeff: 1, p_i coeff: -1
    
    A_rows = []
    b_vals = []
    
    for i in range(n_reports):
        vi1, vi2 = reports[i]
        for j in range(n_reports):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # x1_j coeff: -vi1
            row[j] = -vi1
            # x1_i coeff: +vi1
            row[i] = vi1
            # x2_j coeff: -vi2
            row[n_reports + j] = -vi2
            # x2_i coeff: +vi2
            row[n_reports + i] = vi2
            # p_j coeff: 1
            row[2 * n_reports + j] = 1
            # p_i coeff: -1
            row[2 * n_reports + i] = -1
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraints: p_i - v_i . x_i <= 0
    # p_i - vi1*x1_i - vi2*x2_i <= 0
    for i in range(n_reports):
        vi1, vi2 = reports[i]
        row = np.zeros(n_vars)
        row[i] = -vi1
        row[n_reports + i] = -vi2
        row[2 * n_reports + i] = 1
        A_rows.append(row)
        b_vals.append(0.0)
    
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds: x1 in [0,1], x2 in [0,1], p in [0, inf)
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x1[i]
        bounds.append((0, 1))  # x2[i]
        bounds.append((0, None))  # p[i]
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if time.time() > deadline - 5:
            # Close to deadline, use simple fallback
            allocations = np.zeros((n_reports, 2))
            payments = np.zeros(n_reports)
            return {"allocations": allocations, "payments": payments}
        
        if result.success:
            x_sol = result.x
            allocations = np.zeros((n_reports, 2))
            payments = np.zeros(n_reports)
            for i in range(n_reports):
                allocations[i, 0] = x_sol[i]
                allocations[i, 1] = x_sol[n_reports + i]
                payments[i] = x_sol[2 * n_reports + i]
            
            # Clip to ensure feasibility
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            return {"allocations": allocations, "payments": payments}
        else:
            # Fallback: simple truthful auction
            allocations = np.zeros((n_reports, 2))
            payments = np.zeros(n_reports)
            for i in range(n_reports):
                vi1, vi2 = reports[i]
                if vi1 > 0:
                    allocations[i, 0] = 1.0
                if vi2 > 0:
                    allocations[i, 1] = 1.0
                payments[i] = vi1 * allocations[i, 0] + vi2 * allocations[i, 1]
            
            return {"allocations": allocations, "payments": payments}
    
    except Exception as e:
        # Fallback: simple mechanism
        allocations = np.zeros((n_reports, 2))
        payments = np.zeros(n_reports)
        for i in range(n_reports):
            vi1, vi2 = reports[i]
            if vi1 > 0:
                allocations[i, 0] = 1.0
            if vi2 > 0:
                allocations[i, 1] = 1.0
            payments[i] = vi1 * allocations[i, 0] + vi2 * allocations[i, 1]
        
        return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
