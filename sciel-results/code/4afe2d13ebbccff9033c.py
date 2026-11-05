"""Incentive-compatible finite auction design solver with explicit LP formulation."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Parse inputs
    values = [0, 1, 2, 3]
    n_values = len(values)
    
    # Determine probabilities for each good
    if isinstance(distribution_spec, dict):
        # Try to extract probabilities from dict
        p1 = None
        p2 = None
        if 'p1' in distribution_spec:
            p1 = distribution_spec['p1']
        if 'p2' in distribution_spec:
            p2 = distribution_spec['p2']
        if 'g1' in distribution_spec:
            p1 = distribution_spec['g1']
        if 'g2' in distribution_spec:
            p2 = distribution_spec['g2']
        if 'probs' in distribution_spec:
            probs = distribution_spec['probs']
            if len(probs) >= 2:
                p1 = probs[0]
                p2 = probs[1]
        if 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if len(dist) >= 2:
                p1 = dist[0]
                p2 = dist[1]
        
        # If still None, try to infer from other keys
        if p1 is None or p2 is None:
            # Try to find arrays of length 4
            for key, val in distribution_spec.items():
                if isinstance(val, (list, np.ndarray)) and len(val) == 4:
                    if p1 is None:
                        p1 = val
                    elif p2 is None:
                        p2 = val
                    else:
                        break
        
        # Default if still not found
        if p1 is None:
            p1 = [0.25, 0.25, 0.25, 0.25]
        if p2 is None:
            p2 = [0.25, 0.25, 0.25, 0.25]
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) >= 2:
            p1 = distribution_spec[0]
            p2 = distribution_spec[1]
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    # Normalize probabilities
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.array([0.25, 0.25, 0.25, 0.25])
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Support: list of allowed reports (value pairs)
    # Each report is a pair (v1, v2) where v1, v2 in {0,1,2,3}
    if support is None or len(support) == 0:
        # Default: all 16 possible reports
        support = [(v1, v2) for v1 in values for v2 in values]
    
    # Convert support to list of tuples
    support_list = []
    for s in support:
        if isinstance(s, (list, tuple, np.ndarray)) and len(s) == 2:
            support_list.append((int(s[0]), int(s[1])))
        elif isinstance(s, (list, tuple, np.ndarray)) and len(s) == 4:
            # Might be flat representation
            support_list.append((int(s[0]), int(s[1])))
        else:
            support_list.append((int(s), int(s)))
    
    # Deduplicate and ensure valid
    support_list = list(set(support_list))
    support_list = [(v1, v2) for v1, v2 in support_list if v1 in values and v2 in values]
    
    if len(support_list) == 0:
        support_list = [(v1, v2) for v1 in values for v2 in values]
    
    n_reports = len(support_list)
    report_to_idx = {r: i for i, r in enumerate(support_list)}
    
    # Joint distribution: P(v1, v2) = p1[v1] * p2[v2] (independent)
    # For each report r=(r1, r2), the probability of true type (v1, v2) given report r
    # Actually, we need the prior probability of each type
    
    # Types are (v1, v2) where v1, v2 in {0,1,2,3}
    # All 16 types, each with probability p1[v1]*p2[v2]
    
    all_types = [(v1, v2) for v1 in values for v2 in values]
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    n_types = len(all_types)
    
    # Prior probabilities
    prior = np.zeros(n_types)
    for t in all_types:
        prior[type_to_idx[t]] = p1[t[0]] * p2[t[1]]
    
    # LP Variables:
    # For each report r and type t: x[r, t] = allocation probability of good 1 when report r is made and true type is t
    # y[r, t] = allocation probability of good 2 when report r is made and true type is t
    # q[r] = payment when report r is made (same for all types, but in general can depend on type)
    # Actually, in mechanism design, allocation and payment can depend on the reported type.
    # Let's define:
    # a1[r] = allocation of good 1 when report r is made
    # a2[r] = allocation of good 2 when report r is made  
    # q[r] = payment when report r is made
    
    # But for incentive compatibility, we need to consider that a bidder with true type t
    # reporting r gets utility: v1[t]*a1[r] + v2[t]*a2[r] - q[r]
    # And they should prefer reporting their true type t over any other report r:
    # v1[t]*a1[t] + v2[t]*a2[t] - q[t] >= v1[t]*a1[r] + v2[t]*a2[r] - q[r] for all r
    
    # Variables: a1[r], a2[r], q[r] for each report r
    # Total variables: 3 * n_reports
    
    # Objective: Maximize expected revenue = sum_t prior[t] * q[t]
    # (Since payment is made when report t is made, and true type is t with probability prior[t])
    
    # Wait, the expected revenue is sum over all true types t of prior[t] * (payment when reporting t)
    # Because a truthful bidder reports their true type.
    
    # Constraints:
    # 1. Feasibility: 0 <= a1[r] <= 1, 0 <= a2[r] <= 1 for all r
    # 2. Nonnegative payments: q[r] >= 0 for all r
    # 3. Individual rationality: For each type t, utility when reporting t >= 0
    #    v1[t]*a1[t] + v2[t]*a2[t] - q[t] >= 0
    # 4. Incentive compatibility: For each pair (t, r), 
    #    v1[t]*a1[t] + v2[t]*a2[t] - q[t] >= v1[t]*a1[r] + v2[t]*a2[r] - q[r]
    #    Rearranged: q[r] - q[t] >= v1[t]*(a1[r] - a1[t]) + v2[t]*(a2[r] - a2[t])
    
    n_vars = 3 * n_reports
    # Variable indices:
    # a1[r] at index r
    # a2[r] at index n_reports + r
    # q[r] at index 2*n_reports + r
    
    # Objective: maximize sum_t prior[t] * q[t]
    # In linprog, we minimize, so minimize -sum_t prior[t] * q[t]
    c = np.zeros(n_vars)
    for t in all_types:
        if t in report_to_idx:
            r_idx = report_to_idx[t]
            c[2*n_reports + r_idx] = -prior[type_to_idx[t]]
    
    # Bounds
    bounds = []
    for r in range(n_reports):
        bounds.append((0, 1))  # a1[r]
        bounds.append((0, 1))  # a2[r]
        bounds.append((0, None))  # q[r]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []
    
    # IR constraints: for each type t that is in support
    for t in all_types:
        if t in report_to_idx:
            r_idx = report_to_idx[t]
            # v1[t]*a1[t] + v2[t]*a2[t] - q[t] >= 0
            # => -v1[t]*a1[t] - v2[t]*a2[t] + q[t] <= 0
            row = np.zeros(n_vars)
            row[r_idx] = -t[0]
            row[n_reports + r_idx] = -t[1]
            row[2*n_reports + r_idx] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    # IC constraints: for each pair (t, r) where t is in support
    for t in all_types:
        if t not in report_to_idx:
            continue
        t_idx = report_to_idx[t]
        for r in support_list:
            if r not in report_to_idx:
                continue
            r_idx = report_to_idx[r]
            if t_idx == r_idx:
                continue
            # q[r] - q[t] >= v1[t]*(a1[r] - a1[t]) + v2[t]*(a2[r] - a2[t])
            # => -v1[t]*a1[r] + v1[t]*a1[t] - v2[t]*a2[r] + v2[t]*a2[t] + q[r] - q[t] <= 0
            row = np.zeros(n_vars)
            row[r_idx] = -t[0]      # -v1[t]*a1[r]
            row[t_idx] = t[0]       # +v1[t]*a1[t]
            row[n_reports + r_idx] = -t[1]  # -v2[t]*a2[r]
            row[n_reports + t_idx] = t[1]   # +v2[t]*a2[t]
            row[2*n_reports + r_idx] = 1    # +q[r]
            row[2*n_reports + t_idx] = -1   # -q[t]
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return a simple mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    a1 = result.x[:n_reports]
    a2 = result.x[n_reports:2*n_reports]
    q = result.x[2*n_reports:3*n_reports]
    
    # Build output: allocations (16, 2) and payments (16,)
    # The output should be indexed by report index 0-15, corresponding to (v1, v2) in some order
    # Let's use the natural ordering: index = v1*4 + v2
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i, r in enumerate(support_list):
        idx = r[0] * 4 + r[1]
        allocations[idx, 0] = a1[i]
        allocations[idx, 1] = a2[i]
        payments[idx] = q[i]
    
    # Check time
    if time.time() > deadline:
        pass  # We're already done
    
    return {'allocations': allocations, 'payments': payments}
