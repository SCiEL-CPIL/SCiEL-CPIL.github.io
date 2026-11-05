"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # --- Parse inputs ---
    # distribution_spec: likely a list of 4 probabilities for values {0,1,2,3}
    # support: set of allowed reports, probably (v1, v2) pairs where v1,v2 in {0,1,2,3}
    # constraints: dict or object with constraints

    # We need to handle the specific format. Let's assume:
    # distribution_spec is a list [p0, p1, p2, p3] for one good, and values are independent.
    # Or it could be a joint distribution. The task says "three fixed distributions" and "each value in {0,1,2,3}".
    # Let's assume independent identical distributions for both goods unless specified otherwise.
    
    # Let's try to infer the structure from typical auction problems.
    # Usually, we have a type space. Here type is (v1, v2).
    # If distribution_spec is a list of 4, it's the marginal.
    
    try:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
            if len(probs) == 4:
                # Independent goods
                p_vals = probs
                # Joint probability for (v1, v2) is p[v1] * p[v2]
            else:
                # Maybe it's a flat joint distribution? 4*4=16
                p_vals = probs.reshape(4, 4) if len(probs) == 16 else probs
        elif isinstance(distribution_spec, dict):
            # Handle dict format if needed
            p_vals = np.zeros((4,4))
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    p_vals[k[0], k[1]] = v
            # If it's marginal dict
            if all(isinstance(k, int) for k in distribution_spec.keys()):
                p_vals = np.array([distribution_spec.get(i, 0) for i in range(4)])
                p_vals = np.outer(p_vals, p_vals)
        else:
            # Default: uniform
            p_vals = np.ones((4,4)) / 16.0
    except Exception:
        p_vals = np.ones((4,4)) / 16.0

    # Normalize probabilities
    p_vals = p_vals / p_vals.sum()
    
    # Ensure p_vals is 4x4
    if p_vals.shape != (4,4):
        if p_vals.size == 4:
            p_vals = np.outer(p_vals, p_vals)
        else:
            p_vals = np.ones((4,4)) / 16.0
        p_vals = p_vals / p_vals.sum()

    # Support: set of allowed reports
    # If support is None, assume all (v1, v2) in {0,1,2,3}^2
    if support is None:
        reports = [(i, j) for i in range(4) for j in range(4)]
    else:
        # support might be a list of tuples
        if isinstance(support, (list, tuple, set)):
            reports = list(support)
        else:
            reports = [(i, j) for i in range(4) for j in range(4)]
    
    # Ensure reports are tuples of ints
    reports = [tuple(r) for r in reports]
    
    # If reports is empty, use all
    if not reports:
        reports = [(i, j) for i in range(4) for j in range(4)]

    n_reports = len(reports)
    report_index = {r: i for i, r in enumerate(reports)}
    
    # Variables:
    # x[r, g] = allocation probability for good g (0 or 1) when report r
    # p[r] = payment when report r
    
    # Total variables: n_reports * 2 (allocations) + n_reports (payments)
    # But allocations are 0/1? No, "feasible probabilities" suggests mixed strategies or randomized allocations.
    # "allocations:(16,2)" implies the output is a matrix of size 16x2.
    # Wait, the output interface is `allocations:(16,2), payments:(16,)`.
    # This implies there are exactly 16 reports in the output, corresponding to all (v1,v2) in {0,1,2,3}^2.
    # So the output is always for the full type space, even if support is smaller?
    # "Every allowed report is enumerated" - likely the support is the full space or a subset.
    # But the return shape is fixed at 16. So we must assume the full space {0,1,2,3}^2.
    
    # Let's force reports to be the full space for the output, but optimize over the allowed support?
    # Actually, if support is smaller, we only care about truthfulness on that support.
    # But the output must be 16x2. So we define allocations for all 16 types.
    
    # Let's assume the optimization is over the full 16 types, but constraints only apply to the 'support' subset?
    # Or is 'support' just the set of types that have non-zero probability?
    # "Truthfulness holds only on the declared report/value support."
    
    # Let's define the full set of types
    all_types = [(i, j) for i in range(4) for j in range(4)]
    type_index = {t: i for i, t in enumerate(all_types)}
    
    # If 'support' is provided, it might be the set of types where we require incentive compatibility.
    # Let's assume we optimize over all 16 types.
    
    # Objective: Maximize expected revenue
    # Revenue = sum_{t} prob(t) * (payment(t) - value(t))  ... wait, revenue is payment - cost?
    # In auction design, revenue is the sum of payments. The platform doesn't pay for goods.
    # So Expected Revenue = sum_{t} prob(t) * payment(t).
    # But individual rationality requires payment(t) <= value(t) (for the bidder).
    # So we maximize sum p[t] * pay[t].
    
    # Constraints:
    # 1. Feasibility: 0 <= alloc[t, g] <= 1
    # 2. Individual Rationality (IR): payment(t) <= sum_g alloc[t, g] * t[g]  (for all t in support)
    #    Wait, IR is usually: utility(t, t) >= 0.
    #    Utility(t, r) = sum_g alloc[r, g] * t[g] - payment(r)
    #    So IR: sum_g alloc[t, g] * t[g] - payment(t) >= 0  => payment(t) <= sum_g alloc[t, g] * t[g]
    # 3. Incentive Compatibility (IC): Utility(t, t) >= Utility(t, r) for all t, r in support.
    #    sum_g alloc[t, g] * t[g] - payment(t) >= sum_g alloc[r, g] * t[g] - payment(r)
    #    payment(r) - payment(t) >= sum_g (alloc[r, g] - alloc[t, g]) * t[g]
    
    # This is a linear program.
    # Variables: alloc[16, 2], pay[16]
    # Total 48 variables.
    
    # Let's use scipy.optimize.linprog
    # Maximize c^T x  =>  Minimize -c^T x
    
    # c vector:
    # For alloc variables: 0
    # For pay variables: -prob(t) (since we maximize sum prob(t)*pay(t), we minimize -sum prob(t)*pay(t))
    
    x = np.zeros((16, 2) + (16,))
    # Order: alloc[0,0], alloc[0,1], ..., alloc[15,1], pay[0], ..., pay[15]
    # Indices 0-31: alloc
    # Indices 32-47: pay
    
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -p_vals[i//4, i%4]
    
    # Constraints:
    # 1. 0 <= alloc <= 1
    #    This is handled by bounds.
    
    # 2. IR: payment(t) <= sum_g alloc[t, g] * t[g]
    #    payment(t) - sum_g alloc[t, g] * t[g] <= 0
    #    For each t in support (or all types?):
    #    Let's assume support is all types if not specified.
    
    # 3. IC: payment(r) - payment(t) - sum_g (alloc[r, g] - alloc[t, g]) * t[g] <= 0
    #    For all t, r in support.
    
    # Let's build the A_ub matrix.
    # Number of constraints:
    # IR: 16
    # IC: 16 * 16 = 256
    # Total: 272
    
    A_ub = []
    b_ub = []
    
    # IR constraints
    for t in all_types:
        row = np.zeros(48)
        # payment(t) <= alloc[t,0]*t[0] + alloc[t,1]*t[1]
        # payment(t) - alloc[t,0]*t[0] - alloc[t,1]*t[1] <= 0
        ti = type_index[t]
        row[32 + ti] = 1
        row[ti*2 + 0] = -t[0]
        row[ti*2 + 1] = -t[1]
        A_ub.append(row)
        b_ub.append(0)
        
    # IC constraints
    for t in all_types:
        for r in all_types:
            row = np.zeros(48)
            ti = type_index[t]
            ri = type_index[r]
            # payment(r) - payment(t) - (alloc[r,0]-alloc[t,0])*t[0] - (alloc[r,1]-alloc[t,1])*t[1] <= 0
            row[32 + ri] = 1
            row[32 + ti] = -1
            row[ri*2 + 0] = -t[0]
            row[ti*2 + 0] = t[0]
            row[ri*2 + 1] = -t[1]
            row[ti*2 + 1] = t[1]
            A_ub.append(row)
            b_ub.append(0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))  # Payments can be any non-negative? "nonnegative payments"
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            alloc = res.x[:32].reshape(16, 2)
            pay = res.x[32:]
            # Ensure non-negative payments
            pay = np.maximum(pay, 0)
            # Check if we are within deadline
            if time.time() > deadline:
                return {"allocations": alloc, "payments": pay}
            return {"allocations": alloc, "payments": pay}
        else:
            # Fallback: simple mechanism
            # e.g., VCG or second price
            # For simplicity, return uniform allocation and 0 payment? No, must be IR.
            # Let's try a simple heuristic: allocate good g to bidder if value > threshold, pay threshold.
            # But we need to satisfy constraints.
            # Let's just return the baseline if LP fails.
            pass
    except Exception:
        pass
        
    # Fallback mechanism
    # Simple: allocate good 1 if v1 >= 2, good 2 if v2 >= 2. Pay 2 if allocated.
    alloc = np.zeros((16, 2))
    pay = np.zeros(16)
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            if i >= 2:
                alloc[idx, 0] = 1
                pay[idx] += 2
            if j >= 2:
                alloc[idx, 1] = 1
                pay[idx] += 2
    return {"allocations": alloc, "payments": pay}

# EVOLVE-BLOCK-END
