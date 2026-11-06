"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # 1. Setup
    # Support is typically a list of 16 tuples (v1, v2) or similar.
    # The task says "Every allowed report is enumerated".
    # Values in {0,1,2,3}.
    if support is None or len(support) != 16:
        # Default support: all combinations of {0,1,2,3} x {0,1,2,3}
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        
    n_reports = len(support)
    
    # Parse distribution_spec
    # Expected format: likely a list of probabilities corresponding to the support,
    # or a dict, or a specific structure from the suite.
    # Given "three fixed distributions", distribution_spec might be an index or a name.
    # However, the contract says "distribution_spec" is passed.
    # Let's assume distribution_spec is a list/array of probabilities of length 16,
    # OR it's a key into a predefined set.
    # Looking at the baseline usage `auction_baseline(distribution_spec, support, constraints)`,
    # it likely handles the parsing. 
    # Since I am replacing the baseline, I must handle it.
    # Common formats:
    # 1. List of floats (probs)
    # 2. Dict {report: prob}
    # 3. String/Int index into a global registry (less likely for a generic interface unless defined elsewhere).
    
    # Let's try to infer probabilities. If it's a list, use it.
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_reports:
            probs = np.array(distribution_spec, dtype=float)
        else:
            # Maybe it's a 2D array?
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) != n_reports:
                raise ValueError("Could not parse distribution_spec into 16 probabilities")
    elif isinstance(distribution_spec, dict):
        probs = np.zeros(n_reports)
        for i, rep in enumerate(support):
            if rep in distribution_spec:
                probs[i] = distribution_spec[rep]
            # Try tuple conversion if keys are lists
            elif tuple(rep) in distribution_spec:
                probs[i] = distribution_spec[tuple(rep)]
            else:
                # Fallback: maybe keys are indices?
                if i in distribution_spec:
                    probs[i] = distribution_spec[i]
    else:
        # If it's an int or string, we might need to guess the distribution.
        # The task mentions "three fixed distributions".
        # Without knowing the exact mapping, we assume a uniform distribution as a safe fallback 
        # or try to interpret it as an index if possible. 
        # However, usually these tasks provide explicit data.
        # Let's assume if it's not a list/dict, it might be an index 0,1,2 for 3 dists.
        # But we don't have the dists. 
        # Let's assume uniform if we can't parse it, to ensure validity.
        probs = np.ones(n_reports) / n_reports
        
    probs = probs / np.sum(probs) # Normalize just in case
    
    # 2. Formulate LP
    # Variables:
    # x[i, g] : Probability that good g is allocated to bidder i (i in 0..15, g in 0..1)
    # p[i]    : Payment made by bidder i
    # Total variables: 16*2 + 16 = 48
    
    # Objective: Maximize Expected Revenue
    # Max sum_i probs[i] * p[i]
    # Linprog minimizes, so we minimize -sum_i probs[i] * p[i]
    
    c = np.zeros(48)
    # p[i] is at index 16 + i
    for i in range(n_reports):
        c[16 + i] = -probs[i]
        
    # Constraints:
    # 1. Feasibility: 0 <= x[i,g] <= 1
    #    These are handled by bounds.
    
    # 2. Individual Rationality (IR):
    #    Utility u_i(i) >= 0 for all i.
    #    u_i(i) = v_i[0]*x[i,0] + v_i[1]*x[i,1] - p[i] >= 0
    #    => p[i] - v_i[0]*x[i,0] - v_i[1]*x[i,1] <= 0
    #    This is an inequality constraint: A_ub @ vars <= b_ub
    
    # 3. Incentive Compatibility (IC):
    #    u_i(i) >= u_i(j) for all i, j.
    #    v_i[0]*x[i,0] + v_i[1]*x[i,1] - p[i] >= v_i[0]*x[j,0] + v_i[1]*x[j,1] - p[j]
    #    => p[i] - p[j] + v_i[0]*(x[i,0]-x[j,0]) + v_i[1]*(x[i,1]-x[j,1]) <= 0
    
    # Let's build A_ub and b_ub
    # Number of constraints: 16 (IR) + 16*16 (IC) = 16 + 256 = 272
    
    n_constraints = 16 + 16 * 16
    A_ub = np.zeros((n_constraints, 48))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    # IR Constraints
    for i in range(n_reports):
        v1, v2 = support[i]
        # p[i] - v1*x[i,0] - v2*x[i,1] <= 0
        A_ub[row, 16 + i] = 1.0       # coeff for p[i]
        A_ub[row, i * 2] = -v1        # coeff for x[i,0]
        A_ub[row, i * 2 + 1] = -v2    # coeff for x[i,1]
        b_ub[row] = 0.0
        row += 1
        
    # IC Constraints
    for i in range(n_reports):
        vi1, vi2 = support[i]
        for j in range(n_reports):
            vj1, vj2 = support[j]
            # p[i] - p[j] + vi1*x[i,0] - vi1*x[j,0] + vi2*x[i,1] - vi2*x[j,1] <= 0
            A_ub[row, 16 + i] = 1.0       # p[i]
            A_ub[row, 16 + j] = -1.0      # p[j]
            A_ub[row, i * 2] = vi1        # x[i,0]
            A_ub[row, i * 2 + 1] = vi2    # x[i,1]
            A_ub[row, j * 2] = -vj1       # x[j,0]
            A_ub[row, j * 2 + 1] = -vj2   # x[j,1]
            b_ub[row] = 0.0
            row += 1
            
    # Bounds for variables
    # x[i,g] in [0, 1]
    # p[i] in [0, inf) (Non-negative payments)
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1)) # x[i,0]
        bounds.append((0, 1)) # x[i,1]
    for i in range(n_reports):
        bounds.append((0, None)) # p[i]
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception as e:
        # Fallback to uniform allocation, zero payment if LP fails
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
        
    if not res.success:
        # If LP fails, return a trivially valid auction (no trade)
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
        
    x_opt = res.x
    
    # Extract allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(n_reports):
        allocations[i, 0] = x_opt[i * 2]
        allocations[i, 1] = x_opt[i * 2 + 1]
        payments[i] = x_opt[16 + i]
        
    # Clean up numerical noise
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, we still return the result as it's already computed
        pass
        
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
