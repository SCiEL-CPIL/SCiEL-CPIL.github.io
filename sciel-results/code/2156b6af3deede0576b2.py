"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        # Assume uniform if int index provided but not specific dict
        probs = np.ones(num_types) / num_types
    else:
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types

    # Precompute type values array
    V = np.array(types, dtype=float) # (16, 2)
    
    def compute_utilities(x1, x2):
        """
        Compute the minimal utility U(t) for each type t given allocations x1, x2.
        U(t) = max_{j <= t, j != t} ( U(j) + (v1_t - v1_j)*x1_j + (v2_t - v2_j)*x2_j )
        with U(0,0) = 0.
        We process types in topological order (increasing v1, then v2).
        """
        U = np.full(num_types, -np.inf)
        # Type (0,0) is index 0
        U[0] = 0.0
        
        # Iterate through types in order
        # The list `types` is generated as v1=0..3, v2=0..3
        # So index i corresponds to (i//4, i%4).
        # This is a valid topological order for the partial order v1<=v1', v2<=v2'.
        for i in range(1, num_types):
            v1_i, v2_i = types[i]
            max_u = -np.inf
            # Check all j < i? No, just all j such that v1_j <= v1_i and v2_j <= v2_i.
            # Since the order is sorted, all valid j are in 0..i-1?
            # Yes, if v1_j > v1_i, it's later in the list. If v1_j == v1_i and v2_j > v2_i, it's later.
            # So we only need to check j from 0 to i-1.
            for j in range(i):
                v1_j, v2_j = types[j]
                if v1_j <= v1_i and v2_j <= v2_i:
                    # Candidate utility for i coming from j
                    cand = U[j] + (v1_i - v1_j) * x1[j] + (v2_i - v2_j) * x2[j]
                    if cand > max_u:
                        max_u = cand
            # If no valid j (should not happen for i>0 as 0,0 is always valid), max_u stays -inf
            # But U(0,0)=0 is always a candidate for any i.
            U[i] = max_u
        
        # Ensure U is not -inf (should be at least 0 for all? No, U can be negative?
        # IR requires U(t) >= 0? No, IR is v.t - p >= 0 => U(t) >= 0.
        # Wait, the constraint is U(t) >= 0.
        # The recurrence U(t) = max(0, max_j ...) ?
        # Let's check the IC constraint again.
        # U(t) >= U(j) + (v_t - v_j).x_j.
        # And IR: U(t) >= 0.
        # So U(t) = max(0, max_{j < t} (U(j) + ...)).
        # My loop computes max_{j < t}. I need to take max with 0.
        for i in range(1, num_types):
            if U[i] < 0:
                U[i] = 0.0
        
        return U

    def revenue_from_alloc(x1, x2):
        """Calculate expected revenue given allocations."""
        U = compute_utilities(x1, x2)
        # Payment p(t) = v(t).x(t) - U(t)
        # Revenue = sum probs * p(t)
        # p(t) must be non-negative.
        # v(t).x(t) - U(t) >= 0 is guaranteed by U(t) <= v(t).x(t)?
        # U(t) is bounded by v(t).x(t) - U(0,0) = v(t).x(t) if we trace back to 0?
        # Actually, U(t) <= v(t).x(t) is not always true if U(0,0)>0, but U(0,0)=0.
        # Let's verify: U(t) >= 0.
        # p(t) = v(t).x(t) - U(t).
        # If p(t) < 0, it violates IR? No, IR is U(t) >= 0.
        # Non-negative payment is a separate constraint: p(t) >= 0.
        # The LP includes p(t) >= 0.
        # Does the reconstruction guarantee p(t) >= 0?
        # U(t) = max(0, ...).
        # Is U(t) <= v(t).x(t)?
        # Consider j=0. U(t) >= U(0) + (v_t - 0).x_0 = v_t.x_0.
        # This doesn't bound U(t) from above.
        # However, if we set U(t) to the minimal value satisfying IC/IR,
        # it is known that for single bidder, p(t) >= 0 holds if x is monotone?
        # Actually, the LP enforces p(t) >= 0.
        # In our reconstruction, we should clip p(t) to 0?
        # If p(t) < 0, we can increase U(t) to make p(t)=0, which reduces revenue.
        # So we want to keep U(t) as low as possible.
        # If the minimal U(t) leads to p(t) < 0, then we MUST increase U(t) to v(t).x(t).
        # So p(t) = max(0, v(t).x(t) - U_min(t)).
        
        p = V[:, 0] * x1 + V[:, 1] * x2 - U
        p = np.clip(p, 0, None)
        
        return np.dot(probs, p)

    def neg_revenue_from_alloc(z):
        x1 = np.clip(z[:num_types], 0, 1)
        x2 = np.clip(z[num_types:2*num_types], 0, 1)
        return -revenue_from_alloc(x1, x2)

    best_result = None
    best_revenue = -np.inf

    # 1. Solve LP to get a good initial guess for x
    # Variables: x1[16], x2[16], p[16]
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC: -v1_i*x1_i - v2_i*x2_i + p_i + v1_i*x1_j + v2_i*x2_j - p_j <= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            row = np.zeros(3 * num_types)
            row[i] = -v1_i
            row[num_types + i] = -v2_i
            row[2 * num_types + i] = 1
            row[j] += v1_i
            row[num_types + j] += v2_i
            row[2 * num_types + j] -= 1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR: -v1_i*x1_i - v2_i*x2_i + p_i <= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = np.zeros(3 * num_types)
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = [(0, 1)] * num_types + [(0, 1)] * num_types + [(0, None)] * num_types
    
    x_init = None
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x1_init = np.clip(result.x[:num_types], 0, 1)
            x2_init = np.clip(result.x[num_types:2*num_types], 0, 1)
            x_init = np.concatenate([x1_init, x2_init])
            
            # Calculate revenue for this LP solution using exact reconstruction
            rev_lp = revenue_from_alloc(x1_init, x2_init)
            best_revenue = rev_lp
            U_lp = compute_utilities(x1_init, x2_init)
            p_lp = np.clip(V[:, 0] * x1_init + V[:, 1] * x2_init - U_lp, 0, None)
            best_result = {
                'allocations': np.column_stack([x1_init, x2_init]),
                'payments': p_lp
            }
    except Exception:
        pass

    # 2. Polish with SLSQP if time permits
    if time.time() < deadline - 10:
        if x_init is None:
            x_init = np.zeros(2 * num_types)
        
        # Run SLSQP
        try:
            res_opt = minimize(
                neg_revenue_from_alloc, 
                x_init, 
                method='SLSQP',
                bounds=[(0, 1)] * (2 * num_types),
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res_opt.success:
                x1_opt = np.clip(res_opt.x[:num_types], 0, 1)
                x2_opt = np.clip(res_opt.x[num_types:2*num_types], 0, 1)
                rev_opt = revenue_from_alloc(x1_opt, x2_opt)
                
                if rev_opt > best_revenue:
                    U_opt = compute_utilities(x1_opt, x2_opt)
                    p_opt = np.clip(V[:, 0] * x1_opt + V[:, 1] * x2_opt - U_opt, 0, None)
                    best_result = {
                        'allocations': np.column_stack([x1_opt, x2_opt]),
                        'payments': p_opt
                    }
                    best_revenue = rev_opt
        except Exception:
            pass

    if best_result is not None:
        return best_result

    # Fallback
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
