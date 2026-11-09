"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: Specification of the value distribution
        support: Support of the value space
        constraints: Additional constraints
        
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit

    values_per_good = [0, 1, 2, 3]
    all_reports = list(product(values_per_good, values_per_good))
    n_types = 16
    V = np.array(all_reports, dtype=float)  # (16, 2)

    # Parse distribution_spec to get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([float(distribution_spec.get(r, 0.0)) for r in all_reports])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.asarray(distribution_spec, dtype=float).flatten()
        if arr.size == 16:
            probs = arr
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0

    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(16) / 16.0

    # ---- Build the full 2-D finite-mechanism LP ----
    # Variables: x_{t,g} for t in 0..15, g in 0..1 (32 vars), then p_t (16 vars) => 48 vars
    n_vars = 48
    def x_idx(t, g): return t * 2 + g
    def p_idx(t): return 32 + t

    # Objective: maximize sum_t p_t * pi_t  => minimize -sum_t p_t * pi_t
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]

    A_ub_rows = []
    b_ub_rows = []

    # IC constraints: utility is convex in type => for any two types a,b:
    #   u(a) - u(b) >= <V_a - V_b, x_a>  (supergradient at a is x_a)
    # Equivalently: u(a) >= u(b) + <V_a - V_b, x_a>
    # In variables: p_a - p_b - <V_a, x_a> + <V_b, x_a> <= 0  ... wait, let me re-derive.
    # u(a) = <V_a, x_a> - p_a
    # u(b) = <V_b, x_b> - p_b
    # Convexity: u(a) - u(b) >= <V_a - V_b, x_a>
    # => <V_a, x_a> - p_a - (<V_b, x_b> - p_b) >= <V_a - V_b, x_a>
    # => <V_a, x_a> - p_a - <V_b, x_b> + p_b >= <V_a, x_a> - <V_b, x_a>
    # => -p_a - <V_b, x_b> + p_b >= -<V_b, x_a>
    # => p_a - p_b + <V_b, x_b> - <V_b, x_a> <= 0
    # => p_a - p_b + <V_b, (x_b - x_a)> <= 0
    # This is the standard IC: the payment difference is bounded by the value-weighted allocation difference.
    # Actually, the cleanest form for a single bidder is:
    #   u(a) >= u(b) + <V_a - V_b, x_a>   (IC: truth-telling maximizes utility)
    # Let's just enforce: for all a, b:
    #   <V_a, x_a> - p_a >= <V_a, x_b> - p_b
    # => p_a - p_b + <V_a, x_b> - <V_a, x_a> <= 0
    # => p_a - p_b - <V_a, (x_a - x_b)> <= 0
    # This is 16*16 = 256 constraints. But we can reduce to adjacent types (48 constraints)
    # because IC on a lattice with convex utility is equivalent to IC on edges.
    # For safety and since 256 is small, let's use all pairs.

    for a in range(n_types):
        for b in range(n_types):
            if a == b:
                continue
            row = np.zeros(n_vars)
            row[p_idx(a)] = 1.0
            row[p_idx(b)] = -1.0
            # -<V_a, x_a - x_b> = -V_a[0]*(x_a0 - x_b0) - V_a[1]*(x_a1 - x_b1)
            # = -V_a[0]*x_a0 + V_a[0]*x_b0 - V_a[1]*x_a1 + V_a[1]*x_b1
            row[x_idx(a, 0)] = -V[a, 0]
            row[x_idx(b, 0)] = V[a, 0]
            row[x_idx(a, 1)] = -V[a, 1]
            row[x_idx(b, 1)] = V[a, 1]
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

    # IR constraints: u(t) >= 0 for all t
    # <V_t, x_t> - p_t >= 0  =>  p_t - <V_t, x_t> <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(t)] = 1.0
        row[x_idx(t, 0)] = -V[t, 0]
        row[x_idx(t, 1)] = -V[t, 1]
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # Bounds: 0 <= x_{t,g} <= 1, 0 <= p_t <= infinity (but IR + IC will bound it)
    bounds = []
    for t in range(n_types):
        for g in range(2):
            bounds.append((0.0, 1.0))
        bounds.append((0.0, None))

    allocations = np.zeros((16, 2))
    payments = np.zeros(16)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            for t in range(n_types):
                for g in range(2):
                    allocations[t, g] = res.x[x_idx(t, g)]
                payments[t] = res.x[p_idx(t)]
        else:
            # LP failed; use fallback
            raise RuntimeError("LP failed: " + res.message)
    except Exception:
        # Fallback: simple threshold mechanism (always IC/IR)
        for i, (v0, v1) in enumerate(all_reports):
            allocations[i, 0] = 1.0 if v0 >= 1 else 0.0
            allocations[i, 1] = 1.0 if v1 >= 1 else 0.0
            payments[i] = (1.0 if v0 >= 1 else 0.0) + (1.0 if v1 >= 1 else 0.0)

    # Final validation
    max_regret = 0.0
    min_ir = np.inf
    min_pay = np.inf
    for t in range(16):
        v_t = V[t]
        util_t = v_t @ allocations[t] - payments[t]
        if util_t < min_ir:
            min_ir = util_t
        if payments[t] < min_pay:
            min_pay = payments[t]
        for tp in range(16):
            util_tp_as_t = v_t @ allocations[tp] - payments[tp]
            regret = util_tp_as_t - util_t
            if regret > max_regret:
                max_regret = regret

    if max_regret > 1e-8 or min_ir < -1e-8 or min_pay < -1e-8:
        # Fallback to safe threshold
        for i, (v0, v1) in enumerate(all_reports):
            allocations[i, 0] = 1.0 if v0 >= 1 else 0.0
            allocations[i, 1] = 1.0 if v1 >= 1 else 0.0
            payments[i] = (1.0 if v0 >= 1 else 0.0) + (1.0 if v1 >= 1 else 0.0)

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
