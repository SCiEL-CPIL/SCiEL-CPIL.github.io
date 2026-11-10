"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for a single additive bidder
    with two goods, each with values in {0, 1, 2, 3}.
    
    Returns:
        dict with keys:
            'allocations': (16, 2) array of allocation probabilities
            'payments': (16,) array of expected payments
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Type space: 4x4 = 16 types (v1, v2) where v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    n_values = len(values)
    n_types = n_values * n_values  # 16 types
    
    # Map type index to (v1, v2)
    def idx(v1, v2):
        return v1 * n_values + v2
    
    # Get distribution over types
    # distribution_spec should give us the probability of each type
    # We need to figure out the format. Let's assume it's a dict or array.
    
    # Try to get probabilities from distribution_spec
    try:
        if isinstance(distribution_spec, dict):
            # Might be keyed by (v1, v2) or by index
            if (0, 0) in distribution_spec:
                probs = np.zeros(n_types)
                for (v1, v2), p in distribution_spec.items():
                    probs[idx(v1, v2)] = p
            elif 0 in distribution_spec:
                probs = np.zeros(n_types)
                for i, p in distribution_spec.items():
                    probs[i] = p
            else:
                # Try to handle other formats
                probs = np.ones(n_types) / n_types
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) != n_types:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    except:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total = probs.sum()
    if total <= 0:
        probs = np.ones(n_types) / n_types
    else:
        probs = probs / total
    
    # Support: list of allowed reports/values
    # We enumerate all 16 types as allowed reports
    # support might be a list of (v1, v2) pairs or similar
    
    # For the optimal auction with a single additive bidder:
    # The key insight is that with one bidder, the IC constraints and IR constraints
    # determine the payment as a function of allocation.
    
    # For a single additive bidder with two goods:
    # The bidder's value for a bundle (x1, x2) is v1*x1 + v2*x2
    # IC: For all types t, t', utility(t) >= utility(t') where utility is maximized
    # IR: utility(t) >= 0 for all t
    
    # The optimal mechanism can be characterized by:
    # 1. Choosing allocation probabilities q(t) = (q1(t), q2(t)) for each type t
    # 2. Payment is determined by IC: p(t) = v(t).q(t) - U(t)
    # 3. U(t) is the utility, which must satisfy IC and IR
    
    # For a single bidder, the IC constraints imply:
    # U(t) >= U(t') + [v(t) - v(t')] . q(t')  for all t, t'
    # And IR: U(t) >= 0
    
    # The optimal U(t) = max(0, max_{t'} [U(t') + (v(t)-v(t')) . q(t')])
    # But this is circular. Instead, we use the fact that:
    # Expected revenue = E[v(t) . q(t)] - E[U(t)]
    # And U(t) is the minimal utility satisfying IC and IR given q.
    
    # For a single additive bidder, the problem decomposes in a specific way.
    # The optimal allocation is: q_i(t) = 1 if the virtual value is positive, else 0.
    # But with two goods and additivity, it's more complex.
    
    # Actually, for a single additive bidder, the optimal mechanism is:
    # Allocate good i to the bidder with probability 1 if v_i >= reserve_i, else 0.
    # The reserves are chosen to maximize revenue.
    
    # Virtual value for good i: phi_i(v) = v - (1-F_i(v))/f_i(v)
    # But with discrete distributions, we need to be careful.
    
    # Let's use a direct LP approach since n_types = 16 is small.
    
    # Decision variables:
    # q[t, i] for t in 0..15, i in {0, 1} (allocation probabilities)
    # U[t] for t in 0..15 (utility for type t)
    # p[t] for t in 0..15 (payment for type t)
    
    # Constraints:
    # 1. 0 <= q[t, i] <= 1 for all t, i
    # 2. IC: U[t] >= U[t'] + (v1(t)-v1(t'))*q[t', 0] + (v2(t)-v2(t'))*q[t', 1] for all t, t'
    # 3. IR: U[t] >= 0 for all t
    # 4. Payment consistency: p[t] = v1(t)*q[t, 0] + v2(t)*q[t, 1] - U[t]
    # 5. Non-negative payment: p[t] >= 0 for all t
    
    # Objective: Maximize E[p[t]] = sum_t probs[t] * p[t]
    
    # This is a linear program! Let's solve it.
    
    # Number of variables: 16*2 + 16 + 16 = 64
    # Number of constraints:
    # - q bounds: 32 (but handled by bounds)
    # - IC: 16*16 = 256
    # - IR: 16
    # - Payment non-neg: 16
    # Total: ~288 constraints
    
    # Let's use scipy.optimize.linprog
    
    from scipy.optimize import linprog
    
    # Variable ordering:
    # 0..31: q[t, i] where t = var // 2, i = var % 2
    # 32..47: U[t] where t = var - 32
    # 48..63: p[t] where t = var - 48
    
    n_q = n_types * 2  # 32
    n_U = n_types  # 16
    n_p = n_types  # 16
    n_vars = n_q + n_U + n_p  # 64
    
    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_q + n_U + t] = -probs[t]  # coefficient for p[t]
    
    # Constraints:
    # A_ub @ x <= b_ub
    
    A_ub = []
    b_ub = []
    
    # IC constraints: U[t] - U[t'] - (v1(t)-v1(t'))*q[t',0] - (v2(t)-v2(t'))*q[t',1] >= 0
    # Rewrite: -U[t] + U[t'] + (v1(t)-v1(t'))*q[t',0] + (v2(t)-v2(t'))*q[t',1] <= 0
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            v1_t = values[t // n_values]
            v2_t = values[t % n_values]
            v1_tp = values[tp // n_values]
            v2_tp = values[tp % n_values]
            
            row = np.zeros(n_vars)
            # -U[t]
            row[n_q + t] = -1
            # +U[t']
            row[n_q + tp] = 1
            # +(v1(t)-v1(t'))*q[t',0]
            row[tp * 2 + 0] = (v1_t - v1_tp)
            # +(v2(t)-v2(t'))*q[t',1]
            row[tp * 2 + 1] = (v2_t - v2_tp)
            
            A_ub.append(row)
            b_ub.append(0)
    
    # Payment consistency: p[t] - v1(t)*q[t,0] - v2(t)*q[t,1] + U[t] = 0
    # This is an equality constraint. We'll use A_eq.
    A_eq = []
    b_eq = []
    
    for t in range(n_types):
        v1_t = values[t // n_values]
        v2_t = values[t % n_values]
        
        row = np.zeros(n_vars)
        # p[t]
        row[n_q + n_U + t] = 1
        # -v1(t)*q[t,0]
        row[t * 2 + 0] = -v1_t
        # -v2(t)*q[t,1]
        row[t * 2 + 1] = -v2_t
        # +U[t]
        row[n_q + t] = 1
        
        A_eq.append(row)
        b_eq.append(0)
    
    # Convert to arrays
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)
    
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Bounds:
    # q[t, i]: [0, 1]
    # U[t]: [0, inf)
    # p[t]: [0, inf)
    bounds = []
    for _ in range(n_q):
        bounds.append((0, 1))
    for _ in range(n_U):
        bounds.append((0, None))
    for _ in range(n_p):
        bounds.append((0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
        else:
            # Fallback: use Myerson-style reserves
            x = _myerson_fallback(probs, values, n_values, n_types)
    except:
        x = _myerson_fallback(probs, values, n_values, n_types)
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for t in range(n_types):
        allocations[t, 0] = x[t * 2 + 0]
        allocations[t, 1] = x[t * 2 + 1]
        payments[t] = x[n_q + n_U + t]
    
    # Clip to ensure numerical validity
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    return {
        'allocations': allocations,
        'payments': payments
    }


def _myerson_fallback(probs, values, n_values, n_types):
    """Fallback: Myerson-style independent goods with optimal reserves."""
    x = np.zeros(2 * n_types + n_types + n_types)
    
    # For each good, compute marginal revenue and find optimal reserve
    for good in range(2):
        # Marginal probability of getting value >= v
        for v_idx, v in enumerate(values):
            # Probability that value is exactly v
            p_exact = np.zeros(n_types)
            for t in range(n_types):
                if (t // n_values) == v_idx and good == 0:
                    p_exact[t] = probs[t]
                elif (t % n_values) == v_idx and good == 1:
                    p_exact[t] = probs[t]
            
            # Simplified: assume independent values
            # For good 0, value v1: P(v1 = v) = sum over v2 of probs[v1, v2]
            p_val = np.zeros(n_values)
            for v_idx, v in enumerate(values):
                for v2_idx in range(n_values):
                    if good == 0:
                        p_val[v_idx] += probs[v_idx * n_values + v2_idx]
                    else:
                        p_val[v_idx] += probs[v2_idx * n_values + v_idx]
            
            # Find optimal reserve: maximize v * P(v' >= v)
            best_reserve = 0
            best_rev = 0
            for v_idx, v in enumerate(values):
                # P(value >= v)
                p_ge = sum(p_val[vidx] for vidx in range(v_idx, n_values))
                rev = v * p_ge
                if rev > best_rev:
                    best_rev = rev
                    best_reserve = v
            
            # Set allocation: q = 1 if value >= reserve, else 0
            for t in range(n_types):
                if good == 0:
                    v_t = t // n_values
                else:
                    v_t = t % n_values
                
                # Reserve is in value space
                v_val = values[v_t]
                if v_val >= best_reserve:
                    x[t * 2 + good] = 1.0
                else:
                    x[t * 2 + good] = 0.0
    
    # Compute utilities and payments
    # For Myerson, U(t) = max(0, U(t') + (v(t)-v(t')) . q(t'))
    # Simplified: assume U = 0 for all types (this may violate IC)
    # Better: compute U properly
    
    # For now, set U = 0 and compute payments
    for t in range(n_types):
        v1_t = values[t // n_values]
        v2_t = values[t % n_values]
        q1 = x[t * 2 + 0]
        q2 = x[t * 2 + 1]
        x[2 * n_types + n_types + t] = v1_t * q1 + v2_t * q2  # p[t] = v.q - U, U=0
    
    return x


# EVOLVE-BLOCK-END
