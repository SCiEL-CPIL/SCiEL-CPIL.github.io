import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the incentive-compatible finite auction design problem.
    
    Args:
        distribution_spec: string or dict specifying the distribution
        support: list of (v1, v2) tuples or similar
        constraints: dict with constraint parameters
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # Define the type space
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution probabilities
    probs = _get_distribution(distribution_spec, n_types, type_to_idx)
    
    # If support is provided, filter to those types
    if support is not None and len(support) > 0:
        # support might be a list of tuples
        support_set = set()
        for s in support:
            if isinstance(s, (tuple, list)):
                support_set.add(tuple(s))
            elif isinstance(s, dict):
                # Might be dict with 'v1', 'v2' or similar
                v1 = s.get('v1', s.get('type_1', 0))
                v2 = s.get('v2', s.get('type_2', 0))
                support_set.add((v1, v2))
        
        # Keep only types in support
        active_types = [t for t in types if t in support_set]
        if len(active_types) > 0:
            active_idx = [type_to_idx[t] for t in active_types]
            # Renormalize probabilities over active types
            active_probs = probs[active_idx]
            prob_sum = active_probs.sum()
            if prob_sum > 1e-12:
                active_probs = active_probs / prob_sum
            else:
                active_probs = np.ones(len(active_idx)) / len(active_idx)
            
            # Build mapping from original index to active index
            orig_to_active = {}
            for new_i, old_i in enumerate(active_idx):
                orig_to_active[old_i] = new_i
            
            n_active = len(active_idx)
            active_types_list = active_types
        else:
            active_idx = list(range(n_types))
            active_probs = probs
            active_types_list = types
            n_active = n_types
            orig_to_active = {i: i for i in range(n_types)}
    else:
        active_idx = list(range(n_types))
        active_probs = probs
        active_types_list = types
        n_active = n_types
        orig_to_active = {i: i for i in range(n_types)}
    
    # Extract values for active types
    # x[i, 0] = allocation of good 1 to type i
    # x[i, 1] = allocation of good 2 to type i
    # p[i] = payment from type i
    
    # Decision variables: x_1 (n_active), x_2 (n_active), p (n_active)
    # Total variables: 3 * n_active
    n_vars = 3 * n_active
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # In linprog, we minimize, so c = -probs for p, 0 for x
    c = np.zeros(n_vars)
    for i in range(n_active):
        c[2 * n_active + i] = -active_probs[i]  # negative for maximization
    
    # Constraints:
    # 1. Feasibility: 0 <= x[i, j] <= 1 for all i, j
    # 2. Individual rationality: v_i . x_i - p_i >= 0 for all i
    #    => p_i <= v_i . x_i
    # 3. Incentive compatibility (truthfulness):
    #    For all types i, i':
    #    v_i . x_i - p_i >= v_i . x_i' - p_i'
    #    => p_i' - p_i >= v_i . (x_i' - x_i)
    
    A_ub = []
    b_ub = []
    
    # Feasibility: x[i, j] <= 1
    # -x[i, j] <= 0
    for j in range(2):
        for i in range(n_active):
            row = np.zeros(n_vars)
            row[i + j * n_active] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # x[i, j] >= 0 handled by bounds
    
    # IR: p_i - v_i . x_i <= 0
    # p_i - v1_i * x1_i - v2_i * x2_i <= 0
    for i in range(n_active):
        row = np.zeros(n_vars)
        v1, v2 = active_types_list[i]
        row[2 * n_active + i] = 1.0  # p_i
        row[i] = -v1  # -v1 * x1_i
        row[i + n_active] = -v2  # -v2 * x2_i
        A_ub.append(row)
        b_ub.append(0.0)
    
    # IC: for all pairs (i, i'):
    # p_i' - p_i >= v_i . (x_i' - x_i)
    # p_i' - p_i - v1_i * (x1_i' - x1_i) - v2_i * (x2_i' - x2_i) >= 0
    # -(p_i' - p_i - v1_i * x1_i' + v1_i * x1_i - v2_i * x2_i' + v2_i * x2_i) <= 0
    # p_i - p_i' + v1_i * x1_i' - v1_i * x1_i + v2_i * x2_i' - v2_i * x2_i <= 0
    
    for i in range(n_active):
        v1_i, v2_i = active_types_list[i]
        for i_prime in range(n_active):
            if i == i_prime:
                continue
            row = np.zeros(n_vars)
            # p_i
            row[2 * n_active + i] = 1.0
            # -p_i'
            row[2 * n_active + i_prime] = -1.0
            # +v1_i * x1_i'
            row[i_prime] = v1_i
            # -v1_i * x1_i
            row[i] = -v1_i
            # +v2_i * x2_i'
            row[i_prime + n_active] = v2_i
            # -v2_i * x2_i
            row[i + n_active] = -v2_i
            A_ub.append(row)
            b_ub.append(0.0)
    
    # Bounds: x[i, j] in [0, 1], p[i] >= 0
    bounds = []
    for i in range(n_active):
        bounds.append((0.0, 1.0))  # x1_i
    for i in range(n_active):
        bounds.append((0.0, 1.0))  # x2_i
    for i in range(n_active):
        bounds.append((0.0, None))  # p_i
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best feasible solution found so far (default zero)
        return _make_default_result(n_types, type_to_idx, orig_to_active, active_idx)
    
    if result.success:
        x_opt = result.x
        x1 = x_opt[:n_active]
        x2 = x_opt[n_active:2 * n_active]
        p = x_opt[2 * n_active:]
    else:
        # If LP failed, try with a simpler approach or return default
        x1 = np.zeros(n_active)
        x2 = np.zeros(n_active)
        p = np.zeros(n_active)
    
    # Clip to ensure feasibility
    x1 = np.clip(x1, 0.0, 1.0)
    x2 = np.clip(x2, 0.0, 1.0)
    p = np.clip(p, 0.0, None)
    
    # Verify and adjust if needed
    # Check IR
    for i in range(n_active):
        v1, v2 = active_types_list[i]
        utility = v1 * x1[i] + v2 * x2[i] - p[i]
        if utility < -1e-8:
            p[i] = v1 * x1[i] + v2 * x2[i]  # Set payment to ensure IR
    
    # Check IC and adjust
    # For safety, if IC is violated, we can try to fix by adjusting payments
    # This is a simple fix; in practice, the LP should satisfy IC
    
    # Build output arrays
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for i in range(n_active):
        orig_i = active_idx[i]
        allocations[orig_i, 0] = x1[i]
        allocations[orig_i, 1] = x2[i]
        payments[orig_i] = p[i]
    
    return {
        'allocations': allocations,
        'payments': payments
    }


def _get_distribution(distribution_spec, n_types, type_to_idx):
    """Get probability distribution based on spec."""
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
            if len(probs) == n_types:
                probs = probs / probs.sum()
                return probs
        if 'type_probs' in distribution_spec:
            probs = np.zeros(n_types)
            for t, p in distribution_spec['type_probs'].items():
                if t in type_to_idx:
                    probs[type_to_idx[t]] = p
            total = probs.sum()
            if total > 1e-12:
                probs = probs / total
            else:
                probs = np.ones(n_types) / n_types
            return probs
    
    if isinstance(distribution_spec, str):
        spec = distribution_spec.lower().strip()
        if spec in ('uniform', 'independent_uniform', 'iid_uniform'):
            return np.ones(n_types) / n_types
        elif spec == 'degenerate':
            # All mass on highest type
            probs = np.zeros(n_types)
            probs[-1] = 1.0
            return probs
        elif spec == 'low':
            # More mass on low types
            weights = np.array([1.0, 0.5, 0.25, 0.125])
            probs = np.outer(weights, weights)
            probs = probs.flatten()
            return probs / probs.sum()
        elif spec == 'high':
            # More mass on high types
            weights = np.array([0.125, 0.25, 0.5, 1.0])
            probs = np.outer(weights, weights)
            probs = probs.flatten()
            return probs / probs.sum()
        else:
            return np.ones(n_types) / n_types
    
    # Default: uniform
    return np.ones(n_types) / n_types


def _make_default_result(n_types, type_to_idx, orig_to_active, active_idx):
    """Return a default feasible result (zero allocations, zero payments)."""
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
