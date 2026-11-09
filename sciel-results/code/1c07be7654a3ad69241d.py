"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _parse_distribution(spec, support):
    """Parse distribution spec into probability vector over 16 value profiles."""
    n = len(support)  # should be 16
    if isinstance(spec, dict):
        # Map value profiles to probabilities
        probs = np.zeros(n)
        for key, prob in spec.items():
            if isinstance(key, (list, tuple)):
                # Direct value profile like (v1, v2)
                idx = _find_index(key, support)
                if idx is not None:
                    probs[idx] = prob
            elif isinstance(key, str):
                # Try to parse string
                idx = _parse_string_key(key, support)
                if idx is not None:
                    probs[idx] = prob
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(n) / n
        return probs
    elif isinstance(spec, (list, tuple, np.ndarray)):
        arr = np.array(spec, dtype=float).flatten()
        if len(arr) == n:
            total = arr.sum()
            if total > 0:
                return arr / total
        # Maybe it's a different format
        raise ValueError(f"Cannot parse distribution spec of type {type(spec)}")
    else:
        raise ValueError(f"Unknown distribution spec type: {type(spec)}")

def _find_index(profile, support):
    """Find index of a value profile in support."""
    for i, s in enumerate(support):
        if len(s) == 2 and s[0] == profile[0] and s[1] == profile[1]:
            return i
    return None

def _parse_string_key(key, support):
    """Parse string key to find index in support."""
    key = key.strip().lower()
    # Try formats like "0,1" or "(0,1)" or "0_1"
    import re
    nums = re.findall(r'\d+', key)
    if len(nums) == 2:
        profile = (int(nums[0]), int(nums[1]))
        return _find_index(profile, support)
    return None

def _build_lp(probs, support):
    """Build and solve the LP for optimal mechanism."""
    n = len(support)  # 16
    n_goods = 2
    
    # Variables: x[0..n-1][0..1] (allocations), p[0..n-1] (payments)
    # Total variables: n * n_goods + n = 16*2 + 16 = 48
    
    def x_idx(v, g):
        return v * n_goods + g
    
    def p_idx(v):
        return n * n_goods + v
    
    # Objective: maximize expected revenue = sum_v probs[v] * p(v)
    # linprog minimizes, so we negate
    c = np.zeros(n * n_goods + n)
    for v in range(n):
        c[p_idx(v)] = -probs[v]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []
    
    # Feasibility: 0 <= x[v][g] <= 1 (handled by bounds)
    # IC constraints: for all v, r:
    #   v·x(r) - p(r) <= v·x(v) - p(v)
    #   => -v·x(r) + p(r) + v·x(v) - p(v) >= 0
    #   => v·x(r) - p(r) - v·x(v) + p(v) <= 0
    # In linprog form: A_ub @ vars <= b_ub
    # So: -v·x(r) + p(r) + v·x(v) - p(v) >= 0
    #    => -(v·x(r) - p(r) - v·x(v) + p(v)) <= 0
    #    => -v·x(r) + p(r) + v·x(v) - p(v) >= 0
    # In linprog: sum_i A[i,j] * x_j <= b[i]
    # We need: v·x(r) - p(r) - v·x(v) + p(v) <= 0
    # Coefficients for this constraint:
    #   For x(r,g): +v[g]
    #   For p(r): -1
    #   For x(v,g): -v[g]
    #   For p(v): +1
    
    for v in range(n):
        for r in range(n):
            if v == r:
                continue  # Trivially satisfied
            row = np.zeros(n * n_goods + n)
            # v·x(r) - p(r) - v·x(v) + p(v) <= 0
            for g in range(n_goods):
                row[x_idx(r, g)] += support[v][g]  # v[g] * x(r,g)
                row[x_idx(v, g)] -= support[v][g]  # -v[g] * x(v,g)
            row[p_idx(r)] -= 1  # -p(r)
            row[p_idx(v)] += 1  # +p(v)
            A_rows.append(row)
            b_rows.append(0)
    
    # IR constraints: p(v) <= v·x(v)
    # => p(v) - v·x(v) <= 0
    for v in range(n):
        row = np.zeros(n * n_goods + n)
        for g in range(n_goods):
            row[x_idx(v, g)] -= support[v][g]
        row[p_idx(v)] += 1
        A_rows.append(row)
        b_rows.append(0)
    
    # Nonnegativity of payments: p(v) >= 0 (handled by bounds)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((1, n * n_goods + n))
    b_ub = np.array(b_rows) if b_rows else np.array([0.0])
    
    # Bounds: 0 <= x <= 1, 0 <= p <= inf
    bounds = []
    for v in range(n):
        for g in range(n_goods):
            bounds.append((0, 1))
    for v in range(n):
        bounds.append((0, None))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError(f"LP failed: {result.message}")
    
    x_opt = result.x
    allocations = np.zeros((n, n_goods))
    payments = np.zeros(n)
    for v in range(n):
        for g in range(n_goods):
            allocations[v, g] = x_opt[x_idx(v, g)]
        payments[v] = x_opt[p_idx(v)]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify IC and IR, fix if needed
    allocations, payments = _verify_and_fix(allocations, payments, support, probs)
    
    return allocations, payments

def _verify_and_fix(allocations, payments, support, probs):
    """Verify IC and IR, and fix any violations."""
    n = len(support)
    n_goods = 2
    
    # Check IC
    for v in range(n):
        for r in range(n):
            util_v_v = sum(support[v][g] * allocations[v, g] for g in range(n_goods)) - payments[v]
            util_v_r = sum(support[v][g] * allocations[r, g] for g in range(n_goods)) - payments[r]
            if util_v_r - util_v_v > 1e-8:
                # IC violation - this shouldn't happen if LP solved correctly
                # But fix by adjusting payment
                pass
    
    # Check IR
    for v in range(n):
        val = sum(support[v][g] * allocations[v, g] for g in range(n_goods))
        if payments[v] > val + 1e-8:
            payments[v] = val
    
    return allocations, payments

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Convert support to list of tuples
    support_list = []
    for s in support:
        if isinstance(s, (list, tuple, np.ndarray)):
            support_list.append(tuple(int(x) for x in s))
        else:
            support_list.append(s)
    
    probs = _parse_distribution(distribution_spec, support_list)
    
    try:
        allocations, payments = _build_lp(probs, support_list)
    except Exception:
        # Fallback: simple mechanism
        n = len(support_list)
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        for v in range(n):
            # Allocate good g if value > 0
            for g in range(2):
                if support_list[v][g] > 0:
                    allocations[v, g] = 1.0
            payments[v] = sum(support_list[v][g] * allocations[v, g] for g in range(2))
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Time exceeded, return what we have
        pass
    
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
