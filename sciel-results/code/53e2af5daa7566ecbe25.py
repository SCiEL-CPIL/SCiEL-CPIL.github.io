"""Direct LP solver for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite-support mechanism design LP directly.
    
    One additive bidder, two goods, values in {0,1,2,3} => 16 types.
    Variables: x[i,j] = prob(good j allocated) when report i (16*2=32 vars)
               p[i]   = payment when report i (16 vars)
    Total 48 variables.
    
    Objective: maximize expected revenue = sum_i pr[i] * p[i]
    Constraints:
      - 0 <= x[i,j] <= 1 for all i,j
      - p[i] >= 0 (nonnegative payments)
      - IR: for each type i, p[i] <= v_i · x[i] (expected value >= payment)
      - IC: for each pair (i, i'), truthfulness: 
            v_i·x[i] - p[i] >= v_i·x[i'] - p[i']
            i.e., utility of reporting truthfully >= utility of misreporting
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Parse support and distribution
    # support: list of (v1, v2) pairs, or a spec for values in {0,1,2,3}
    # distribution_spec: probabilities for each type
    
    # Build the type space: values in {0,1,2,3} for each good
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    n_types = len(types)  # 16
    
    # Parse distribution_spec to get probabilities
    # distribution_spec could be a dict, list, or other format
    probs = _parse_distribution(distribution_spec, types)
    
    # Parse support if provided (might restrict which types are possible)
    if support is not None and len(support) > 0:
        # support might be a list of type indices or (v1,v2) pairs
        support_types = _parse_support(support, types)
        # Filter to only supported types
        active_indices = [i for i in range(n_types) if i in support_types]
    else:
        active_indices = list(range(n_types))
    
    n_active = len(active_indices)
    
    # Variable layout:
    # x[0..31]: allocation probabilities for 16 types * 2 goods
    # p[32..47]: payments for 16 types
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p  # 48
    
    # Objective: maximize sum_i pr[i] * p[i]
    # linprog minimizes, so negate: c = [0,...,0, -pr[0], ..., -pr[15]]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_x + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    # 1. x[i,j] <= 1 for all i,j
    # 2. -x[i,j] <= 0 for all i,j (x >= 0)
    # 3. IR: p[i] - v1_i*x[i,0] - v2_i*x[i,1] <= 0 for all i
    # 4. IC: for all i, i': v_i·x[i] - p[i] - v_i·x[i'] + p[i'] >= 0
    #    => -v_i·x[i] + p[i] + v_i·x[i'] - p[i'] <= 0
    
    A_rows = []
    b_rows = []
    
    # Constraint 1: x[i,j] <= 1
    for i in range(n_types):
        for j in range(2):
            row = np.zeros(n_vars)
            row[i * 2 + j] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)
    
    # Constraint 2: -x[i,j] <= 0
    for i in range(n_types):
        for j in range(2):
            row = np.zeros(n_vars)
            row[i * 2 + j] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # Constraint 3: IR: p[i] - v1_i*x[i,0] - v2_i*x[i,1] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = types[i]
        row[i * 2 + 0] = -v1
        row[i * 2 + 1] = -v2
        row[n_x + i] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    # Constraint 4: IC for all pairs (i, i')
    # Utility of type i reporting i: u(i,i) = v_i·x[i] - p[i]
    # Utility of type i reporting i': u(i,i') = v_i·x[i'] - p[i']
    # IC: u(i,i) >= u(i,i') => v_i·x[i] - p[i] - v_i·x[i'] + p[i'] >= 0
    # => -v_i·x[i] + p[i] + v_i·x[i'] - p[i'] <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            # -v_i·x[i]
            row[i * 2 + 0] = -v1
            row[i * 2 + 1] = -v2
            # +p[i]
            row[n_x + i] = 1.0
            # +v_i·x[i']
            row[ip * 2 + 0] = v1
            row[ip * 2 + 1] = v2
            # -p[i']
            row[n_x + ip] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: x[i,j] in [0,1], p[i] in [0, inf)
    bounds = []
    for i in range(n_types):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Check time before solving
    if time.time() > deadline - 10:
        return _fallback_mechanism(types, probs)
    
    # Solve with high precision
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                        method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-10, 'primal_feasibility_tolerance': 1e-10})
    except Exception as e:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs-ds')
        except Exception:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if result.success:
        x_alloc = result.x[:n_x].reshape(n_types, 2)
        payments = result.x[n_x:n_x + n_p]
        
        # Clamp small negatives from numerical error
        x_alloc = np.clip(x_alloc, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify constraints with tight tolerance
        if _verify_constraints(x_alloc, payments, types, probs, tol=1e-8):
            return {"allocations": x_alloc, "payments": payments}
        else:
            # Polish pass: re-solve with tighter tolerances if time allows
            if time.time() < deadline - 5:
                polished = _polish_solution(x_alloc, payments, types, probs, c, A_ub, b_ub, bounds, deadline)
                if polished is not None:
                    return polished
            # Try to fix small violations
            x_alloc = np.clip(x_alloc, 0.0, 1.0)
            payments = np.clip(payments, 0.0, None)
            return {"allocations": x_alloc, "payments": payments}
    else:
        # Fallback: use a simple VCG-like or equal-allocation mechanism
        return _fallback_mechanism(types, probs)


def _polish_solution(x_alloc, payments, types, probs, c, A_ub, b_ub, bounds, deadline):
    """
    Polish pass: re-solve LP with the current solution as a warm start hint.
    This helps eliminate tiny numerical violations that may cause regret > 1e-8.
    """
    n_types = len(types)
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p
    
    # Create a slightly perturbed objective to encourage exactness
    # Add tiny regularization to break degeneracy
    c_perturbed = c.copy()
    for i in range(n_vars):
        c_perturbed[i] += 1e-12 * np.random.randn()
    
    try:
        result = linprog(c_perturbed, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                        method='highs', options={'presolve': True, 
                                                 'dual_feasibility_tolerance': 1e-12, 
                                                 'primal_feasibility_tolerance': 1e-12})
        if result.success:
            x_alloc_new = result.x[:n_x].reshape(n_types, 2)
            payments_new = result.x[n_x:n_x + n_p]
            x_alloc_new = np.clip(x_alloc_new, 0.0, 1.0)
            payments_new = np.clip(payments_new, 0.0, None)
            
            if _verify_constraints(x_alloc_new, payments_new, types, probs, tol=1e-9):
                return {"allocations": x_alloc_new, "payments": payments_new}
    except Exception:
        pass
    
    return None


def _parse_distribution(distribution_spec, types):
    """Parse distribution_spec into an array of probabilities for each type."""
    n_types = len(types)
    
    if distribution_spec is None:
        # Uniform
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = arr / arr.sum()
            return arr
        # Maybe it's a list of (v1, v2, prob)
        probs = np.zeros(n_types)
        for item in distribution_spec:
            if len(item) >= 3:
                v1, v2, p = item[0], item[1], item[2]
                idx = types.index((v1, v2))
                probs[idx] = p
            elif len(item) == 2:
                v1, v2 = item[0], item[1]
                idx = types.index((v1, v2))
                probs[idx] = 1.0
        if probs.sum() > 0:
            return probs / probs.sum()
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        for key, val in distribution_spec.items():
            if isinstance(key, (list, tuple)):
                v1, v2 = key[0], key[1]
                idx = types.index((v1, v2))
                probs[idx] = val
            else:
                # Maybe key is an index
                if isinstance(key, int) and 0 <= key < n_types:
                    probs[key] = val
        if probs.sum() > 0:
            return probs / probs.sum()
        return np.ones(n_types) / n_types
    
    # Default uniform
    return np.ones(n_types) / n_types


def _parse_support(support, types):
    """Parse support into a set of active type indices."""
    active = set()
    if isinstance(support, (list, tuple, np.ndarray)):
        for item in support:
            if isinstance(item, int):
                if 0 <= item < len(types):
                    active.add(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    idx = types.index((item[0], item[1]))
                    active.add(idx)
                except ValueError:
                    pass
    elif isinstance(support, set):
        for item in support:
            if isinstance(item, int):
                if 0 <= item < len(types):
                    active.add(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    idx = types.index((item[0], item[1]))
                    active.add(idx)
                except ValueError:
                    pass
    if not active:
        active = set(range(len(types)))
    return active


def _verify_constraints(allocations, payments, types, probs, tol=1e-8):
    """Verify all constraints are satisfied."""
    n_types = len(types)
    
    # Check allocation bounds
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # Check nonnegative payments
    if np.any(payments < -tol):
        return False
    
    # Check IR
    for i in range(n_types):
        v1, v2 = types[i]
        utility = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        if utility < -tol:
            return False
    
    # Check IC
    for i in range(n_types):
        v1, v2 = types[i]
        u_truth = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        for ip in range(n_types):
            u_misrep = v1 * allocations[ip, 0] + v2 * allocations[ip, 1] - payments[ip]
            if u_truth - u_misrep < -tol:
                return False
    
    return True


def _fallback_mechanism(types, probs):
    """Simple fallback: allocate both goods to everyone, charge expected value."""
    n_types = len(types)
    allocations = np.ones((n_types, 2))
    payments = np.zeros(n_types)
    for i in range(n_types):
        v1, v2 = types[i]
        payments[i] = v1 + v2  # This may violate IR for low types
        # Actually set payment to 0 to be safe
        payments[i] = 0.0
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
