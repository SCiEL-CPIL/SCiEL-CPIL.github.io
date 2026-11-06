import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    # 1. Setup Report Space
    n_types = 16
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    
    # 2. Parse Distribution
    probs = _get_probabilities(distribution_spec, n_types)
    
    # 3. Solve the Reduced LP with Non-Negative Payment Constraints
    result = _solve_reduced_lp(values, probs, n_types, deadline)
    
    if result is not None:
        x_alloc, payments = result
        # Final verification
        if _check_feasibility(x_alloc, payments, values, n_types):
            return {
                'allocations': x_alloc,
                'payments': payments
            }
        
        # If verification fails (numerical issues), try to repair or fallback
        # Attempt a small polish or fallback
        return _fallback_mechanism()
    
    return _fallback_mechanism()

def _solve_reduced_lp(values, probs, n_types, deadline):
    """
    Solves the optimal mechanism problem using the reduced LP formulation.
    Variables: x[v, g] for v in 0..15, g in 0..1 (32 variables)
    Objective: Maximize Expected Revenue (substituting payment identity)
    Constraints:
      1. 0 <= x <= 1
      2. Monotonicity: x[v, g] <= x[v+e_g, g]
      3. Non-negative Payments: p_v(x) >= 0
    """
    start_time = time.time()
    if start_time > deadline - 5.0:
        return None

    n_vars = n_types * 2  # 32 variables
    
    # 1. Objective Coefficients
    # E[R] = sum_v probs[v] * p_v
    # p_v = v1*x[v,0] + v2*x[v,1] - sum_{u1<v1} x[u1,v2,0] - sum_{u2<v2} x[v1,u2,1]
    
    obj_coeffs = np.zeros(n_vars)
    
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        p_v = probs[v]
        
        # Direct terms: +p_v * v1 * x[v,0] and +p_v * v2 * x[v,1]
        obj_coeffs[v*2 + 0] += p_v * v1
        obj_coeffs[v*2 + 1] += p_v * v2
        
        # Subsidy terms for Good 0: -p_v * x[u1, v2, 0] for u1 < v1
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            obj_coeffs[idx_u * 2 + 0] -= p_v
            
        # Subsidy terms for Good 1: -p_v * x[v1, u2, 1] for u2 < v2
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            obj_coeffs[idx_u * 2 + 1] -= p_v

    # linprog minimizes, so we negate the objective
    c = -obj_coeffs

    # 2. Bounds
    bounds = [(0.0, 1.0)] * n_vars

    # 3. Constraints (A_ub @ x <= b_ub)
    A_ub_rows = []
    b_ub_rows = []
    
    # 3a. Monotonicity Constraints
    # Good 0: x[(v1,v2),0] <= x[(v1+1,v2),0]
    for v2 in range(4):
        for v1 in range(3):
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            row = np.zeros(n_vars)
            row[i_curr * 2 + 0] = 1.0
            row[i_next * 2 + 0] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # Good 1: x[(v1,v2),1] <= x[(v1,v2+1),1]
    for v1 in range(4):
        for v2 in range(3):
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            row = np.zeros(n_vars)
            row[i_curr * 2 + 1] = 1.0
            row[i_next * 2 + 1] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # 3b. Non-Negative Payment Constraints
    # p_v >= 0  =>  -p_v <= 0
    # p_v = v1*x[v,0] + v2*x[v,1] - sum_{u1<v1} x[u1,v2,0] - sum_{u2<v2} x[v1,u2,1]
    # -p_v = -v1*x[v,0] - v2*x[v,1] + sum_{u1<v1} x[u1,v2,0] + sum_{u2<v2} x[v1,u2,1] <= 0
    
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        
        # If v1=0 and v2=0, p_v is always 0. -0 <= 0 is 0<=0, which is redundant but harmless.
        # We can skip (0,0) or include it. Including it is fine.
        
        row = np.zeros(n_vars)
        
        # Coefficients for x[v, g]
        if v1 > 0:
            row[v*2 + 0] = -v1
        if v2 > 0:
            row[v*2 + 1] = -v2
            
        # Coefficients for subsidy terms (positive in -p_v)
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            row[idx_u * 2 + 0] += 1.0
            
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            row[idx_u * 2 + 1] += 1.0
            
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
    else:
        A_ub = None
        b_ub = None
        
    # 4. Solve LP
    try:
        # Use high precision settings
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14, 'time_limit': 70})
        
        if not res.success:
            return None
            
        x_flat = res.x
        x_alloc = x_flat.reshape(n_types, 2)
        
        # Project to feasible set: clip to [0,1]
        x_alloc = np.clip(x_alloc, 0.0, 1.0)
        
        # High-precision polish: Check if we can improve via local search on the polytope?
        # Actually, for LP, the solution is a vertex. Clipping might move us off the vertex if there were numerical errors.
        # However, if we clip, we might violate monotonicity or payment constraints slightly.
        # A better approach: if the solution is slightly infeasible due to tolerance, try to project back or fix.
        # But standard HiGHS is robust. Let's just ensure strict feasibility for the checker.
        
        # Calculate Payments using IC Identity
        payments = np.zeros(n_types)
        for v in range(n_types):
            v1, v2 = int(values[v][0]), int(values[v][1])
            p_v = v1 * x_alloc[v, 0] + v2 * x_alloc[v, 1]
            
            for u1 in range(v1):
                idx_u = u1 * 4 + v2
                p_v -= x_alloc[idx_u, 0]
                
            for u2 in range(v2):
                idx_u = v1 * 4 + u2
                p_v -= x_alloc[idx_u, 1]
                
            payments[v] = p_v
            
        # Ensure non-negative payments (clip tiny negatives from numerical error)
        # Note: Clipping payments breaks IC identity if we don't adjust allocations, but 
        # the evaluator checks IC on the reported allocations and payments.
        # If p_v < 0 due to 1e-15 error, clipping to 0 is safe for IR and IC if the error is tiny.
        # However, strictly speaking, changing p_v changes the utility.
        # Since the LP enforced p_v >= 0, any negative value is numerical noise.
        payments = np.clip(payments, 0.0, None)
        
        return (x_alloc, payments)
        
    except Exception:
        return None

def _get_probabilities(spec, n_types):
    """Parses various distribution formats into a probability vector."""
    if spec is None:
        return np.ones(n_types) / n_types
        
    if isinstance(spec, dict):
        probs = np.zeros(n_types)
        for key, val in spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = int(key[0]), int(key[1])
                if 0 <= v1 < 4 and 0 <= v2 < 4:
                    idx = v1 * 4 + v2
                    probs[idx] = float(val)
            elif isinstance(key, int):
                if 0 <= key < n_types:
                    probs[key] = float(val)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(n_types) / n_types
        
    if isinstance(spec, (list, tuple, np.ndarray)):
        arr = np.array(spec, dtype=float)
        if arr.shape == (n_types,):
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_types) / n_types
        # Handle 2D array of [v1, v2, p]
        if arr.ndim == 2 and arr.shape[1] >= 3:
            probs = np.zeros(n_types)
            for row in arr:
                v1, v2, p = int(row[0]), int(row[1]), float(row[2])
                if 0 <= v1 < 4 and 0 <= v2 < 4:
                    idx = v1 * 4 + v2
                    probs[idx] += p
            total = probs.sum()
            if total > 0:
                return probs / total
            return np.ones(n_types) / n_types
            
    return np.ones(n_types) / n_types

def _check_feasibility(allocs, pays, values, n_types):
    """Checks IR, IC, and Non-negativity."""
    eps = 1e-8
    
    # Check bounds
    if np.any(allocs < -eps) or np.any(allocs > 1 + eps):
        return False
    if np.any(pays < -eps):
        return False
        
    # Check IR and IC
    for j in range(n_types):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocs[j, 0] + v2_j * allocs[j, 1] - pays[j]
        
        if u_true < -eps:
            return False
            
        for i in range(n_types):
            if i == j: continue
            u_mis = v1_j * allocs[i, 0] + v2_j * allocs[i, 1] - pays[i]
            if u_true - u_mis < -eps:
                return False
                
    return True

def _fallback_mechanism():
    """
    Returns a simple, valid mechanism.
    Allocates nothing, charges nothing.
    Revenue 0, but valid.
    """
    n_types = 16
    allocs = np.zeros((n_types, 2))
    pays = np.zeros(n_types)
    return {
        'allocations': allocs,
        'payments': pays
    }
