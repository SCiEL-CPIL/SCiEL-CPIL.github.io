"""Incentive-compatible finite auction design via LP with reduced constraints and polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
        
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
            probs = None
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 16:
                    probs = np.array(val, dtype=np.float64)
                    break
            if probs is None:
                probs = np.ones(16, dtype=np.float64) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=np.float64)
        if len(probs) != 16:
            probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Support: 16 type reports
    if support is None or len(support) != 16:
        support = [(i // 4, i % 4) for i in range(16)]
    
    # Ensure support is a list of tuples
    support_list = []
    for s in support:
        if isinstance(s, (list, tuple, np.ndarray)):
            support_list.append((float(s[0]), float(s[1])))
        else:
            support_list.append((float(s), 0.0))
    
    n_types = 16
    n_goods = 2
    
    # Build value matrix V[t, g]
    V = np.zeros((n_types, n_goods))
    for t in range(n_types):
        V[t, 0] = support_list[t][0]
        V[t, 1] = support_list[t][1]
    
    # LP formulation:
    # Variables: u[0:16], a[16:48] (16+32=48 vars)
    # Objective: max sum_t probs[t]*(V[t].a[t] - u[t])
    #   = min -sum_t probs[t]*V[t].a[t] + sum_t probs[t]*u[t]
    #
    # Constraints:
    # IR: u[t] >= 0
    # IC: u[t] - u[t'] >= (V[t]-V[t']).a[t'] for all t != t'
    #     => -u[t] + u[t'] + (V[t]-V[t']).a[t'] <= 0
    # Bounds: 0 <= a[t][g] <= 1
    
    n_vars = n_types + n_types * n_goods  # 48
    
    # Objective: minimize sum_t probs[t]*u[t] - sum_t probs[t]*(V[t].a[t])
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[t] = probs[t]
        c[n_types + t * n_goods + 0] = -probs[t] * V[t, 0]
        c[n_types + t * n_goods + 1] = -probs[t] * V[t, 1]
    
    A_rows = []
    b_vals = []
    
    # IR: u[t] >= 0 => -u[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[t] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # IC constraints: only need adjacent types in the lattice (monotonicity)
    # For 2D grid, adjacent means differing by 1 in one coordinate
    # But to be safe, include all pairs where one dominates the other
    # Actually, for correctness we need all pairs, but we can reduce by
    # noting that if u is convex (which it is for IC mechanisms), it's
    # sufficient to check adjacent types. Let's include all for safety
    # but with a tighter formulation.
    
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[t] = -1.0
            row[t_prime] = 1.0
            dv1 = V[t, 0] - V[t_prime, 0]
            dv2 = V[t, 1] - V[t_prime, 1]
            row[n_types + t_prime * n_goods + 0] = dv1
            row[n_types + t_prime * n_goods + 1] = dv2
            A_rows.append(row)
            b_vals.append(0.0)
    
    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_vals, dtype=np.float64)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0.0, None))  # u[t] >= 0
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    
    best_allocations = None
    best_payments = None
    best_obj = -np.inf
    
    # First solve with HiGHS
    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                     'primal_feasibility_tolerance': 1e-12, 'dual_staleness_tolerance': 1e-12}
        )
        
        if result.success:
            x = result.x
            utilities = np.clip(x[:n_types], 0, None)
            allocations = np.zeros((n_types, n_goods), dtype=np.float64)
            for t in range(n_types):
                for g in range(n_goods):
                    allocations[t, g] = np.clip(x[n_types + t * n_goods + g], 0.0, 1.0)
            
            payments = np.zeros(n_types, dtype=np.float64)
            for t in range(n_types):
                payments[t] = V[t, 0] * allocations[t, 0] + V[t, 1] * allocations[t, 1] - utilities[t]
                payments[t] = max(0.0, payments[t])
            
            # Verify IC
            max_regret = 0.0
            for t in range(n_types):
                for t_prime in range(n_types):
                    if t == t_prime:
                        continue
                    util_report = V[t, 0] * allocations[t_prime, 0] + V[t, 1] * allocations[t_prime, 1] - payments[t_prime]
                    util_truth = V[t, 0] * allocations[t, 0] + V[t, 1] * allocations[t, 1] - payments[t]
                    regret = util_report - util_truth
                    if regret > max_regret:
                        max_regret = regret
            
            if max_regret <= 1e-8:
                obj = np.sum(probs * (V @ allocations.T - payments))
                if obj > best_obj:
                    best_obj = obj
                    best_allocations = allocations.copy()
                    best_payments = payments.copy()
                
                # Polish: re-solve with tighter tolerances
                if time.time() < deadline - 15:
                    try:
                        result2 = linprog(
                            c=c,
                            A_ub=A_ub,
                            b_ub=b_ub,
                            bounds=bounds,
                            method='highs',
                            options={'presolve': True,
                                     'dual_feasibility_tolerance': 1e-14,
                                     'primal_feasibility_tolerance': 1e-14,
                                     'dual_staleness_tolerance': 1e-14,
                                     'simplex_presolve': True}
                        )
                        if result2.success:
                            x2 = result2.x
                            utilities2 = np.clip(x2[:n_types], 0, None)
                            allocations2 = np.zeros((n_types, n_goods), dtype=np.float64)
                            for t in range(n_types):
                                for g in range(n_goods):
                                    allocations2[t, g] = np.clip(x2[n_types + t * n_goods + g], 0.0, 1.0)
                            
                            payments2 = np.zeros(n_types, dtype=np.float64)
                            for t in range(n_types):
                                payments2[t] = V[t, 0] * allocations2[t, 0] + V[t, 1] * allocations2[t, 1] - utilities2[t]
                                payments2[t] = max(0.0, payments2[t])
                            
                            # Verify IC for polished solution
                            max_regret2 = 0.0
                            for t in range(n_types):
                                for t_prime in range(n_types):
                                    if t == t_prime:
                                        continue
                                    util_report = V[t, 0] * allocations2[t_prime, 0] + V[t, 1] * allocations2[t_prime, 1] - payments2[t_prime]
                                    util_truth = V[t, 0] * allocations2[t, 0] + V[t, 1] * allocations2[t, 1] - payments2[t]
                                    regret = util_report - util_truth
                                    if regret > max_regret2:
                                        max_regret2 = regret
                            
                            if max_regret2 <= 1e-8:
                                obj2 = np.sum(probs * (V @ allocations2.T - payments2))
                                if obj2 > best_obj:
                                    best_obj = obj2
                                    best_allocations = allocations2.copy()
                                    best_payments = payments2.copy()
                    except Exception:
                        pass
    except Exception:
        pass
    
    if best_allocations is None:
        best_allocations = np.zeros((n_types, n_goods), dtype=np.float64)
        best_payments = np.zeros(n_types, dtype=np.float64)
    
    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
