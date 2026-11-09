"""Rewritten program: efficient LP with pivot-type formulation for IC constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin before 90s kill
    
    # Parse the problem structure
    values = [0, 1, 2, 3]
    
    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"
    
    # type_to_vals mapping
    type_to_vals = {}
    for idx, (v1, v2) in enumerate(types):
        type_to_vals[idx] = (v1, v2)
    
    # Pivot formulation:
    # Choose type 0 = (0,0) as pivot (value 0 for both goods).
    # IR at pivot: p(0) >= 0, and u(0) = 0*x1 + 0*x2 - p(0) = -p(0)
    # Since p(0) >= 0, u(0) <= 0. For IR, u(0) >= 0, so u(0) = 0, p(0) = 0.
    #
    # For any type t, IC relative to pivot:
    # u(t) >= v1(t)*x1(0) + v2(t)*x2(0) - p(0) = v1(t)*x1(0) + v2(t)*x2(0)
    # (since p(0) = 0)
    #
    # So the minimal u(t) is: u(t) = max_{t'} [v1(t)*x1(t') + v2(t)*x2(t') - p(t')]
    # With p(0) = 0 and IC being a set of pairwise constraints, we can use:
    # u(t) >= v1(t)*x1(t') + v2(t)*x2(t') - p(t') for all t'
    #
    # The key simplification: since u(t) = v1(t)*x1(t) + v2(t)*x2(t) - p(t),
    # the IC constraint becomes:
    # v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    # => p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
    #
    # This is a set of 16*15 = 240 linear constraints on (x, p).
    # Plus IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # Plus p(t) >= 0, x in [0,1]
    #
    # Variables: x1[0..15], x2[0..15], p[0..15] = 48 vars
    # Constraints: 240 (IC) + 16 (IR) = 256 rows
    
    # For efficiency, precompute all type pairs
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # Reformulate using envelope condition to eliminate payment variables.
    # For single bidder, IC + IR imply:
    # u(t) = max(0, max_{t' != t} [v1(t)*x1(t') + v2(t)*x2(t') - p(t')])
    # But this is nonlinear. Instead, use the fact that:
    # p(t) - p(t') <= v1(t)*(x1(t)-x1(t')) + v2(t)*(x2(t)-x2(t'))
    # and p(t) <= v1(t)*x1(t) + v2(t)*x2(t), p(t) >= 0
    #
    # Key insight: the minimal feasible payment vector given allocations x is:
    # p_min(t) = max(0, max_{t'} [v1(t)*x1(t') + v2(t)*x2(t') - u_min(t')])
    # This is complex. Instead, keep the LP but reduce constraints.
    #
    # For a 4x4 grid, IC between non-adjacent types is implied by adjacent ones.
    # Define adjacency: types (i,j) and (i',j') are adjacent if |i-i'|+|j-j'|==1
    # This gives ~48 IC constraints instead of 240.
    
    # Map type index to (i,j) coordinates
    type_coords = {}
    for idx, (v1, v2) in enumerate(types):
        type_coords[idx] = (v1, v2)
    
    # Get unique coordinate values
    v1_vals = sorted(set(v1 for v1, v2 in types))
    v2_vals = sorted(set(v2 for v1, v2 in types))
    v1_to_idx = {v: i for i, v in enumerate(v1_vals)}
    v2_to_idx = {v: i for i, v in enumerate(v2_vals)}
    
    # Build adjacency: for each type, find neighbors in the grid
    type_neighbors = {t: [] for t in range(16)}
    for t in range(16):
        i1, j1 = type_coords[t]
        for t2 in range(16):
            if t2 == t:
                continue
            i2, j2 = type_coords[t2]
            if abs(i1 - i2) + abs(j1 - j2) == 1:
                type_neighbors[t].append(t2)
    
    n_vars = 48
    # x1: 0-15, x2: 16-31, p: 32-47
    
    A_rows = []
    b_rows = []
    
    # IC constraints: only adjacent types (reduces from 240 to ~48)
    for t in range(16):
        for t2 in type_neighbors[t]:
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: maximize sum p(t) * probs[t]
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    # Bounds
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve with tighter tolerances
    try:
        options = {'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            result = None
    
    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: try with all IC constraints
        A_rows_full = []
        b_rows_full = []
        for t in range(16):
            for t2 in range(16):
                if t == t2:
                    continue
                row = np.zeros(n_vars)
                row[t] = -v1_all[t]
                row[t2] = v1_all[t]
                row[16 + t] = -v2_all[t]
                row[16 + t2] = v2_all[t]
                row[32 + t] = 1.0
                row[32 + t2] = -1.0
                A_rows_full.append(row)
                b_rows_full.append(0.0)
        for t in range(16):
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[16 + t] = -v2_all[t]
            row[32 + t] = 1.0
            A_rows_full.append(row)
            b_rows_full.append(0.0)
        
        A_ub_full = np.array(A_rows_full)
        b_ub_full = np.array(b_rows_full)
        try:
            result = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs')
        except Exception:
            result = None
        
        if result is not None and result.success:
            x = result.x
            x1 = np.clip(x[0:16], 0, 1)
            x2 = np.clip(x[16:32], 0, 1)
            p = np.maximum(x[32:48], 0)
            allocations = np.column_stack((x1, x2))
            payments = p
        else:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
    
    # Post-verification: check all IC and IR constraints (not just adjacent)
    # Fix IR violations
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util < -1e-8:
            payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    # Check IC for all pairs; if violated, compute minimal payment increase needed
    # and adjust payments to satisfy IC while maintaining IR
    max_violation = 0.0
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(16):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            lhs = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            rhs = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
            if lhs < rhs - 1e-8:
                max_violation = max(max_violation, rhs - lhs)
    
    if max_violation > 1e-8:
        # Reduce payments to fix IC violations
        payments = np.maximum(payments - max_violation, 0.0)
        # Re-check IR
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            if util < -1e-8:
                payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
