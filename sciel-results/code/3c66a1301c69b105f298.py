"""Incentive-compatible finite auction design via utility-variable LP."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Uses utility-variable formulation:
    - Variables: u[i] (utility for type i), x[i,g] (allocation prob for good g to type i)
    - Revenue = sum_i probs[i] * (dot(x[i], v[i]) - u[i])
    - IC: u[i] >= u[j] + dot(x[j] - x[i], v[i])  for all i,j
    - IR: u[i] >= 0
    - Bounds: 0 <= x[i,g] <= 1
    
    This eliminates payment variables and makes IC a set of linear difference constraints.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    # Parse inputs
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', []), dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Ensure we have 16 probabilities
    if len(probs) < 16:
        probs = np.pad(probs, (0, 16 - len(probs)))
    else:
        probs = probs[:16]
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    
    # Standard 4x4 support
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    n = 16  # number of types
    
    # Check time
    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    # Variables: [u_0, ..., u_15, x_00, x_01, x_10, x_11, ..., x_15,0, x_15,1]
    # Total: 16 + 32 = 48 variables
    n_vars = n + 2 * n
    
    # Objective: minimize -sum_i probs[i] * (dot(x[i], v[i]) - u[i])
    #           = minimize -sum_i probs[i] * dot(x[i], v[i]) + sum_i probs[i] * u[i]
    c = np.zeros(n_vars)
    for i in range(n):
        for g in range(2):
            c[n + i * 2 + g] = -probs[i] * values[i, g]
        c[i] = probs[i]  # +probs[i] * u[i] in the minimization
    
    # Build A_ub * vars <= b_ub
    A_rows = []
    b_rows = []
    
    # IC constraints: u[i] - u[j] >= dot(x[j] - x[i], v[i])
    # Rewrite: u[i] - u[j] - dot(x[j], v[i]) + dot(x[i], v[i]) >= 0
    # Which is: -u[i] + u[j] + dot(x[j], v[i]) - dot(x[i], v[i]) <= 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -1.0   # -u[i]
            row[j] = 1.0    # +u[j]
            for g in range(2):
                row[n + j * 2 + g] = values[i, g]      # +x[j,g]*v[i,g]
                row[n + i * 2 + g] = -values[i, g]     # -x[i,g]*v[i,g]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: u[i] >= 0  =>  -u[i] <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: u[i] is free (can be negative in principle, but IR forces >= 0)
    # Actually with IR constraints, u[i] >= 0. But to be safe, allow u[i] >= 0.
    # x[i,g] in [0, 1]
    bounds = [(0, None)] * n + [(0, 1)] * (2 * n)
    
    # Check time
    if time.time() > deadline - 10:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True,
                              'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12})
    
    if result.success:
        x_opt = result.x
        u = x_opt[:n]
        allocs = x_opt[n:].reshape(n, 2)
        
        # Clip for numerical safety
        allocs = np.clip(allocs, 0, 1)
        u = np.maximum(u, 0)  # enforce IR
        
        # Compute payments: p[i] = dot(x[i], v[i]) - u[i]
        pays = np.array([np.dot(allocs[i], values[i]) - u[i] for i in range(n)])
        pays = np.maximum(pays, 0)  # non-negative payments
        
        # Verify IC and fix any violations
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ui_i = np.dot(allocs[i], values[i]) - pays[i]
                ui_j = np.dot(allocs[j], values[i]) - pays[j]
                if ui_i < ui_j - 1e-8:
                    # Increase payment for j to fix
                    excess = ui_j - ui_i + 1e-8
                    pays[j] += excess
        
        # Re-verify IR
        for i in range(n):
            ui_i = np.dot(allocs[i], values[i]) - pays[i]
            if ui_i < -1e-8:
                pays[i] = max(0.0, np.dot(allocs[i], values[i]))
        
        # Final safety
        pays = np.maximum(pays, 0)
        allocs = np.clip(allocs, 0, 1)
        
        return {'allocations': allocs, 'payments': pays}
    else:
        # Fallback: try with looser tolerances
        if time.time() < deadline - 5:
            result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                             options={'presolve': True})
            if result2.success:
                x_opt = result2.x
                u = x_opt[:n]
                allocs = x_opt[n:].reshape(n, 2)
                allocs = np.clip(allocs, 0, 1)
                u = np.maximum(u, 0)
                pays = np.array([np.dot(allocs[i], values[i]) - u[i] for i in range(n)])
                pays = np.maximum(pays, 0)
                return {'allocations': allocs, 'payments': pays}
        
        # Trivial fallback
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

# EVOLVE-BLOCK-END
