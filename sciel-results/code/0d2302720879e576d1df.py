import time
import sys
import numpy as np

_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Target differences
    target_diffs = list(range(1, 31))
    
    # We search for the minimum D
    # Lower bound: 30
    # Upper bound: Let's try up to 200.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        solver_available = True
    except ImportError:
        solver_available = False

    if solver_available:
        for D in range(30, 200):
            if time.time() - _START > 110:
                break
                
            # Variables:
            # x[0..D] : binary
            # z[i, d] : binary for i in 0..D-d, d in 1..D
            n_x = D + 1
            z_indices = {}
            z_count = 0
            for d in range(1, D + 1):
                for i in range(0, D - d + 1):
                    z_indices[(i, d)] = z_count
                    z_count += 1
            
            n_vars = n_x + z_count
            
            c = np.zeros(n_vars)
            bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
            
            rows = []
            cols = []
            vals = []
            b_lbs = []
            b_ubs = []
            
            # 1. x[0] = 1
            rows.append(0)
            cols.append(0)
            vals.append(1)
            b_lbs.append(1)
            b_ubs.append(1)
            
            # 2 & 3. Difference constraints
            for d in range(1, D + 1):
                z_idxs = [z_indices[(i, d)] + n_x for i in range(0, D - d + 1)]
                row_idx = len(b_lbs)
                
                for idx in z_idxs:
                    rows.append(row_idx)
                    cols.append(idx)
                    vals.append(1)
                    
                if d <= 30:
                    b_lbs.append(1)
                    b_ubs.append(1)
                else:
                    b_lbs.append(0)
                    b_ubs.append(1)
            
            # 4. Linking constraints
            for (i, d), z_idx in z_indices.items():
                z_flat = z_idx + n_x
                x_i_flat = i
                x_id_flat = i + d
                
                # z <= x_i
                row_idx = len(b_lbs)
                rows.append(row_idx)
                cols.append(z_flat)
                vals.append(1)
                rows.append(row_idx)
                cols.append(x_i_flat)
                vals.append(-1)
                b_lbs.append(0)
                b_ubs.append(0)
                
                # z <= x_{i+d}
                row_idx = len(b_lbs)
                rows.append(row_idx)
                cols.append(z_flat)
                vals.append(1)
                rows.append(row_idx)
                cols.append(x_id_flat)
                vals.append(-1)
                b_lbs.append(0)
                b_ubs.append(0)
                
                # z >= x_i + x_{i+d} - 1
                row_idx = len(b_lbs)
                rows.append(row_idx)
                cols.append(z_flat)
                vals.append(1)
                rows.append(row_idx)
                cols.append(x_i_flat)
                vals.append(-1)
                rows.append(row_idx)
                cols.append(x_id_flat)
                vals.append(-1)
                b_lbs.append(0)
                b_ubs.append(1)
                
            A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b_lbs), n_vars))
            cons = LinearConstraint(A, lb=b_lbs, ub=b_ubs)
            
            try:
                res = milp(c, constraints=cons, bounds=bounds, integrality=np.ones(n_vars))
                if res.status == 0:
                    x_sol = res.x[:n_x]
                    elements = [i for i in range(D + 1) if x_sol[i] > 0.5]
                    if 0 in elements:
                        return elements
            except Exception:
                pass
                
    # Fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
