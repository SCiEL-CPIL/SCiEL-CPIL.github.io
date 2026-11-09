import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def _get_divisors(n):
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def _greedy_construct(base=51975):
    covered = bytearray(base)
    classes = []
    for m in [d for d in range(3, base + 1, 2) if base % d == 0]:
        best = max(range(m), key=lambda r: covered[r::m].count(0))
        classes.append([m, best])
        covered[best::m] = b'\x01' * len(range(best, base, m))
    return classes

def _local_search(classes, base=51975, time_limit=10):
    start_time = time.time()
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    best_score = _score_classes(classes, base)
    best_classes = [c[:] for c in classes]
    
    # Simple local search: try changing residues
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i, (m, r) in enumerate(best_classes):
            if time.time() - start_time > time_limit:
                break
            for new_r in range(m):
                if new_r == r:
                    continue
                new_classes = [c[:] for c in best_classes]
                new_classes[i] = [m, new_r]
                s = _score_classes(new_classes, base)
                if s < best_score:
                    best_score = s
                    best_classes = new_classes
                    r = new_r
                    improved = True
    return best_classes

def _score_classes(classes, base=51975):
    covered = bytearray(base)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, base, m))
    return base - sum(covered)

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140
    
    divisors = _get_divisors(base)
    num_moduli = len(divisors)
    
    # Prepare MILP
    y_offsets = {}
    y_size = 0
    for m in divisors:
        y_offsets[m] = y_size
        y_size += m
        
    z_size = base
    total_vars = y_size + z_size
    
    c = np.zeros(total_vars)
    c[y_size:] = -1
    
    rows = []
    cols = []
    vals = []
    
    # Constraint 1: Moduli selection
    for i, m in enumerate(divisors):
        offset = y_offsets[m]
        for r in range(m):
            rows.append(i)
            cols.append(offset + r)
            vals.append(1)
            
    # Constraint 2: Coverage
    for x in range(base):
        row_idx = num_moduli + x
        rows.append(row_idx)
        cols.append(y_size + x)
        vals.append(1)
        
        for m in divisors:
            r = x % m
            col_idx = y_offsets[m] + r
            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(-1)
            
    A = sp.csr_matrix((vals, (rows, cols)), shape=(num_moduli + base, total_vars))
    
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    integrality = np.ones(total_vars)
    
    cons_lb = np.full(num_moduli + base, -np.inf)
    cons_ub = np.zeros(num_moduli + base)
    cons_ub[:num_moduli] = 1
    
    constraints = LinearConstraint(A, cons_lb, cons_ub)
    
    # Time limit for solver
    solver_time_limit = max(1, int(deadline - time.time()) - 10)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": solver_time_limit})
        
        if res.x is not None:
            y_sol = res.x[:y_size]
            classes = []
            for m in divisors:
                offset = y_offsets[m]
                r_vals = y_sol[offset:offset+m]
                indices = np.where(r_vals > 0.5)[0]
                if len(indices) > 0:
                    classes.append([m, int(indices[0])])
            
            # Polish with local search if time remains
            if time.time() < deadline - 5:
                classes = _local_search(classes, base, time_limit=min(10, int(deadline - time.time()) - 2))
            
            return classes
    except Exception:
        pass
        
    # Fallback
    classes = _greedy_construct(base)
    if time.time() < deadline - 5:
        classes = _local_search(classes, base, time_limit=min(10, int(deadline - time.time()) - 2))
    return classes
