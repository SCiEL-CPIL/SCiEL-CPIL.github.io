import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute pair indices for vectorized overlap checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def _is_valid(x, y, r, tol=1e-7):
        # Boundary
        if np.any(x < r - tol) or np.any(1 - x < r - tol) or \
           np.any(y < r - tol) or np.any(1 - y < r - tol):
            return False
        # Pairwise
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(r[i_idx] + r[j_idx] > dist + tol):
            return False
        return True
    
    def _objective(vars):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        
        # Sum of radii (to maximize)
        val = -np.sum(r)
        
        # Penalty for violations (log-sum-exp style soft-max for worst violations)
        # Boundary violations
        bv = np.maximum(0, r - x)
        bv = np.concatenate([bv, np.maximum(0, r - (1 - x))])
        bv = np.concatenate([bv, np.maximum(0, r - y)])
        bv = np.concatenate([bv, np.maximum(0, r - (1 - y))])
        
        # Pairwise violations
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        pv = np.maximum(0, r[i_idx] + r[j_idx] - dist)
        
        # Soft-max penalty
        def soft_max(v, beta=50.0):
            if np.max(v) < 1e-10:
                return 0.0
            v = np.clip(v, 1e-15, None)
            return np.log(np.sum(np.exp(beta * v)) / len(v)) / beta
        
        pen = 1000.0 * (soft_max(bv) + soft_max(pv))
        
        return val + pen
    
    def _post_process(x, y, r):
        # Ensure strict validity by shrinking if needed
        for _ in range(200):
            if _is_valid(x, y, r, tol=1e-9):
                break
            # Shrink radii slightly
            r *= 0.995
            # Clip positions
            r = np.minimum(r, np.minimum(x, 1 - x))
            r = np.minimum(r, np.minimum(y, 1 - y))
        return x, y, r
    
    def _slsqp_polish(x0, y0, r0, max_iter=500):
        # Use SLSQP with explicit constraints for high-precision polish
        def obj(vars):
            x = vars[:N]
            y = vars[N:2*N]
            r = vars[2*N:3*N]
            return -np.sum(r)
        
        # Inequality constraints: g(vars) >= 0
        constraints = []
        for i in range(N):
            # x_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            # 1 - x_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*N+i]})
            # y_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            # 1 - y_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[N+i] - v[2*N+i]})
        
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            # dist(i,j) >= r_i + r_j
            def pair_constraint(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[N+i] - v[N+j]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - v[2*N+i] - v[2*N+j]
            constraints.append({'type': 'ineq', 'fun': pair_constraint})
        
        # Bounds: r_i > 0
        bounds = [(0, 1)] * (3 * N)
        
        x0_vars = np.concatenate([x0, y0, r0])
        
        try:
            res = minimize(obj, x0_vars, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-12})
            if res.success:
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                x, y, r = _post_process(x, y, r)
                return x, y, r
        except Exception:
            pass
        return x0, y0, r0
    
    def _lbfgsb_polish(x0, y0, r0, max_iter=1000):
        # L-BFGS-B with penalty
        x0_vars = np.concatenate([x0, y0, r0])
        bounds = [(0, 1)] * (3 * N)
        try:
            res = minimize(_objective, x0_vars, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            x = res.x[:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            x, y, r = _post_process(x, y, r)
            return x, y, r
        except Exception:
            pass
        return x0, y0, r0
    
    # Multi-restart strategy
    best_score = -np.inf
    best_state = None
    
    def _make_hexagonal():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        side = 5
        indices = 0
        for i in range(side):
            for j in range(side):
                if indices < N:
                    x[indices] = (i + 0.5) / side
                    y[indices] = (j + 0.5) / side
                    r[indices] = 0.5 / side * 0.9
                    indices += 1
        x += np.random.uniform(-0.02, 0.02, N)
        y += np.random.uniform(-0.02, 0.02, N)
        # Fix validity
        for _ in range(50):
            if _is_valid(x, y, r, tol=1e-6):
                break
            r *= 0.95
        return x, y, r
    
    def _make_grid():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        side = 5
        indices = 0
        for i in range(side):
            for j in range(side):
                if indices < N:
                    x[indices] = (i + 0.5) / side
                    y[indices] = (j + 0.5) / side
                    r[indices] = 0.5 / side * 0.85
                    indices += 1
        x += np.random.uniform(-0.01, 0.01, N)
        y += np.random.uniform(-0.01, 0.01, N)
        for _ in range(50):
            if _is_valid(x, y, r, tol=1e-6):
                break
            r *= 0.95
        return x, y, r
    
    def _make_random():
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.06, N)
        for _ in range(100):
            if _is_valid(x, y, r, tol=1e-6):
                break
            r *= 0.9
        return x, y, r
    
    def _make_center_ring():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Center circle
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.12
        # Ring of circles
        n_ring = N - 1
        angle = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
        radius = 0.3
        for i in range(n_ring):
            idx = i + 1
            x[idx] = 0.5 + radius * np.cos(angle[i])
            y[idx] = 0.5 + radius * np.sin(angle[i])
            r[idx] = 0.07
        x += np.random.uniform(-0.01, 0.01, N)
        y += np.random.uniform(-0.01, 0.01, N)
        for _ in range(100):
            if _is_valid(x, y, r, tol=1e-6):
                break
            r *= 0.95
        return x, y, r
    
    generators = [_make_hexagonal, _make_grid, _make_random, _make_center_ring]
    
    for gen in generators:
        if time.time() > deadline - 20.0:
            break
        
        x, y, r = gen()
        score = np.sum(r)
        
        # L-BFGS-B polish
        x2, y2, r2 = _lbfgsb_polish(x, y, r, max_iter=500)
        score2 = np.sum(r2)
        if score2 > score:
            x, y, r = x2, y2, r2
            score = score2
        
        # SLSQP polish for higher precision
        if time.time() < deadline - 10.0:
            x3, y3, r3 = _slsqp_polish(x, y, r, max_iter=200)
            score3 = np.sum(r3)
            if score3 > score:
                x, y, r = x3, y3, r3
                score = score3
        
        if score > best_score:
            best_score = score
            best_state = (x.copy(), y.copy(), r.copy())
    
    # Additional perturbations around best state
    if best_state is not None and time.time() < deadline - 15.0:
        bx, by, br = best_state
        for pert in [0.005, 0.01, 0.02]:
            if time.time() > deadline - 10.0:
                break
            x = bx + np.random.uniform(-pert, pert, N)
            y = by + np.random.uniform(-pert, pert, N)
            r = br.copy()
            x, y, r = _post_process(x, y, r)
            
            x2, y2, r2 = _lbfgsb_polish(x, y, r, max_iter=300)
            score2 = np.sum(r2)
            if score2 > best_score:
                best_score = score2
                best_state = (x2.copy(), y2.copy(), r2.copy())
            
            if time.time() < deadline - 10.0:
                x3, y3, r3 = _slsqp_polish(x2, y2, r2, max_iter=150)
                score3 = np.sum(r3)
                if score3 > best_score:
                    best_score = score3
                    best_state = (x3.copy(), y3.copy(), r3.copy())
    
    if best_state is None:
        # Fallback
        x, y, r = _make_hexagonal()
        best_state = (x, y, r)
    
    bx, by, br = best_state
    bx, by, br = _post_process(bx, by, br)
    
    final_state = np.column_stack([bx, by, br])
    return final_state

# EVOLVE-BLOCK-END
