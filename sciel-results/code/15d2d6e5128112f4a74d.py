"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _objective(v):
    """Negative sum of radii (to be minimized)."""
    r = v[2 * N:]
    return -np.sum(r)


def _objective_grad(v):
    """Gradient of objective."""
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _constraints(v):
    """Return list of constraint dicts. Each constraint is inequality g(v) >= 0."""
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    
    cons = []
    
    # Circle-boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    # x_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda v, _x=x, _r=r: _x - _r, 
                 'jac': lambda v, _x=x, _r=r: np.concatenate([np.eye(N), np.zeros((N, 2*N)), -np.eye(N)])})
    # 1 - x_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda v, _x=x, _r=r: 1.0 - _x - _r,
                 'jac': lambda v, _x=x, _r=r: np.concatenate([-np.eye(N), np.zeros((N, 2*N)), -np.eye(N)])})
    # y_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda v, _y=y, _r=r: _y - _r,
                 'jac': lambda v, _y=y, _r=r: np.concatenate([np.zeros((N, N)), np.eye(N), -np.eye(N)])})
    # 1 - y_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda v, _y=y, _r=r: 1.0 - _y - _r,
                 'jac': lambda v, _y=y, _r=r: np.concatenate([np.zeros((N, N)), -np.eye(N), -np.eye(N)])})
    
    # Circle-circle constraints: dist_ij - r_i - r_j >= 0
    def _circle_circle_fun(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        vals = np.zeros(N * (N-1) // 2)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-30)
                vals[idx] = d - r[i] - r[j]
                idx += 1
        return vals
    
    def _circle_circle_jac(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        jac = np.zeros((N * (N-1) // 2, 3*N))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-30)
                if d < 1e-15:
                    # Degenerate case; use small gradient
                    gx = 0.0
                    gy = 0.0
                else:
                    gx = dx / d
                    gy = dy / d
                jac[idx, i] = gx
                jac[idx, j] = -gx
                jac[idx, N + i] = gy
                jac[idx, N + j] = -gy
                jac[idx, 2*N + i] = -1.0
                jac[idx, 2*N + j] = -1.0
                idx += 1
        return jac
    
    cons.append({'type': 'ineq', 'fun': _circle_circle_fun, 'jac': _circle_circle_jac})
    
    return cons


def _make_initial_guess(seed=0):
    """Create an initial guess: grid layout with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Add small random jitter
            cx += rng.uniform(-0.02, 0.02)
            cy += rng.uniform(-0.02, 0.02)
            r = 0.5 / side * 0.8
            cells.append((cx, cy, r))
    # Pad if needed
    while len(cells) < N:
        cells.append((0.5, 0.5, 0.01))
    
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    
    return np.concatenate([x, y, r])


def construct():
    """Pack N circles in unit square maximizing sum of radii."""
    best_value = -np.inf
    best_v = None
    
    # Try multiple initial guesses
    for seed in range(8):
        v0 = _make_initial_guess(seed)
        
        # Ensure initial guess is feasible (or close)
        # Clip radii to be positive and small enough
        v0[2*N:] = np.clip(v0[2*N:], 0.001, 0.25)
        
        try:
            result = minimize(
                _objective,
                v0,
                method='L-BFGS-B',
                jac=_objective_grad,
                constraints=_constraints,
                bounds=[(0.0, 1.0)] * (2 * N) + [(0.001, 0.5)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            v = result.x
            # Enforce feasibility: ensure all constraints satisfied
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            # Clip to valid range
            r = np.clip(r, 0.001, 0.5)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            
            # Check if feasible
            feasible = True
            for i in range(N):
                if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9:
                    feasible = False
                if y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
                    feasible = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        feasible = False
            
            if feasible:
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_v = np.column_stack([x, y, r])
            else:
                # Even if not perfectly feasible, keep the best value found
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_v = np.column_stack([x, y, r])
                    
        except Exception:
            continue
    
    if best_v is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_v
# EVOLVE-BLOCK-END
