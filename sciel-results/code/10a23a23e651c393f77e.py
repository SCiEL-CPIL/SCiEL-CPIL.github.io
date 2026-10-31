"""Circle packing in unit square: SLSQP-based constrained optimization with multi-start."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26


def _valid(arr, eps=1e-9):
    """Check that all circles are inside [0,1]^2 and have disjoint interiors."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < -eps) or np.any(y < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True


def _score(arr):
    """Sum of radii, only if valid; else -inf."""
    if not _valid(arr):
        return -np.inf
    return float(np.sum(arr[:, 2]))


def _slsqp_objective(vars_vec):
    """
    vars_vec: length 3*N, ordered [x0,y0,r0, x1,y1,r1, ..., x25,y25,r25]
    Objective: minimize -sum(r_i) i.e. maximize sum of radii
    """
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    return -np.sum(r)


def _slsqp_constraints():
    """
    Build constraint functions for SLSQP.
    Returns list of dicts with 'type': 'ineq' (f(x) >= 0)
    """
    cons = []
    
    # Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(N):
        # x_i >= r_i  =>  x_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.array([1.0, 0.0, -1.0, *[0.0]*((3*i)-3) if i > 0 else [], *[0.0]*(3*(N-1-i)) if i < N-1 else []])
        })
        # 1 - x_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.array([*[0.0]*(3*i), -1.0, 0.0, -1.0, *[0.0]*(3*(N-1-i))])
        })
        # y_i >= r_i
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.array([*[0.0]*(3*i), 0.0, 1.0, -1.0, *[0.0]*(3*(N-1-i))])
        })
        # 1 - y_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.array([*[0.0]*(3*i), 0.0, -1.0, -1.0, *[0.0]*(3*(N-1-i))])
        })
        # r_i >= 0.001 (small positive)
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+2] - 0.001,
            'jac': lambda v, i=i: np.array([*[0.0]*(3*i), 0.0, 0.0, 1.0, *[0.0]*(3*(N-1-i))])
        })
    
    # Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2,
                'jac': lambda v, i=i, j=j: _nonoverlap_jac(v, i, j)
            })
    
    return cons


def _nonoverlap_jac(v, i, j):
    """Jacobian for non-overlap constraint between circles i and j."""
    jac = np.zeros(3*N)
    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
    dx = xi - xj
    dy = yi - yj
    # d/dx_i: 2*dx
    jac[3*i] = 2*dx
    # d/dy_i: 2*dy
    jac[3*i+1] = 2*dy
    # d/dr_i: -2*(ri+rj)
    jac[3*i+2] = -2*(ri+rj)
    # d/dx_j: -2*dx
    jac[3*j] = -2*dx
    # d/dy_j: -2*dy
    jac[3*j+1] = -2*dy
    # d/dr_j: -2*(ri+rj)
    jac[3*j+2] = -2*(ri+rj)
    return jac


def _slsqp_bounds():
    """Box bounds for SLSQP."""
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))   # x_i
        bounds.append((0.0, 1.0))   # y_i
        bounds.append((0.001, 0.5))  # r_i
    return bounds


def _slsqp_start(rng, init_radius=0.03):
    """Generate a random feasible-ish starting point."""
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    r = np.full(N, init_radius)
    
    # Simple de-overlap: push apart overlapping circles
    for _ in range(50):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    dx, dy, d = 1e-8, 0.0, 1e-8
                need = r[i] + r[j]
                if d < need:
                    push = (need - d) / 2.0 + 1e-6
                    ux, uy = dx/d, dy/d
                    x[i] += push * ux
                    y[i] += push * uy
                    x[j] -= push * ux
                    y[j] -= push * uy
                    moved = True
        # Clamp to bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])
        if not moved:
            break
    
    vars_vec = np.zeros(3*N)
    vars_vec[0::3] = x
    vars_vec[1::3] = y
    vars_vec[2::3] = r
    return vars_vec


def construct():
    best_arr = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    n_restarts = 20
    constraints = _slsqp_constraints()
    bounds = _slsqp_bounds()
    
    for restart in range(n_restarts):
        x0 = _slsqp_start(rng, init_radius=0.02 + 0.01 * (restart % 5))
        
        try:
            res = minimize(
                _slsqp_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < 0:  # fun < 0 means positive sum of radii
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Clamp to be safe
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
                r = np.clip(r, 0.001, 0.5)
                
                arr = np.column_stack([x, y, r])
                
                # Verify and score
                sc = _score(arr)
                if sc > best_score:
                    best_score = sc
                    best_arr = arr.copy()
                    
        except Exception:
            continue
    
    if best_arr is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr
