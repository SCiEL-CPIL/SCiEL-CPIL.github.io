"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Structural change: replaces force-based relaxation with simulated annealing
over a smooth penalty landscape + vectorized L-BFGS-B coordinate polish.
SA explores the non-convex landscape via random perturbations with geometric
cooling; L-BFGS-B then refines with exact gradient information.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 115.0  # Safety margin
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5

def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < s[mask] - tol):
        return False
    return True

def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf

def _clamp(c):
    """Clamp positions/radii to a feasible box."""
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])

def _smooth_max_0(v, T):
    """Smooth approximation of max(0, v) via softplus."""
    a = v / T
    return T * np.where(a > 0, a + np.log1p(np.exp(-a)), np.log1p(np.exp(a)))

def _penalty_objective(c, T, w=500.0):
    """Smooth penalty objective: minimize -sum(r) + w*(wall_viol + overlap_viol)."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    
    wall = (
        _smooth_max_0(r - x, T) + _smooth_max_0(x + r - 1.0, T) +
        _smooth_max_0(r - y, T) + _smooth_max_0(y + r - 1.0, T)
    )
    wall_pen = np.sum(wall)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    overlap = _smooth_max_0(s - d, T)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(overlap[mask])
    
    return -np.sum(r) + w * (wall_pen + overlap_pen)

def _sa_search(c0, T0=0.05, T_end=0.002, n_iter=3000, scale=0.04):
    """Simulated annealing on the smooth penalty landscape.
    
    At each step: propose a random perturbation, accept if penalty decreases
    or with probability exp(-delta/T). Track best feasible solution.
    """
    c = c0.copy()
    T = T0
    cool_rate = (T_end / T0) ** (1.0 / max(n_iter, 1))
    
    # Current penalty value
    cur_pen = _penalty_objective(c, T0)
    cur_valid = _is_valid(c)
    best_c = c.copy() if cur_valid else None
    best_score = _score(c) if cur_valid else -np.inf
    
    for it in range(n_iter):
        if time.time() > DEADLINE - 5.0:
            break
        
        T = T0 * (cool_rate ** it)
        
        # Propose perturbation: move 1-3 random circles
        n_move = RNG.integers(1, 4)
        idx = RNG.choice(N, size=n_move, replace=False)
        
        c_new = c.copy()
        for i in idx:
            c_new[i, 0] += RNG.normal(0, scale)
            c_new[i, 1] += RNG.normal(0, scale)
            c_new[i, 2] += RNG.normal(0, scale * 0.3)
        
        c_new = _clamp(c_new)
        
        # Evaluate
        new_pen = _penalty_objective(c_new, T)
        delta = new_pen - cur_pen
        
        # Accept with Metropolis criterion
        if delta < 0 or RNG.random() < np.exp(-delta / max(T, 1e-8)):
            c = c_new
            cur_pen = new_pen
            cur_valid = _is_valid(c)
            if cur_valid:
                s = _score(c)
                if s > best_score:
                    best_score = s
                    best_c = c.copy()
        else:
            # Rejected: still check if it happened to be valid (unlikely but possible)
            pass
    
    return best_c, best_score

def _coordinate_descent(c0, max_rounds=200):
    """Coordinate descent: optimize one circle at a time using L-BFGS-B.
    
    For each circle i, fix all others and optimize (x_i, y_i, r_i) to
    minimize the penalty. This is a structural change from the global
    force-based approach - it's a sequential refinement method.
    """
    c = c0.copy()
    best_c = c.copy()
    best_val = _score(c)
    T = 0.02
    
    for rnd in range(max_rounds):
        if time.time() > DEADLINE - 8.0:
            break
        
        improved = False
        # Randomize order each round
        order = RNG.permutation(N)
        
        for i in order:
            if time.time() > DEADLINE - 6.0:
                break
            
            # Fix all other circles, optimize circle i
            others = np.delete(c, i, axis=0)
            
            def _obj_i(params):
                x_i, y_i, r_i = params
                r_i = np.clip(r_i, R_MIN, R_MAX)
                x_i = np.clip(x_i, r_i, 1.0 - r_i)
                y_i = np.clip(y_i, r_i, 1.0 - r_i)
                
                # Wall penalty for circle i
                wall = (
                    _smooth_max_0(r_i - x_i, T) + _smooth_max_0(x_i + r_i - 1.0, T) +
                    _smooth_max_0(r_i - y_i, T) + _smooth_max_0(y_i + r_i - 1.0, T)
                )
                wall_pen = np.sum(wall)
                
                # Overlap penalty with all others
                dx = x_i - others[:, 0]
                dy = y_i - others[:, 1]
                d = np.sqrt(dx**2 + dy**2)
                s = r_i + others[:, 2]
                ov = _smooth_max_0(s - d, T)
                overlap_pen = np.sum(ov)
                
                return -r_i + 500.0 * (wall_pen + overlap_pen)
            
            try:
                x0_i = c[i, :].copy()
                res = minimize(_obj_i, x0_i, method='L-BFGS-B',
                              bounds=[(R_MIN, 1.0), (R_MIN, 1.0), (R_MIN, 0.5)],
                              options={'maxiter': 50, 'ftol': 1e-10})
                
                new_i = np.clip(res.x, [R_MIN, R_MIN, R_MIN], [1.0, 1.0, 0.5])
                new_i[0] = np.clip(new_i[0], new_i[2], 1.0 - new_i[2])
                new_i[1] = np.clip(new_i[1], new_i[2], 1.0 - new_i[2])
                
                c[i] = new_i
                improved = True
            except Exception:
                pass
        
        c = _clamp(c)
        val = _score(c)
        if val > best_val:
            best_val = val
            best_c = c.copy()
        
        # Cool temperature
        T *= 0.95
        if T < 0.001:
            break
    
    return best_c, best_val

def _lbfgs_global(c0, T_start=0.03, max_iter=200):
    """Vectorized L-BFGS-B on the full parameter vector with smooth penalty."""
    if time.time() > DEADLINE - 10.0:
        return c0, _score(c0)
    
    x0 = c0.flatten()
    best_c = c0.copy()
    best_val = _score(c0)
    
    for stage in range(3):
        T = T_start * (0.4 ** stage)
        
        def _obj(params):
            c = params.reshape(-1, 3)
            x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
            
            wall = (
                _smooth_max_0(r - x, T) + _smooth_max_0(x + r - 1.0, T) +
                _smooth_max_0(r - y, T) + _smooth_max_0(y + r - 1.0, T)
            )
            wall_pen = np.sum(wall)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            s = r[:, None] + r[None, :]
            overlap = _smooth_max_0(s - d, T)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            overlap_pen = np.sum(overlap[mask])
            
            return -np.sum(r) + 200.0 * (wall_pen + overlap_pen)
        
        def _grad(params):
            # Numerical gradient (central differences)
            eps = 1e-7
            g = np.zeros_like(params)
            for k in range(len(params)):
                p_plus = params.copy(); p_plus[k] += eps
                p_minus = params.copy(); p_minus[k] -= eps
                g[k] = (_obj(p_plus) - _obj(p_minus)) / (2 * eps)
            return g
        
        try:
            result = minimize(
                _obj, x0, jac=_grad,
                method='L-BFGS-B',
                bounds=[(R_MIN, 1.0)] * N + [(R_MIN, 1.0)] * N + [(R_MIN, 0.5)] * N,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_polished = result.x.reshape(-1, 3)
            c_polished = _clamp(c_polished)
            
            if _is_valid(c_polished, tol=1e-6):
                val = _score(c_polished)
                if val > best_val:
                    best_val = val
                    best_c = c_polished.copy()
            
            x0 = c_polished.flatten()
        except Exception:
            pass
        
        if time.time() > DEADLINE - 5.0:
            break
    
    return best_c, best_val

def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _seed_hex():
    """Hexagonal packing initialization."""
    positions = []
    r = 0.18
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(5):
        for col in range(5):
            x = r + col * dx + (row % 2) * dx / 2
            y = r + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y, r))
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r_small = RNG.uniform(0.05, 0.1)
        positions.append((x, y, r_small))
    return np.array(positions[:N])

def _seed_random():
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.03, 0.1, N)
    return np.column_stack([x, y, r])

def _seed_corner_heavy():
    """Initialize with larger circles in corners, smaller in center."""
    positions = []
    radii = []
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy))
        radii.append(0.13)
    for cx, cy in [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]:
        positions.append((cx, cy))
        radii.append(0.1)
    positions.append((0.5, 0.5))
    radii.append(0.12)
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.04, 0.08)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])

def _seed_uniform_large():
    """All circles start at moderate size, spread out."""
    x = RNG.uniform(0.1, 0.9, N)
    y = RNG.uniform(0.1, 0.9, N)
    r = np.full(N, 0.12)
    return np.column_stack([x, y, r])

def construct():
    t0 = time.time()
    
    initializers = [_seed_grid, _seed_hex, _seed_corner_heavy, _seed_random, _seed_uniform_large]
    best_c = None
    best_val = -np.inf
    
    # Phase 1: SA from each initialization (different search family than force relaxation)
    for init_func in initializers:
        if time.time() > DEADLINE - 50.0:
            break
        
        c0 = init_func()
        cc, val = _sa_search(c0, T0=0.06, T_end=0.003, n_iter=2500, scale=0.035)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 2: Additional SA random starts
    for i in range(8):
        if time.time() > DEADLINE - 35.0:
            break
        c0 = _seed_random()
        cc, val = _sa_search(c0, T0=0.05, T_end=0.002, n_iter=1500, scale=0.03)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 3: Coordinate descent refinement (sequential per-circle optimization)
    if best_c is not None and time.time() < DEADLINE - 25.0:
        cc, val = _coordinate_descent(best_c, max_rounds=150)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 4: Vectorized L-BFGS-B global polish
    if best_c is not None and time.time() < DEADLINE - 12.0:
        cc, val = _lbfgs_global(best_c, T_start=0.02, max_iter=150)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 5: Final coordinate descent pass
    if best_c is not None and time.time() < DEADLINE - 8.0:
        cc, val = _coordinate_descent(best_c, max_rounds=50)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Final safety
    if best_c is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_c = np.array(cells[:N])
    
    return best_c
