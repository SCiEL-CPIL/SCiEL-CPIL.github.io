# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = []
    for i in range(6):
        p1 = V[i]
        p2 = V[(i + 1) % 6]
        e = p2 - p1
        L = np.hypot(e[0], e[1])
        n.append(np.array([e[1], -e[0]]) / L)
    return np.array(n)

def _gap(A, B):
    """Return best separation margin between two hexagons A and B (6,2) vertices.
    Larger = more separated. Valid if > 1e-6."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best_margin = -np.inf
    for n in nA:
        projA = A @ n
        projB = B @ n
        loA, hiA = projA.min(), projA.max()
        loB, hiB = projB.min(), projB.max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best_margin:
            best_margin = sep
    for n in nB:
        projA = A @ n
        projB = B @ n
        loA, hiA = projA.min(), projA.max()
        loB, hiB = projB.min(), projB.max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best_margin:
            best_margin = sep
    return best_margin

def _outer_side_for_points(pts):
    """Find minimum side length of regular hexagon (centered at origin) containing all pts."""
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 361):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a

def _compute_all(state, time_budget=None):
    """Compute objective and penalty for a given state.
    state: array of shape (33,) = [x0,y0,a0, x1,y1,a1, ...]
    Returns (outer_side, penalty, valid_flag)."""
    x = state[0::3]
    y = state[1::3]
    a = state[2::3]
    
    # Center of mass
    cx = x.mean()
    cy = y.mean()
    pts = np.column_stack((x - cx, y - cy))
    
    # Outer hexagon side
    s, _ = _outer_side_for_points(pts)
    
    # Separation check
    Vs = []
    for i in range(11):
        Vs.append(_hex_vertices(x[i], y[i], 1.0, a[i]))
    
    pen = 0.0
    valid = True
    for i in range(11):
        for j in range(i + 1, 11):
            m = _gap(Vs[i], Vs[j])
            if m <= 1e-6:
                valid = False
                pen += max(0.0, 1e-6 - m) ** 2
    
    return s, pen, valid

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 580.0  # Safety margin before 630s kill
    
    # Seed configurations - multiple starting points
    seeds = []
    
    # Seed 1: Compact hex arrangement (center + 6 ring + 4 outer)
    seed1 = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.866, 0.0],
        [-1.5, 0.866, 0.0],
        [1.5, -0.866, 0.0],
        [-1.5, -0.866, 0.0],
        [0.0, 1.732, 0.0],
        [0.0, -1.732, 0.0],
        [3.0, 1.732, 0.0],
        [-3.0, 1.732, 0.0],
        [3.0, -1.732, 0.0],
        [-3.0, -1.732, 0.0],
    ])
    seeds.append(seed1)
    
    # Seed 2: 3-5-3 grid arrangement
    seed2 = np.array([
        [0.0, 0.0, 0.0],
        [1.8, 0.0, 0.0],
        [-1.8, 0.0, 0.0],
        [0.9, 1.56, 0.0],
        [-0.9, 1.56, 0.0],
        [2.7, 1.56, 0.0],
        [-2.7, 1.56, 0.0],
        [0.9, -1.56, 0.0],
        [-0.9, -1.56, 0.0],
        [2.7, -1.56, 0.0],
        [-2.7, -1.56, 0.0],
    ])
    seeds.append(seed2)
    
    # Seed 3: Slightly perturbed version for diversity
    seed3 = seed1.copy()
    rng = np.random.RandomState(42)
    seed3[:, :2] += rng.normal(0, 0.1, (11, 2))
    seeds.append(seed3)
    
    best_val = -np.inf
    best_state = None
    best_s = np.inf
    best_a_outer = 0.0
    
    # Evaluate seeds first
    for seed in seeds:
        s, pen, valid = _compute_all(seed)
        if valid and s < best_s:
            best_s = s
            best_state = seed.copy()
            x = seed[0::3]
            y = seed[1::3]
            cx = x.mean()
            cy = y.mean()
            pts = np.column_stack((x - cx, y - cy))
            _, best_a_outer = _outer_side_for_points(pts)
            best_val = 1.0 / s
    
    # Differential evolution for global optimization
    bounds = [(-5, 5)] * 11 + [(-5, 5)] * 11 + [(-180, 180)] * 11
    
    def de_objective(state):
        """Objective for differential evolution: minimize side + penalty."""
        x = state[0::3]
        y = state[1::3]
        a = state[2::3]
        
        # Center of mass
        cx = x.mean()
        cy = y.mean()
        pts = np.column_stack((x - cx, y - cy))
        
        # Outer hexagon side
        s, _ = _outer_side_for_points(pts)
        
        # Separation check
        Vs = []
        for i in range(11):
            Vs.append(_hex_vertices(x[i], y[i], 1.0, a[i]))
        
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                m = _gap(Vs[i], Vs[j])
                if m <= 1e-6:
                    pen += max(0.0, 1e-6 - m) ** 2
        
        return s + 1000.0 * pen
    
    # Run differential evolution with multiple restarts
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        # Different initial populations for each restart
        if restart == 0:
            init = 'sobol'
        else:
            init = 'latinhypercube'
        
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                seed=restart,
                maxiter=50,
                popsize=8,
                tol=1e-4,
                polish=False,
                init=init if init == 'sobol' else 'latinhypercube',
                time_budget=deadline - time.time() - 30
            )
            
            s, pen, valid = _compute_all(result.x)
            if valid and s < best_s:
                best_s = s
                best_state = result.x.copy()
                x = result.x[0::3]
                y = result.x[1::3]
                cx = x.mean()
                cy = y.mean()
                pts = np.column_stack((x - cx, y - cy))
                _, best_a_outer = _outer_side_for_points(pts)
                best_val = 1.0 / s
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if best_state is not None and time.time() < deadline - 30:
        from scipy.optimize import minimize
        
        def local_objective(state):
            x = state[0::3]
            y = state[1::3]
            a = state[2::3]
            
            cx = x.mean()
            cy = y.mean()
            pts = np.column_stack((x - cx, y - cy))
            
            s, _ = _outer_side_for_points(pts)
            
            Vs = []
            for i in range(11):
                Vs.append(_hex_vertices(x[i], y[i], 1.0, a[i]))
            
            pen = 0.0
            for i in range(11):
                for j in range(i + 1, 11):
                    m = _gap(Vs[i], Vs[j])
                    if m <= 1e-6:
                        pen += max(0.0, 1e-6 - m) ** 2
            
            return s + 1000.0 * pen
        
        try:
            result_local = minimize(
                local_objective,
                best_state,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-6, 'xatol': 1e-5}
            )
            
            s, pen, valid = _compute_all(result_local.x)
            if valid and s < best_s:
                best_s = s
                best_state = result_local.x.copy()
                x = result_local.x[0::3]
                y = result_local.x[1::3]
                cx = x.mean()
                cy = y.mean()
                pts = np.column_stack((x - cx, y - cy))
                _, best_a_outer = _outer_side_for_points(pts)
                best_val = 1.0 / s
        except Exception:
            pass
    
    # Final validation and output
    if best_state is None or best_s == np.inf:
        # Fallback: return a known valid loose packing
        inner = np.array([
            [0.0, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [-1.25, 2.17, 0.0],
            [1.25, 2.17, 0.0],
            [-1.25, -2.17, 0.0],
            [1.25, -2.17, 0.0],
            [-3.75, 2.17, 0.0],
            [3.75, 2.17, 0.0],
            [-3.75, -2.17, 0.0],
            [3.75, -2.17, 0.0],
        ])
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, 8.0
    
    x = best_state[0::3]
    y = best_state[1::3]
    a = best_state[2::3]
    cx = x.mean()
    cy = y.mean()
    pts = np.column_stack((x - cx, y - cy))
    
    # Find best outer hex angle
    best_s_final, best_a_final = _outer_side_for_points(pts)
    
    # Final validation
    Vs = []
    for i in range(11):
        Vs.append(_hex_vertices(x[i], y[i], 1.0, a[i]))
    
    valid = True
    for i in range(11):
        for j in range(i + 1, 11):
            m = _gap(Vs[i], Vs[j])
            if m <= 1e-6:
                valid = False
    
    if not valid or best_s_final <= 0:
        # Fallback
        inner = np.array([
            [0.0, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [-1.25, 2.17, 0.0],
            [1.25, 2.17, 0.0],
            [-1.25, -2.17, 0.0],
            [1.25, -2.17, 0.0],
            [-3.75, 2.17, 0.0],
            [3.75, 2.17, 0.0],
            [-3.75, -2.17, 0.0],
            [3.75, -2.17, 0.0],
        ])
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, 8.0
    
    inner = np.column_stack((x, y, a))
    outer = np.array([cx, cy, best_a_final])
    return inner, outer, float(best_s_final)

# EVOLVE-BLOCK-END
