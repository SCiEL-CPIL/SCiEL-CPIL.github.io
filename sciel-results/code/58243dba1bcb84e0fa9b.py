"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12
SQRT3 = math.sqrt(3.0)

# Precompute support function samples for a unit hexagon (centered at origin)
# We'll use the formula: h(α) = max_k cos(α - k*π/3) for k=0..5
# But for a unit side hexagon, the circumradius is 1 (since side = circumradius for regular hexagon)
# Wait - for a regular hexagon with side length s, circumradius = s, inradius = s*sqrt(3)/2
# So vertices are at distance 1 from center, and the support function is:
# h(α) = max_k dot(vertex_k, unit_dir(α)) = max_k cos(α - θ_k)
# where θ_k are the vertex angles.

# For efficiency, precompute h(α) on a fine grid
H_N_SAMPLES = 720  # every 0.5 degrees
_H_ALPHA_GRID = np.linspace(0, 2*math.pi, H_N_SAMPLES, endpoint=False)
_H_VALUES = np.zeros(H_N_SAMPLES)
for i, alpha in enumerate(_H_ALPHA_GRID):
    _H_VALUES[i] = max(math.cos(alpha - k*math.pi/3.0) for k in range(6))

def _h_at_angle(alpha):
    """Support function of unit hexagon at angle alpha, using linear interpolation."""
    # Normalize alpha to [0, 2π)
    alpha = alpha % (2*math.pi)
    idx_float = alpha / (2*math.pi) * H_N_SAMPLES
    i0 = int(idx_float) % H_N_SAMPLES
    i1 = (i0 + 1) % H_N_SAMPLES
    frac = idx_float - int(idx_float)
    return _H_VALUES[i0] * (1 - frac) + _H_VALUES[i1] * frac

def _min_separation(dtheta):
    """
    Minimum center-to-center distance for two unit hexagons with relative rotation dtheta
    to be non-overlapping. This is the max over all directions of h(α) + h(α - dtheta).
    """
    dtheta = abs(dtheta) % (2*math.pi)
    dtheta = min(dtheta, 2*math.pi - dtheta)
    
    # Compute max over all directions of h(α) + h(α - dtheta)
    # Use the precomputed grid
    max_sep = 0.0
    for i in range(H_N_SAMPLES):
        alpha = _H_ALPHA_GRID[i]
        h1 = _H_VALUES[i]
        # h(α - dtheta) via interpolation
        h2 = _h_at_angle(alpha - dtheta)
        sep = h1 + h2
        if sep > max_sep:
            max_sep = sep
    return max_sep

# Precompute min_separation for a grid of dtheta values
MIN_SEP_N = 720
_MIN_SEP_DTHETA_GRID = np.linspace(0, math.pi/3.0, MIN_SEP_N, endpoint=True)
_MIN_SEP_VALUES = np.zeros(MIN_SEP_N)
for i, dt in enumerate(_MIN_SEP_DTHETA_GRID):
    _MIN_SEP_VALUES[i] = _min_separation(dt)

def _min_sep_lookup(dtheta):
    """Lookup minimum separation using interpolation on precomputed grid."""
    dtheta = abs(dtheta) % (2*math.pi)
    dtheta = min(dtheta, 2*math.pi - dtheta)
    # dtheta is in [0, π/3]
    idx_float = dtheta / (math.pi/3.0) * (MIN_SEP_N - 1)
    i0 = int(idx_float)
    if i0 >= MIN_SEP_N - 1:
        return _MIN_SEP_VALUES[MIN_SEP_N - 1]
    i1 = i0 + 1
    frac = idx_float - i0
    return _MIN_SEP_VALUES[i0] * (1 - frac) + _MIN_SEP_VALUES[i1] * frac

# Normal directions for outer hexagon (6 sides)
# For a regular hexagon of side L centered at origin, the 6 outward normals are at angles
# 30°, 90°, 150°, 210°, 270°, 330° (for flat-top orientation)
_NORMAL_ANGLES = np.array([math.radians(30 + 60*j) for j in range(6)])
_NORMAL_COS = np.cos(_NORMAL_ANGLES)
_NORMAL_SIN = np.sin(_NORMAL_ANGLES)

def _required_L(cx, cy, theta):
    """Minimum side L of outer hex (centered at origin, flat-top) to contain unit hex at (cx,cy) with rotation theta."""
    # Vertices of unit hex at (cx,cy) with rotation theta
    # Vertex k is at (cx + cos(theta + k*pi/3), cy + sin(theta + k*pi/3))
    vx = cx + np.cos(theta + np.arange(6)*math.pi/3.0)
    vy = cy + np.sin(theta + np.arange(6)*math.pi/3.0)
    
    # Project onto each normal direction
    # proj[j] = max over vertices of (vx*ncos[j] + vy*nsin[j])
    projs = np.outer(vx, _NORMAL_COS) + np.outer(vy, _NORMAL_SIN)
    max_projs = np.max(projs, axis=0)  # shape (6,)
    
    # For outer hex of side L (flat-top, centered at origin):
    # The distance from center to each side along normal direction is L*sqrt(3)/2
    # So we need max_projs[j] <= L * sqrt(3)/2 for all j
    # => L >= 2*max(max_projs) / sqrt(3)
    return 2.0 * np.max(max_projs) / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    best_L = float('inf')
    best_params = None  # (N*3,) array: x0,y0,theta0, ..., x11,y11,theta11

    # --- Phase 1: Grid-based initialization ---
    # Generate hex grid centers (axial coords with |q|,|r|,|q+r| <= 2)
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = SQRT3 * (r + q / 2.0)
                all_centers.append((x, y))
    # 19 centers

    # For each cluster rotation, find best subset of 12 with optimal individual thetas
    def _optimal_theta_for_pos(cx, cy):
        """Find theta minimizing required_L for hex at (cx,cy)."""
        best_L_h = float('inf')
        best_t = 0.0
        # Coarse
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = _required_L(cx, cy, t)
            if L < best_L_h:
                best_L_h = L
                best_t = t
        # Fine
        for step in [0.5, 0.05, 0.005]:
            for t in np.linspace(max(0, best_t - 1.0), min(math.pi/3, best_t + 1.0), int(2.0/step)+1):
                L = _required_L(cx, cy, t)
                if L < best_L_h:
                    best_L_h = L
                    best_t = t
        return best_L_h, best_t

    SCALE = 1.0000001

    # Try several cluster rotations
    cluster_angles = [0.0, math.pi/12, math.pi/6, math.pi/4, 5*math.pi/12, math.pi/3]

    for ca in cluster_angles:
        if time.time() > deadline - 60:
            break
        cos_ca, sin_ca = math.cos(ca), math.sin(ca)
        rotated = []
        for x, y in all_centers:
            rx = x*cos_ca - y*sin_ca
            ry = x*sin_ca + y*cos_ca
            rotated.append((rx, ry))

        # Compute optimal theta and L for each center
        info = []
        for x, y in rotated:
            L_h, t_h = _optimal_theta_for_pos(x*SCALE, y*SCALE)
            info.append((L_h, t_h, x*SCALE, y*SCALE))

        # Sort by L_h, take best 12
        info_sorted = sorted(range(19), key=lambda i: info[i][0])
        # Try the top 12 and a few alternatives
        top12 = info_sorted[:12]
        L_candidate = max(info[i][0] for i in top12)
        
        if L_candidate < best_L:
            best_L = L_candidate
            params = np.zeros(N*3)
            for idx, i in enumerate(top12):
                params[idx*3] = info[i][2]
                params[idx*3+1] = info[i][3]
                params[idx*3+2] = info[i][1]
            best_params = params.copy()

    # --- Phase 2: Continuous optimization ---
    if HAS_SCIPY and best_params is not None and time.time() < deadline - 20:
        def _objective(params):
            """Minimize max L + overlap penalties."""
            positions = params[:N*2].reshape(N, 2)
            thetas = params[N*2:3*N]
            L_val = params[-1]
            
            # Compute L for each hex
            Ls = np.array([_required_L(positions[i,0], positions[i,1], thetas[i]) for i in range(N)])
            max_L = np.max(Ls)
            
            # Overlap penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i,0] - positions[j,0]
                    dy = positions[i,1] - positions[j,1]
                    d = math.sqrt(dx*dx + dy*dy)
                    dt = thetas[i] - thetas[j]
                    min_d = _min_sep_lookup(dt)
                    if d < min_d:
                        penalty += (min_d - d)**2 * 100.0
            
            # Containment penalty
            contain_pen = 0.0
            for i in range(N):
                if Ls[i] > L_val:
                    contain_pen += (Ls[i] - L_val)**2 * 100.0
            
            return max_L + penalty + contain_pen

        # Initial parameters
        init = best_params.copy()
        init = np.concatenate([init, [best_L * 1.001]])
        
        # Try Powell first (usually fast and effective)
        best_obj = _objective(init)
        
        try:
            res = minimize(_objective, init, method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x[:-1].copy()
                # Recompute actual L
                pos = best_params[:N*2].reshape(N,2)
                th = best_params[N*2:3*N]
                actual_L = max(_required_L(pos[i,0], pos[i,1], th[i]) for i in range(N))
                best_L = actual_L
        except Exception:
            pass

        # Try Nelder-Mead
        if time.time() < deadline - 10:
            try:
                res = minimize(_objective, init, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_obj:
                    best_obj = res.fun
                    pos = res.x[:N*2].reshape(N,2)
                    th = res.x[N*2:3*N]
                    # Check validity
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = pos[i,0] - pos[j,0]
                            dy = pos[i,1] - pos[j,1]
                            d = math.sqrt(dx*dx + dy*dy)
                            min_d = _min_sep_lookup(th[i] - th[j])
                            if d < min_d - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        actual_L = max(_required_L(pos[i,0], pos[i,1], th[i]) for i in range(N))
                        if actual_L < best_L:
                            best_L = actual_L
                            best_params = res.x[:-1].copy()
            except Exception:
                pass

        # Differential evolution for global search (if time allows)
        if time.time() < deadline - 30:
            try:
                # Bounded objective for DE
                def _de_objective(x):
                    # x: [x0..x11, y0..y11, theta0..theta11] mapped to bounds
                    # Bounds: x,y in [-15, 15], theta in [0, pi/3]
                    positions = x[:N*2].reshape(N, 2)
                    thetas = x[N*2:3*N] * (math.pi/3.0)  # map [0,1] -> [0, pi/3]
                    
                    Ls = np.array([_required_L(positions[i,0], positions[i,1], thetas[i]) for i in range(N)])
                    max_L = np.max(Ls)
                    
                    penalty = 0.0
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = positions[i,0] - positions[j,0]
                            dy = positions[i,1] - positions[j,1]
                            d = math.sqrt(dx*dx + dy*dy)
                            min_d = _min_sep_lookup(thetas[i] - thetas[j])
                            if d < min_d:
                                penalty += (min_d - d)**2 * 50.0
                    
                    return max_L + penalty

                bounds = [(-15, 15)] * (N*2) + [(0, 1)] * N
                res_de = differential_evolution(_de_objective, bounds, maxiter=100, tol=1e-8, seed=42)
                if res_de.fun < best_obj:
                    pos = res_de.x[:N*2].reshape(N,2)
                    th = res_de.x[N*2:3*N] * (math.pi/3.0)
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = pos[i,0] - pos[j,0]
                            dy = pos[i,1] - pos[j,1]
                            d = math.sqrt(dx*dx + dy*dy)
                            min_d = _min_sep_lookup(th[i] - th[j])
                            if d < min_d - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        actual_L = max(_required_L(pos[i,0], pos[i,1], th[i]) for i in range(N))
                        if actual_L < best_L:
                            best_L = actual_L
                            best_params = np.concatenate([pos.flatten(), th])
            except Exception:
                pass

    # --- Final: Polish with high-precision Nelder-Mead if time remains ---
    if HAS_SCIPY and best_params is not None and time.time() < deadline - 5:
        try:
            init = np.concatenate([best_params, [best_L * 1.0005]])
            res = minimize(_objective, init, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13})
            if res.fun < _objective(init):
                pos = res.x[:N*2].reshape(N,2)
                th = res.x[N*2:3*N]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        d = math.sqrt(dx*dx + dy*dy)
                        min_d = _min_sep_lookup(th[i] - th[j])
                        if d < min_d - 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    actual_L = max(_required_L(pos[i,0], pos[i,1], th[i]) for i in range(N))
                    if actual_L < best_L:
                        best_L = actual_L
                        best_params = res.x[:-1].copy()
        except Exception:
            pass

    # Construct output
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        theta_deg = math.degrees(best_params[i*3+2])
        inner.append([x, y, theta_deg])

    return inner, (0.0, 0.0), best_L, 0.0
