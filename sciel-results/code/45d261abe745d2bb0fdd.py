"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2  # height

def _cart_to_bary(x, y):
    """Convert Cartesian to barycentric coordinates for the unit equilateral triangle."""
    # v0=(0,0), v1=(1,0), v2=(0.5, H)
    # x = b + 0.5*c, y = H*c  =>  c = y/H, b = x - 0.5*c, a = 1-b-c
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return a, b, c

def _bary_to_cart(a, b, c):
    """Convert barycentric to Cartesian."""
    x = b + 0.5 * c
    y = H * c
    return x, y

def _clamp_bary(a, b, c, eps=1e-9):
    """Clamp barycentric coords to be inside the triangle (a,b,c >= 0, sum=1)."""
    # Project to the simplex
    a = max(eps, a)
    b = max(eps, b)
    c = max(eps, c)
    s = a + b + c
    a /= s
    b /= s
    c /= s
    return a, b, c

def _min_area_bary(bary_pts):
    """Compute minimum triangle area from barycentric points (N x 3 array)."""
    n = len(bary_pts)
    # Convert to cartesian
    c = bary_pts[:, 2]
    b = bary_pts[:, 1]
    a = bary_pts[:, 0]
    x = b + 0.5 * c
    y = H * c
    # Compute all triangle areas vectorized
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = x[i], y[i]
                bx, by = x[j], y[j]
                cx, cy = x[k], y[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_a:
                    min_a = area
                    if min_a < 1e-12:
                        return 0.0
    return min_a

def _min_area_cart(pts):
    """Compute minimum triangle area from Cartesian points (N x 2 array)."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i]))
                if area < min_a:
                    min_a = area
                    if min_a < 1e-12:
                        return 0.0
    return min_a

def _smooth_min_area_cart(pts, beta=30.0):
    """Smooth approximation of min area using log-sum-exp (vectorized)."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i]))
                log_area = np.log(area + 1e-14)
                log_sum = np.logaddexp(log_sum, log_area)
    return log_sum / beta

def _clamp_to_triangle(p):
    """Clamp a single point to be inside the equilateral triangle."""
    x, y = p
    a, b, c = _cart_to_bary(x, y)
    a, b, c = _clamp_bary(a, b, c)
    return np.array(_bary_to_cart(a, b, c))

def _clamp_all(pts):
    """Clamp all points to be inside the triangle."""
    out = np.empty_like(pts)
    for i in range(len(pts)):
        out[i] = _clamp_to_triangle(pts[i])
    return out

def _initial_placement():
    """Structured initial placement: triangular lattice in barycentric coords."""
    # Use a 4-row triangular arrangement: 4 + 3 + 2 + 2 = 11
    # Actually let's do a hexagonal-ish packing
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 2 points
    pts = []
    # Bottom row
    y0 = 0.08
    for i in range(4):
        x = 0.08 + i * (0.84 / 3)
        pts.append([x, y0])
    # Second row
    y1 = 0.30
    for i in range(3):
        x = 0.18 + i * (0.64 / 2)
        pts.append([x, y1])
    # Third row
    y2 = 0.52
    for i in range(2):
        x = 0.28 + i * (0.44 / 1)
        pts.append([x, y2])
    # Top row
    y3 = 0.68
    for i in range(2):
        x = 0.38 + i * (0.24 / 1)
        pts.append([x, y3])
    pts = np.array(pts)
    return _clamp_all(pts)

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    best_pts = _initial_placement()
    best_score = _min_area_cart(best_pts)

    # Strategy 1: Differential Evolution for global search
    # Work in Cartesian, clamp inside objective
    def _de_objective(flat):
        pts = flat.reshape(-1, 2)
        pts = _clamp_all(pts)
        return -_min_area_cart(pts)

    def _smooth_objective(flat, beta=30.0):
        pts = flat.reshape(-1, 2)
        pts = _clamp_all(pts)
        return -_smooth_min_area_cart(pts, beta)

    def _sharp_objective(flat):
        pts = flat.reshape(-1, 2)
        pts = _clamp_all(pts)
        return -_min_area_cart(pts)

    # DE bounds: x in [0,1], y in [0, H]
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N

    rng = np.random.default_rng(42)

    # Run DE with moderate population
    if time.time() - start_time < deadline - 30:
        try:
            de_res = differential_evolution(
                _de_objective,
                bounds=bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 0.9),
                recombination=0.8,
                polish=True,
                init='latinhypercube',
            )
            de_pts = _clamp_all(de_res.x.reshape(-1, 2))
            de_score = _min_area_cart(de_pts)
            if de_score > best_score:
                best_pts = de_pts
                best_score = de_score
        except Exception:
            pass

    # Strategy 2: BFGS refinement with sharp objective
    if time.time() - start_time < deadline - 15:
        try:
            res = minimize(
                _sharp_objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            opt_pts = _clamp_all(res.x.reshape(-1, 2))
            opt_score = _min_area_cart(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
        except Exception:
            pass

    # Strategy 3: Smooth BFGS with multiple betas to escape local minima
    if time.time() - start_time < deadline - 20:
        for beta in [15.0, 30.0, 60.0, 100.0]:
            if time.time() - start_time > deadline - 10:
                break
            try:
                res_s = minimize(
                    lambda x: _smooth_objective(x, beta),
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                s_pts = _clamp_all(res_s.x.reshape(-1, 2))
                s_score = _min_area_cart(s_pts)
                if s_score > best_score:
                    best_pts = s_pts
                    best_score = s_score
            except Exception:
                pass

    # Strategy 4: Multiple random restarts with BFGS
    for trial in range(15):
        if time.time() - start_time > deadline - 8:
            break
        # Perturb current best
        noise_scale = 0.08
        trial_pts = best_pts + rng.normal(0, noise_scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        try:
            res = minimize(
                _sharp_objective,
                trial_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            opt_pts = _clamp_all(res.x.reshape(-1, 2))
            opt_score = _min_area_cart(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
        except Exception:
            pass

    # Strategy 5: Simulated annealing refinement
    if time.time() - start_time < deadline - 12:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.015
        for iteration in range(3000):
            if time.time() - start_time > deadline - 3:
                break
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] = _clamp_to_triangle(current_pts[idx] + delta)
            new_score = _min_area_cart(new_pts)
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_pts = current_pts.copy()
                    best_score = current_score
            T *= 0.9995

    # Strategy 6: Final high-precision polish
    if time.time() - start_time < deadline - 3:
        try:
            res_final = minimize(
                _sharp_objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area_cart(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass

    # Final clamp and return
    best_pts = _clamp_all(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
