"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
import time

N = 13
DEADLINE = 120.0  # seconds, with safety margin
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def remaining():
    return DEADLINE - 5.0 - elapsed()  # 5s safety margin


def min_triangle_area(pts):
    """Compute minimum area over all triples of points."""
    n = pts.shape[0]
    min_a = np.inf
    # Vectorized: compute all triangle areas
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if a < min_a:
                    min_a = a
    return min_a


def hull_area(pts):
    """Compute convex hull area."""
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0


def score_pts(pts):
    """Score = min triangle area / hull area."""
    ha = hull_area(pts)
    if ha < 1e-12:
        return 0.0
    ma = min_triangle_area(pts)
    return ma / ha


def regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def params_to_points(radii, angular_shifts):
    """Convert radial and angular perturbation parameters to point coordinates.
    
    radii: shape (N,), radial scale factors (positive)
    angular_shifts: shape (N,), angular offsets from regular polygon
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + angular_shifts
    # Normalize to keep points in a reasonable region
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def objective(params):
    """Negative score to minimize."""
    radii = params[:N]
    ang_shifts = params[N:]
    # Ensure radii are positive
    radii = np.abs(radii) + 1e-6
    pts = params_to_points(radii, ang_shifts)
    ha = hull_area(pts)
    if ha < 1e-12:
        return 0.0
    ma = min_triangle_area(pts)
    return -(ma / ha)


def objective_grad(params):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = objective(params)
    for i in range(len(params)):
        params[i] += eps
        f1 = objective(params)
        params[i] -= 2 * eps
        f2 = objective(params)
        params[i] += eps
        grad[i] = (f1 - f2) / (2 * eps)
    return grad


def optimize_from(params0, max_iter=200, tol=1e-10):
    """Run L-BFGS-B optimization from initial parameters."""
    result = minimize(
        objective,
        params0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': tol, 'disp': False}
    )
    return result


def try_optimization(params0, label=""):
    """Try optimization and return (score, params) if improved."""
    if remaining() < 10:
        return None
    
    result = optimize_from(params0)
    if result.success or result.fun < 0:
        s = -result.fun
        return (s, result.x)
    return None


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_score = 0.0
    best_params = None
    
    # Initial: regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Strategy 1: Regular polygon with radial perturbations
    # Try various radial perturbation patterns
    
    # 1a: Perfectly regular
    r0 = np.ones(N)
    a0 = np.zeros(N)
    p0 = np.concatenate([r0, a0])
    
    res = try_optimization(p0, "regular")
    if res is not None:
        best_score, best_params = res
    
    # 1b: Small radial perturbations (multiple frequencies)
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if remaining() < 15:
            break
        
        # Random radial perturbations with different frequency structures
        if trial == 0:
            # Uniform small noise
            r = 1.0 + 0.1 * rng.randn(N)
            a = 0.05 * rng.randn(N)
        elif trial == 1:
            # First harmonic
            r = 1.0 + 0.15 * np.cos(2 * base_angles)
            a = 0.05 * np.sin(2 * base_angles)
        elif trial == 2:
            # Second harmonic
            r = 1.0 + 0.1 * np.cos(4 * base_angles)
            a = 0.04 * np.sin(4 * base_angles)
        elif trial == 3:
            # Mixed harmonics
            r = 1.0 + 0.1 * np.cos(2 * base_angles) + 0.05 * np.cos(4 * base_angles)
            a = 0.03 * np.sin(2 * base_angles) + 0.03 * np.sin(4 * base_angles)
        elif trial < 8:
            # Random smooth perturbations
            freqs = rng.randint(1, 5, size=3)
            amps = rng.uniform(0.02, 0.15, size=3)
            phases = rng.uniform(0, 2 * np.pi, size=3)
            r = 1.0 + sum(amps[i] * np.cos(freqs[i] * base_angles + phases[i]) for i in range(3))
            a = 0.03 * rng.randn(N)
        else:
            # Larger perturbations
            r = 1.0 + 0.2 * rng.randn(N)
            a = 0.1 * rng.randn(N)
        
        p = np.concatenate([np.abs(r) + 1e-6, a])
        res = try_optimization(p, f"trial{trial}")
        if res is not None:
            if res[0] > best_score:
                best_score, best_params = res
    
    # Strategy 2: Multi-start from best solution with perturbations
    if best_params is not None and remaining() > 20:
        for restart in range(15):
            if remaining() < 10:
                break
            
            # Perturb the best solution
            noise_scale = 0.05 * (1 + restart * 0.5)
            p = best_params + noise_scale * rng.randn(2 * N)
            p[:N] = np.abs(p[:N]) + 1e-6  # Keep radii positive
            
            res = try_optimization(p, f"restart{restart}")
            if res is not None:
                if res[0] > best_score:
                    best_score, best_params = res
    
    # Strategy 3: Try different angular structures
    if remaining() > 15:
        for ang_pattern in range(5):
            if remaining() < 10:
                break
            
            if ang_pattern == 0:
                # Slight angular compression
                a_shift = 0.02 * np.sin(2 * base_angles)
                r = np.ones(N)
            elif ang_pattern == 1:
                # Alternating angular shifts
                a_shift = 0.03 * np.where(np.arange(N) % 2 == 0, 1, -1)
                r = 1.0 + 0.05 * np.where(np.arange(N) % 2 == 0, 1, -1)
            elif ang_pattern == 2:
                # Gradual angular drift
                a_shift = 0.01 * np.arange(N)
                r = np.ones(N)
            elif ang_pattern == 3:
                # Two-group angular separation
                a_shift = 0.04 * np.sign(np.cos(2 * base_angles))
                r = 1.0 + 0.03 * np.sign(np.cos(2 * base_angles))
            else:
                # Random but smooth
                a_shift = 0.05 * rng.randn(N)
                a_shift = np.convolve(a_shift, np.ones(3)/3, mode='same')
                r = 1.0 + 0.08 * rng.randn(N)
            
            p = np.concatenate([np.abs(r) + 1e-6, a_shift])
            res = try_optimization(p, f"ang{ang_pattern}")
            if res is not None:
                if res[0] > best_score:
                    best_score, best_params = res
    
    # Convert best params to points
    if best_params is not None:
        radii = np.abs(best_params[:N]) + 1e-6
        ang_shifts = best_params[N:]
        pts = params_to_points(radii, ang_shifts)
    else:
        pts = regular_polygon(N)
    
    # Normalize to unit area
    ha = hull_area(pts)
    if ha > 1e-12:
        scale = 1.0 / np.sqrt(ha)
        pts = pts * scale
    
    # Final polish: local optimization on actual coordinates
    if remaining() > 5:
        def coord_objective(coords):
            pts = coords.reshape(N, 2)
            ha = hull_area(pts)
            if ha < 1e-12:
                return 0.0
            ma = min_triangle_area(pts)
            return -(ma / ha)
        
        x0 = pts.flatten()
        result = minimize(coord_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 100, 'gtol': 1e-10})
        if result.fun < 0:
            final_pts = result.x.reshape(N, 2)
            final_ha = hull_area(final_pts)
            if final_ha > 1e-12:
                final_pts = final_pts / np.sqrt(final_ha)
                return final_pts
    
    return pts


if __name__ == "__main__":
    pts = construct()
    print(f"Points shape: {pts.shape}")
    print(f"Score: {score_pts(pts):.6f}")
