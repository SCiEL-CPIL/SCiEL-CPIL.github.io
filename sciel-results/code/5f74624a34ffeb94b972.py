import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def normalize_polygon(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return points
    return centered / mean_dist

def get_points_from_angles(thetas):
    """Generate 20 points from angles (radians) in cyclic order."""
    # Ensure angles are in [0, 2*pi) and sorted
    thetas = np.sort(thetas % (2 * np.pi))
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def get_points_from_fourier(coeffs):
    """
    Generate 20 points from Fourier coefficients of angular deviations.
    coeffs: array of length 2*K (cos and sin for k=1..K).
    theta_i = 2*pi*i/20 + sum_{k=1}^{K} (a_k * cos(2*pi*k*i/20) + b_k * sin(2*pi*k*i/20))
    """
    n = 20
    base = np.arange(n) * 2 * np.pi / n
    deviation = np.zeros(n)
    K = len(coeffs) // 2
    for k in range(1, K + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        angles_k = 2 * np.pi * k * np.arange(n) / n
        deviation += a * np.cos(angles_k) + b * np.sin(angles_k)
    thetas = base + deviation
    return get_points_from_angles(thetas)

def vertex_score(v_idx, points):
    """
    Calculate the score for a single vertex.
    """
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
        
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
        
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    if raw == 0:
        return 0.0
        
    score = min(1.0, 1.0 / raw)
    return score

def min_vertex_score_fourier(coeffs):
    """
    Objective: negative of min vertex score for the polygon defined by Fourier coeffs.
    """
    points = get_points_from_fourier(coeffs)
    points_norm = normalize_polygon(points)
    n = 20
    scores = np.array([vertex_score(i, points_norm) for i in range(n)])
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Number of Fourier modes
    # Let's try K=3 (6 parameters) or K=4 (8 parameters)
    # The task says "equidistant vertices" which usually implies a regular polygon,
    # but the scoring function rewards specific non-regular configurations.
    # The "equidistant" in the task title might be misleading or refer to the initial state.
    # The scoring is based on distances between vertices.
    # Regular polygon: all distances from a vertex are distinct except symmetries?
    # In a regular 20-gon, distances from v are: 2*sin(pi*k/20) for k=1..9, then 2, then 2*sin(pi*k/20) for k=10..19 (symmetric).
    # Actually, distances are: d_k = 2*sin(pi*k/20).
    # Sorted: d_1, d_2, ..., d_9, 2, d_9, ..., d_1.
    # Gaps: d_{k+1}-d_k.
    # The "gap" logic in vertex_score looks for the smallest gap in sorted distances.
    # For regular polygon, the gaps might be relatively uniform or have a specific pattern.
    
    # Let's use K=3 first (6 params) to keep it low dimensional and smooth.
    # If that's not enough, we can increase.
    K = 3
    n_params = 2 * K
    
    # Bounds for Fourier coefficients
    # The deviation should be small enough to maintain convexity and cyclic order.
    # Max deviation in angle: sum of |a_k| + |b_k|.
    # If we bound each coeff to [-0.1, 0.1], max deviation is 0.6 rad ~ 34 deg.
    # Base spacing is 18 deg. This might break convexity if not careful.
    # Let's be conservative: [-0.05, 0.05] for each coeff.
    bounds = [(-0.05, 0.05) for _ in range(n_params)]
    
    best_coeffs = np.zeros(n_params)
    best_score = -min_vertex_score_fourier(best_coeffs)
    
    # 1. Differential Evolution (Global)
    # Use a moderate population and iterations
    try:
        res_de = differential_evolution(
            min_vertex_score_fourier,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            polish=False,
            seed=42,
            updating='immediate'
        )
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_coeffs = res_de.x
    except Exception:
        pass
    
    # 2. Local Refinement (L-BFGS-B)
    # Multiple starts from best DE solution and nearby points
    rng = np.random.default_rng(123)
    candidates = [best_coeffs.copy()]
    for _ in range(5):
        candidates.append(best_coeffs + rng.normal(0, 0.01, size=n_params))
    
    for c in candidates:
        if time.time() > deadline - 30:
            break
        try:
            res_local = minimize(
                min_vertex_score_fourier,
                c,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_coeffs = res_local.x
        except Exception:
            pass
    
    # 3. Final Polish
    if time.time() < deadline - 10:
        try:
            res_final = minimize(
                min_vertex_score_fourier,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_coeffs = res_final.x
        except Exception:
            pass
    
    return get_points_from_fourier(best_coeffs)

# Fallback: if Fourier approach fails or is too slow, we could fall back to the previous method,
# but the directive says to keep the design family (Fourier).
# We will stick to Fourier.

if __name__ == "__main__":
    pts = equidistant_points_polygon20()
    print(pts)
