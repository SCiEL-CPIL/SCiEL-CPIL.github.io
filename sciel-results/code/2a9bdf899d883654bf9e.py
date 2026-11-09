import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

DEADLINE = 620.0

def _spherical_to_cartesian(theta, phi):
    """Convert spherical coords (theta=polar angle from z-axis, phi=azimuthal) to cartesian on unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """Convert cartesian points on unit sphere to (theta, phi) pairs."""
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return np.column_stack([theta, phi])

def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 0.0
    if d_min < 1e-10:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _score_spherical(x_flat):
    """
    Score function for spherical coordinates.
    x_flat: (28,) array of 14 (theta, phi) pairs.
    Returns negative score (for minimization).
    """
    theta = x_flat[0::2]
    phi = x_flat[1::2]
    points = _spherical_to_cartesian(theta, phi)
    
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(D2)
    
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    dists = dists_mat[mask]
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10 or d_min < 1e-10:
        return 0.0
        
    score = (d_min / d_max) ** 2
    return -score

def _score_spherical_smooth(x_flat, temperature=0.1):
    """
    Smooth surrogate in spherical coordinates using soft-min/soft-max.
    x_flat: (28,) array of 14 (theta, phi) pairs.
    Returns (value, gradient) for minimization.
    """
    theta = x_flat[0::2]
    phi = x_flat[1::2]
    points = _spherical_to_cartesian(theta, phi)
    
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(D2)
    
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    dists = dists_mat[mask]
    
    if len(dists) == 0 or np.any(dists < 1e-10):
        return 0.0, np.zeros(28)
    
    scale = 1.0 / temperature
    d_min_raw = np.min(dists)
    d_max_raw = np.max(dists)
    
    exp_neg = np.exp(-(dists - d_min_raw) * scale)
    sum_exp_neg = np.sum(exp_neg)
    soft_min = d_min_raw - temperature * np.log(sum_exp_neg)
    
    exp_pos = np.exp((dists - d_max_raw) * scale)
    sum_exp_pos = np.sum(exp_pos)
    soft_max = d_max_raw + temperature * np.log(sum_exp_pos)
    
    if soft_max < 1e-10 or soft_min < 1e-10:
        return 0.0, np.zeros(28)
        
    score = (soft_min / soft_max) ** 2
    val = -score
    
    w_min = exp_neg / sum_exp_neg
    w_max = exp_pos / sum_exp_pos
    
    coeff_min = 2.0 * soft_min / (soft_max * soft_max)
    coeff_max = -2.0 * (soft_min * soft_min) / (soft_max * soft_max * soft_max)
    
    # Compute gradient in cartesian space first
    grad_cart = np.zeros((14, 3))
    
    indices = np.argwhere(mask)
    for i, j in indices:
        d_val = dists_mat[i, j]
        if d_val > 1e-10:
            unit_vec = (points[i] - points[j]) / d_val
            idx = i * 13 - i * (i + 1) // 2 + j - i - 1
            
            g_contrib = coeff_min * w_min[idx] * unit_vec + coeff_max * w_max[idx] * unit_vec
            grad_cart[i] += g_contrib
            grad_cart[j] -= g_contrib
    
    # Transform cartesian gradient to spherical coordinates
    # d/dtheta and d/dphi for each point
    grad_sph = np.zeros(28)
    sin_theta = np.sin(theta)
    for k in range(14):
        # dp/dtheta = (cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta))
        dtheta_vec = np.array([
            np.cos(theta[k]) * np.cos(phi[k]),
            np.cos(theta[k]) * np.sin(phi[k]),
            -np.sin(theta[k])
        ])
        # dp/dphi = (-sin(theta)*sin(phi), sin(theta)*cos(phi), 0)
        dphi_vec = np.array([
            -sin_theta[k] * np.sin(phi[k]),
            sin_theta[k] * np.cos(phi[k]),
            0.0
        ])
        grad_sph[2*k] = np.dot(grad_cart[k], dtheta_vec)
        grad_sph[2*k+1] = np.dot(grad_cart[k], dphi_vec)
    
    return val, grad_sph

def _optimize_spherical(x0_cart, deadline, use_smooth=True, temperature=0.1):
    """
    Optimize starting from x0_cart (14,3) points on sphere, using spherical coordinates.
    Returns (cartesian_points, score).
    """
    if time.time() > deadline:
        return x0_cart, 0.0
    
    # Convert to spherical
    x0_sph = _cartesian_to_spherical(x0_cart).flatten()
    
    try:
        x_current = x0_sph
        
        if use_smooth:
            # First pass with smooth surrogate
            res = minimize(
                lambda x: _score_spherical_smooth(x, temperature)[0],
                x0_sph,
                jac=lambda x: _score_spherical_smooth(x, temperature)[1],
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            x_current = res.x
            
            # Second pass with exact objective
            res2 = minimize(
                lambda x: _score_spherical(x),
                x_current,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            x_current = res2.x
        
        # Third pass: high-precision polish
        res3 = minimize(
            lambda x: _score_spherical(x),
            x_current,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        x_current = res3.x
        
        cand = _spherical_to_cartesian(x_current[0::2], x_current[1::2])
        return cand, _raw_score(cand)
    except Exception:
        return x0_cart, 0.0

def _generate_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    points = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    return points

def _generate_fibonacci_sphere(n=14):
    """
    Generate n points on a sphere using the Fibonacci lattice.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        radius = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_points = np.zeros((14, 3))
    
    rng = np.random.default_rng(42)
    
    # Initial guesses
    guesses = []
    
    # 1. Icosahedron + 2 points
    ico = _generate_icosahedron()
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    
    p1 = rng.normal(size=3)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=3)
    p2 = p2 / np.linalg.norm(p2)
    
    guess1 = np.vstack([ico, p1, p2])
    guesses.append(guess1)
    
    # 2. Fibonacci sphere
    fib = _generate_fibonacci_sphere(14)
    guesses.append(fib)
    
    # 3. Random points on sphere
    for _ in range(10):
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        guesses.append(pts)
        
    # 4. Cubic points
    cube = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(cube, axis=1, keepdims=True)
    cube = cube / norms
    guesses.append(cube)
    
    # 5. Octahedron + extra points
    octa = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(octa, axis=1, keepdims=True)
    octa = octa / norms
    for _ in range(8):
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p)
        octa = np.vstack([octa, p])
    guesses.append(octa)
    
    # Optimize each guess using spherical coordinates
    for guess in guesses:
        if time.time() > deadline - 30:
            break
            
        pts, score = _optimize_spherical(guess, deadline, use_smooth=True, temperature=0.1)
        if score > best_score:
            best_score = score
            best_points = pts
            
    # If we have time, try a few more random starts
    while time.time() < deadline - 60:
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        pts_opt, score = _optimize_spherical(pts, deadline, use_smooth=True, temperature=0.05)
        if score > best_score:
            best_score = score
            best_points = pts_opt

    # Final polish pass on best points
    if time.time() < deadline - 10 and best_score > 0:
        try:
            best_points_final, final_score = _optimize_spherical(
                best_points, deadline, use_smooth=False, temperature=0.01
            )
            if final_score > best_score:
                best_score = final_score
                best_points = best_points_final
        except Exception:
            pass

    if best_score == 0.0:
        # Fallback
        best_points = _generate_icosahedron()[:12]
        p1 = rng.normal(size=3)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=3)
        p2 = p2 / np.linalg.norm(p2)
        best_points = np.vstack([best_points, p1, p2])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms

    return best_points

if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    print(_raw_score(pts))
