import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

DEADLINE = 620.0

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

def _score_and_grad_sphere(x_flat):
    """
    Objective and gradient for sphere-constrained optimization.
    x_flat: (42,) array of 14 points in R^3.
    """
    points = x_flat.reshape(14, 3)
    
    # Calculate distances
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(D2)
    
    # Mask for i < j
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    dists = dists_mat[mask]
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10 or d_min < 1e-10:
        return 0.0, np.zeros(42)
        
    score = (d_min / d_max) ** 2
    val = -score
    
    # Calculate subgradient
    # Identify active min and max pairs
    eps = 1e-6
    min_mask = (dists_mat < d_min + eps) & mask
    max_mask = (dists_mat > d_max - eps) & mask
    
    grad = np.zeros((14, 3))
    
    # Gradient of d_min
    g_min = np.zeros((14, 3))
    min_indices = np.argwhere(min_mask)
    if len(min_indices) > 0:
        for i, j in min_indices:
            d_vec = points[i] - points[j]
            d_val = dists_mat[i, j]
            if d_val > 1e-10:
                unit_vec = d_vec / d_val
                g_min[i] += unit_vec
                g_min[j] -= unit_vec
        g_min /= len(min_indices)
        
    # Gradient of d_max
    g_max = np.zeros((14, 3))
    max_indices = np.argwhere(max_mask)
    if len(max_indices) > 0:
        for i, j in max_indices:
            d_vec = points[i] - points[j]
            d_val = dists_mat[i, j]
            if d_val > 1e-10:
                unit_vec = d_vec / d_val
                g_max[i] += unit_vec
                g_max[j] -= unit_vec
        g_max /= len(max_indices)
        
    # dscore/dX = 2 * (d_min/d_max) * (1/d_max * g_min - d_min/d_max^2 * g_max)
    # = 2 * d_min / d_max^2 * g_min - 2 * d_min^2 / d_max^3 * g_max
    
    coeff_min = 2.0 * d_min / (d_max * d_max)
    coeff_max = 2.0 * (d_min * d_min) / (d_max * d_max * d_max)
    
    grad = coeff_min * g_min - coeff_max * g_max
    
    # We are minimizing -score, so gradient is -grad_score
    grad_flat = (-grad).flatten()
    
    return val, grad_flat

def _score_and_grad_smooth(x_flat, temperature=0.1):
    """
    Smooth surrogate using soft-min and soft-max via log-sum-exp.
    This provides a differentiable approximation to (d_min/d_max)^2.
    """
    points = x_flat.reshape(14, 3)
    
    # Calculate distances
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(D2)
    
    # Mask for i < j
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    dists = dists_mat[mask]
    
    if len(dists) == 0 or np.any(dists < 1e-10):
        return 0.0, np.zeros(42)
    
    scale = 1.0 / temperature
    
    d_min_raw = np.min(dists)
    d_max_raw = np.max(dists)
    
    # Soft min
    exp_neg = np.exp(-(dists - d_min_raw) * scale)
    sum_exp_neg = np.sum(exp_neg)
    soft_min = d_min_raw - temperature * np.log(sum_exp_neg)
    
    # Soft max
    exp_pos = np.exp((dists - d_max_raw) * scale)
    sum_exp_pos = np.sum(exp_pos)
    soft_max = d_max_raw + temperature * np.log(sum_exp_pos)
    
    if soft_max < 1e-10 or soft_min < 1e-10:
        return 0.0, np.zeros(42)
        
    score = (soft_min / soft_max) ** 2
    val = -score
    
    # Gradient computation
    # d(soft_min)/d(dists) = -exp_neg / sum_exp_neg
    # d(soft_max)/d(dists) = exp_pos / sum_exp_pos
    
    w_min = exp_neg / sum_exp_neg  # weights for soft_min
    w_max = exp_pos / sum_exp_pos  # weights for soft_max
    
    # dscore/d(soft_min) = 2 * soft_min / soft_max^2
    # dscore/d(soft_max) = -2 * soft_min^2 / soft_max^3
    
    coeff_min = 2.0 * soft_min / (soft_max * soft_max)
    coeff_max = -2.0 * (soft_min * soft_min) / (soft_max * soft_max * soft_max)
    
    # d(soft_min)/dX = sum_j w_min[j] * d(d_j)/dX
    # d(soft_max)/dX = sum_j w_max[j] * d(d_j)/dX
    
    grad = np.zeros((14, 3))
    
    indices = np.argwhere(mask)
    for i, j in indices:
        d_val = dists_mat[i, j]
        if d_val > 1e-10:
            unit_vec = (points[i] - points[j]) / d_val
            
            # Index into w_min and w_max
            idx = i * 13 - i * (i + 1) // 2 + j - i - 1
            
            # Contribution to soft_min gradient
            g_min_contrib = w_min[idx] * unit_vec
            # Contribution to soft_max gradient
            g_max_contrib = w_max[idx] * unit_vec
            
            grad[i] += coeff_min * g_min_contrib + coeff_max * g_max_contrib
            grad[j] -= coeff_min * g_min_contrib + coeff_max * g_max_contrib
    
    # We are minimizing -score, so gradient is -grad_score
    grad_flat = (-grad).flatten()
    
    return val, grad_flat

def _constraints_sphere(x_flat):
    """
    Constraints for points to lie on the unit sphere.
    """
    points = x_flat.reshape(14, 3)
    cons = []
    for i in range(14):
        cons.append({'type': 'eq', 'fun': lambda x, i=i: np.dot(x[3*i:3*i+3], x[3*i:3*i+3]) - 1.0})
    return cons

def _optimize_sphere(x0, deadline, use_smooth=True, temperature=0.1):
    """
    Optimize starting from x0 on the sphere.
    """
    if time.time() > deadline:
        return x0.reshape(14, 3), 0.0
    
    # Normalize x0
    x0 = x0.reshape(14, 3)
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    x0 = (x0 / norms).flatten()
    
    cons = _constraints_sphere(x0)
    
    try:
        x_current = x0
        
        if use_smooth:
            # First pass with smooth surrogate
            res = minimize(
                lambda x: _score_and_grad_smooth(x, temperature)[0],
                x0,
                jac=lambda x: _score_and_grad_smooth(x, temperature)[1],
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            x_current = res.x
            
            # Second pass with exact objective
            res2 = minimize(
                lambda x: _score_and_grad_sphere(x)[0],
                x_current,
                jac=lambda x: _score_and_grad_sphere(x)[1],
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            x_current = res2.x
        
        # Third pass: high-precision polish with exact objective
        res3 = minimize(
            lambda x: _score_and_grad_sphere(x)[0],
            x_current,
            jac=lambda x: _score_and_grad_sphere(x)[1],
            method='SLSQP',
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-14}
        )
        
        cand = res3.x.reshape(14, 3)
        # Re-normalize to be safe
        norms = np.linalg.norm(cand, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        cand = cand / norms
        
        return cand, _raw_score(cand)
    except Exception:
        return x0.reshape(14, 3), 0.0

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
    
    # Optimize each guess
    for guess in guesses:
        if time.time() > deadline - 30:
            break
            
        pts, score = _optimize_sphere(guess, deadline, use_smooth=True, temperature=0.1)
        if score > best_score:
            best_score = score
            best_points = pts
            
    # If we have time, try a few more random starts
    while time.time() < deadline - 60:
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        pts_opt, score = _optimize_sphere(pts, deadline, use_smooth=True, temperature=0.05)
        if score > best_score:
            best_score = score
            best_points = pts_opt

    # Final polish pass on best points
    if time.time() < deadline - 10 and best_score > 0:
        try:
            best_points_final, final_score = _optimize_sphere(
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
