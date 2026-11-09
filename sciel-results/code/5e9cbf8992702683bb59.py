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

def _spherical_to_cartesian(theta_phi):
    """
    Convert spherical coordinates (theta, phi) for 14 points to Cartesian.
    theta: polar angle (0 to pi)
    phi: azimuthal angle (0 to 2*pi)
    theta_phi: shape (14, 2)
    Returns: shape (14, 3)
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """
    Convert Cartesian coordinates to spherical (theta, phi).
    points: shape (14, 3)
    Returns: shape (14, 2)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2*pi)
    phi = phi % (2 * np.pi)
    return np.column_stack([theta, phi])

def _score_spherical(theta_phi):
    """
    Score function for spherical coordinates.
    theta_phi: shape (14, 2)
    Returns: negative of (d_min/d_max)^2 for minimization
    """
    points = _spherical_to_cartesian(theta_phi)
    score = _raw_score(points)
    return -score

def _repulsion_simulate(points, n_iters=500, deadline=None):
    """
    Simple repulsion-based simulation to spread points on sphere.
    """
    points = points.copy()
    n = len(points)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = points[i] - points[j]
                d = np.linalg.norm(diff)
                if d > 1e-10:
                    # Repulsive force inversely proportional to distance
                    force = diff / (d * d)
                    forces[i] += force
        
        # Normalize forces
        for i in range(n):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-10:
                forces[i] = forces[i] / f_norm
        
        # Update points
        step_size = 0.1 * (1.0 - it / n_iters) + 0.01
        for i in range(n):
            points[i] = points[i] + step_size * forces[i]
            # Project back to sphere
            norm = np.linalg.norm(points[i])
            if norm > 1e-10:
                points[i] = points[i] / norm
    
    return points

def _optimize_spherical(theta_phi0, deadline, maxiter=2000):
    """
    Optimize in spherical coordinates using Nelder-Mead.
    """
    if time.time() > deadline:
        return theta_phi0, 0.0
    
    theta_phi0 = theta_phi0.copy()
    
    try:
        res = minimize(
            _score_spherical,
            theta_phi0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        theta_phi_opt = res.x
        points = _spherical_to_cartesian(theta_phi_opt)
        
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        return theta_phi_opt, _raw_score(points)
    except Exception:
        return theta_phi0, 0.0

def _optimize_spherical_polish(theta_phi0, deadline):
    """
    High-precision polish using multiple Nelder-Mead passes with tighter tolerances.
    """
    if time.time() > deadline:
        return theta_phi0, 0.0
    
    current_tp = theta_phi0.copy()
    current_score = _raw_score(_spherical_to_cartesian(current_tp))
    
    # Multiple polish passes with decreasing tolerances
    tolerances = [(1e-8, 1e-10), (1e-10, 1e-12), (1e-12, 1e-14)]
    
    for xatol, fatol in tolerances:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _score_spherical,
                current_tp,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
            )
            
            new_tp = res.x
            new_score = _raw_score(_spherical_to_cartesian(new_tp))
            
            if new_score > current_score:
                current_tp = new_tp
                current_score = new_score
        except Exception:
            pass
    
    points = _spherical_to_cartesian(current_tp)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    return current_tp, _raw_score(points)

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
    best_theta_phi = np.zeros((14, 2))
    
    rng = np.random.default_rng(42)
    
    # Generate initial guesses in Cartesian, then convert to spherical
    guesses_cart = []
    
    # 1. Icosahedron + 2 points
    ico = _generate_icosahedron()
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    
    p1 = rng.normal(size=3)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=3)
    p2 = p2 / np.linalg.norm(p2)
    
    guess1 = np.vstack([ico, p1, p2])
    guesses_cart.append(guess1)
    
    # 2. Fibonacci sphere
    fib = _generate_fibonacci_sphere(14)
    guesses_cart.append(fib)
    
    # 3. Random points on sphere
    for _ in range(12):
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        guesses_cart.append(pts)
        
    # 4. Cubic points
    cube = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(cube, axis=1, keepdims=True)
    cube = cube / norms
    guesses_cart.append(cube)
    
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
    guesses_cart.append(octa)
    
    # Apply repulsion simulation to each guess, then optimize
    for guess_cart in guesses_cart:
        if time.time() > deadline - 30:
            break
        
        # Repulsion simulation to spread points
        repelled = _repulsion_simulate(guess_cart, n_iters=300, deadline=deadline - 15)
        
        # Convert to spherical
        theta_phi = _cartesian_to_spherical(repelled)
        
        # Optimize in spherical coordinates
        tp_opt, score = _optimize_spherical(theta_phi, deadline - 10, maxiter=1500)
        
        if score > best_score:
            best_score = score
            best_points = _spherical_to_cartesian(tp_opt)
            best_theta_phi = tp_opt
    
    # Polish the best result
    if time.time() < deadline - 10 and best_score > 0:
        tp_polished, polished_score = _optimize_spherical_polish(best_theta_phi, deadline - 5)
        if polished_score > best_score:
            best_score = polished_score
            best_points = _spherical_to_cartesian(tp_polished)
            best_theta_phi = tp_polished
    
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
