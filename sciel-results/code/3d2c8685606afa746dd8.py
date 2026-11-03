import numpy as np
import time
import scipy.optimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms

def _covering_radius(points):
    """
    Compute the covering radius by checking all Voronoi vertices (triple points).
    For each triple (i,j,k), the Voronoi vertex on the sphere is along the
    cross product of (points[i]-points[j]) and (points[i]-points[k]), or more
    precisely, it's the point on the sphere equidistant to the three points.
    
    Actually, a better approach: the covering radius is the maximum over all
    points y on the sphere of min_i angular_dist(y, x_i). This maximum occurs
    at Voronoi vertices, which are equidistant to 3 or more points.
    
    For each triple (i,j,k), find the point on the sphere that is equidistant
    to all three. This is the intersection of the sphere with the line through
    the circumcenter of triangle (i,j,k).
    """
    points = _normalize(points)
    n = len(points)
    max_dist = 0.0
    
    # Check all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                pi, pj, pk = points[i], points[j], points[k]
                
                # The Voronoi vertex is equidistant to pi, pj, pk
                # It lies along the direction: (pi x pj) x (pi x pk) ... 
                # Actually, the point equidistant to pi, pj, pk on the sphere
                # satisfies: y.pi = y.pj = y.pk
                # This means y is orthogonal to (pi - pj) and (pi - pk)
                # So y is parallel to (pi - pj) x (pi - pk)
                
                v1 = pi - pj
                v2 = pi - pk
                cross = np.cross(v1, v2)
                norm_c = np.linalg.norm(cross)
                
                if norm_c < 1e-10:
                    continue
                
                y1 = cross / norm_c
                y2 = -y1  # Check both directions
                
                for y in [y1, y2]:
                    dots = points @ y
                    dots = np.clip(dots, -1.0, 1.0)
                    dists = np.arccos(dots)
                    min_d = np.min(dists)
                    if min_d > max_dist:
                        max_dist = min_d
    
    # Also check the points themselves (degenerate case)
    # Actually the Voronoi vertices should cover it
    
    return np.degrees(max_dist)

def _fast_covering_radius(points, grid_size=2000):
    """Fast estimate using Fibonacci grid."""
    points = _normalize(points)
    M = grid_size
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    G = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = G @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    return np.degrees(np.max(min_dists))

def _riesz_energy_and_grad(v, s=12.0):
    """Compute Riesz s-energy and its gradient for points on the sphere."""
    X = v.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    P = X / norms
    
    dot = P @ P.T
    dist2 = 2.0 - 2.0 * dot
    np.fill_diagonal(dist2, 1.0)
    
    mask = ~np.eye(N, dtype=bool)
    dist2_safe = np.where(mask, dist2, 1.0)
    
    energy_terms = dist2_safe ** (-s / 2.0)
    energy_terms[~mask] = 0.0
    energy = np.sum(energy_terms)
    
    coeff = np.where(mask, -s * (dist2_safe) ** (-(s + 2) / 2.0), 0.0)
    S = np.sum(coeff, axis=1)
    G_P = P * S[:, None] - (coeff @ P)
    
    term1 = G_P / norms
    dots_v = np.sum(v.reshape(N, 3) * G_P, axis=1)
    term2 = v.reshape(N, 3) * dots_v[:, None] / (norms ** 3)
    G_v = term1 - term2
    
    return energy, G_v.flatten()

def _soft_covering_radius(points, temperature=0.5):
    """Smooth approximation of the covering radius using log-sum-exp."""
    points = _normalize(points)
    # Use a moderate grid
    M = 500
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    G = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = G @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Log-sum-exp of exp(-dists/temperature)
    min_dists = np.min(dists, axis=1)
    # Smooth max: temperature * log(sum(exp(dists/temperature)))
    max_val = np.max(min_dists)
    shifted = min_dists - max_val
    log_sum = max_val + temperature * np.log(np.sum(np.exp(shifted / temperature)))
    return log_sum

def _soft_covering_radius_grad(points, temperature=0.5):
    """Gradient of the soft covering radius."""
    points = _normalize(points)
    M = 500
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    G = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = G @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    
    # For each grid point, the nearest point contributes to the gradient
    # Gradient of arccos(y . x_i) w.r.t x_i = -y / sqrt(1 - (y . x_i)^2)
    
    grad = np.zeros((N, 3))
    for m in range(M):
        ni = nearest[m]
        y = G[m]
        dot_val = dots[m, ni]
        denom = np.sqrt(max(1.0 - dot_val * dot_val, 1e-10))
        grad[ni] += -y / denom
    
    return grad

def _velocity_verlet_refine(points, steps=50, dt=0.05, s=12.0):
    """Velocity Verlet integration for Riesz energy minimization."""
    points = _normalize(points.copy())
    
    def force(P):
        dot = P @ P.T
        dist2 = 2.0 - 2.0 * dot
        np.fill_diagonal(dist2, 1.0)
        mask = ~np.eye(N, dtype=bool)
        dist2_safe = np.where(mask, dist2, 1.0)
        coeff = np.where(mask, -s * (dist2_safe) ** (-(s + 2) / 2.0), 0.0)
        S = np.sum(coeff, axis=1)
        G_P = P * S[:, None] - (coeff @ P)
        # Project to tangent space
        dots = np.sum(G_P * P, axis=1, keepdims=True)
        G_P_tangent = G_P - dots * P
        return G_P_tangent
    
    v = np.zeros((N, 3))
    F = force(points)
    
    for step in range(steps):
        if time.time() > START_TIME + TIME_LIMIT:
            break
        # Half step velocity
        v += 0.5 * dt * F
        # Update position
        points = points + dt * v
        points = _normalize(points)
        # Recompute force
        F = force(points)
        # Half step velocity
        v += 0.5 * dt * F
        # Project velocity to tangent space
        dots = np.sum(v * points, axis=1, keepdims=True)
        v = v - dots * points
    
    return points

def _lbfgs_refine(points, maxiter=200):
    """Refine points using L-BFGS-B on the Riesz energy."""
    v0 = points.flatten()
    
    try:
        res = scipy.optimize.minimize(
            lambda v: _riesz_energy_and_grad(v, s=12.0)[0],
            v0,
            jac=lambda v: _riesz_energy_and_grad(v, s=12.0)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300}
        )
        if res.fun < float('inf'):
            return _normalize(res.x.reshape(N, 3))
    except Exception:
        pass
    return points

def _simulated_annealing(points, n_iter=1000, T_start=0.5, T_end=0.01):
    """Simulated annealing to escape local minima."""
    points = _normalize(points.copy())
    current_energy = _riesz_energy_and_grad(points.flatten(), s=12.0)[0]
    
    for i in range(n_iter):
        if time.time() > START_TIME + TIME_LIMIT:
            break
        
        T = T_start * (T_end / T_start) ** (i / max(n_iter - 1, 1))
        
        # Random perturbation
        new_points = points.copy()
        idx = np.random.randint(N)
        noise = np.random.randn(3) * 0.1
        new_points[idx] += noise
        new_points = _normalize(new_points)
        
        new_energy = _riesz_energy_and_grad(new_points.flatten(), s=12.0)[0]
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            points = new_points
            current_energy = new_energy
    
    return points

def _fibonacci_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points():
    """Generate random points on the sphere."""
    v = np.random.randn(N, 3)
    return _normalize(v)

def _icosahedron_based_points():
    """Generate points based on icosahedron symmetry."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ])
    verts = _normalize(verts)
    
    # Create 50 points by subdividing
    points = [verts]
    # Add more points by interpolating and normalizing
    for i in range(N - 12):
        a, b = np.random.randint(12, size=2)
        while a == b:
            b = np.random.randint(12)
        mid = (verts[a] + verts[b]) / 2.0
        mid = _normalize(mid.reshape(1, 3))[0]
        # Add with small perturbation
        noise = np.random.randn(3) * 0.1
        new_pt = _normalize((mid + noise).reshape(1, 3))[0]
        points.append(new_pt)
    
    return np.array(points[:N])

def construct():
    best_points = None
    best_radius = float('inf')
    
    # Try multiple strategies
    strategies = [
        ('fibonacci', _fibonacci_points()),
        ('random', _random_points()),
        ('icosahedron', _icosahedron_based_points()),
    ]
    
    for name, init_points in strategies:
        if time.time() > START_TIME + TIME_LIMIT:
            break
        
        points = init_points.copy()
        
        # Step 1: Velocity Verlet refinement
        points = _velocity_verlet_refine(points, steps=100, dt=0.03, s=12.0)
        
        # Step 2: Simulated annealing
        points = _simulated_annealing(points, n_iter=500, T_start=0.3, T_end=0.01)
        
        # Step 3: L-BFGS refinement
        points = _lbfgs_refine(points, maxiter=200)
        
        # Evaluate
        radius = _fast_covering_radius(points, grid_size=2000)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Additional L-BFGS polishing on the best result
    if best_points is not None and time.time() < START_TIME + TIME_LIMIT - 10:
        best_points = _lbfgs_refine(best_points, maxiter=300)
        best_radius = _fast_covering_radius(best_points, grid_size=2000)
    
    if best_points is None:
        best_points = _fibonacci_points()
    
    return _normalize(best_points)
