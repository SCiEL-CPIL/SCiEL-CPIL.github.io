"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100
DEADLINE = 150.0  # seconds, evaluator kills at this point


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|xi - xj|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Gradient of log energy w.r.t. point positions (in R^3)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dxi log(1/d) = -1/d * d/dxi d = -1/d * (xi-xj)/d = -(xi-xj)/d^2
            # d/dxj log(1/d) = +1/d * d/dxj d = 1/d * (xj-xi)/d = (xj-xi)/d^2 = -(xi-xj)/d^2 * (-1)... 
            # Let's redo: f = log(1/d) = -log(d)
            # df/dxi = -1/d * d(d)/dxi = -1/d * (xi-xj)/d = -(xi-xj)/d^2
            # df/dxj = -1/d * d(d)/dxj = -1/d * (xj-xi)/d = (xi-xj)/d^2
            coeff = 1.0 / (d * d)
            grad[i] -= coeff * diff
            grad[j] += coeff * diff
    return grad


def _project_to_sphere(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _golden_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _conjugate_gradient_sphere(pts, max_iter=500, tol=1e-14):
    """
    Conjugate gradient on the sphere manifold using tangent space projection.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    for it in range(max_iter):
        g = _log_energy_grad(pts)
        
        # Project gradient to tangent space: g_t = g - (g·x)x
        g_t = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        
        # Check convergence
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Use steepest descent with line search on sphere (simplified)
        # Actually, let's do a proper CG-like approach or just gradient descent with adaptive step
        
        # Adaptive step size
        step = 1.0 / (g_norm + 1e-10)
        
        # Try a few step sizes
        for _ in range(20):
            new_pts = pts - step * g_t
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts.copy()
                pts = new_pts
                break
            step *= 0.5
        else:
            break
        
        if time.time() > DEADLINE - 5.0:
            break
    
    return best_pts, best_e


def _firefly_algorithm(pts, max_iter=200, n_particles=10, beta0=0.5, gamma=1.0):
    """
    Firefly-inspired local search: particles attract toward better neighbors.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    # Initialize particles
    particles = [pts.copy() for _ in range(n_particles)]
    energies = [_log_energy(p) for p in particles]
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Sort by energy
        order = np.argsort(energies)
        
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                if energies[j] < energies[i]:
                    # Particle i is attracted to j
                    diff = particles[j] - particles[i]
                    dist = np.linalg.norm(diff, axis=1)
                    # Move toward j with decay
                    move = beta0 * np.exp(-gamma * dist**2) * diff
                    # Add some randomness
                    move += 0.01 * np.random.randn(n, 3)
                    new_p = particles[i] + move
                    new_p = _project_to_sphere(new_p)
                    new_e = _log_energy(new_p)
                    if new_e < energies[i]:
                        particles[i] = new_p
                        energies[i] = new_e
        
        # Track global best
        min_idx = np.argmin(energies)
        if energies[min_idx] < best_e:
            best_e = energies[min_idx]
            best_pts = particles[min_idx].copy()
        
        # Decay beta
        beta0 *= 0.99
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    
    # Start with golden spiral
    pts = _golden_spiral()
    pts = _project_to_sphere(pts)
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    # Phase 1: Conjugate gradient / gradient descent on sphere
    if time.time() - start_time < DEADLINE - 30:
        pts, e = _conjugate_gradient_sphere(pts, max_iter=300, tol=1e-13)
        if e < best_e:
            best_pts = pts
            best_e = e
    
    # Phase 2: Firefly algorithm for further refinement
    if time.time() - start_time < DEADLINE - 15:
        pts, e = _firefly_algorithm(pts, max_iter=100, n_particles=8)
        if e < best_e:
            best_pts = pts
            best_e = e
    
    # Phase 3: Another round of gradient descent from the best point
    if time.time() - start_time < DEADLINE - 5:
        pts, e = _conjugate_gradient_sphere(best_pts, max_iter=200, tol=1e-15)
        if e < best_e:
            best_pts = pts
            best_e = e
    
    # Final check
    final_pts = _project_to_sphere(best_pts)
    final_e = _log_energy(final_pts)
    
    # Ensure validity
    if not np.all(np.isfinite(final_pts)):
        return _golden_spiral()
    
    # Check minimum distance
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(final_pts[i] - final_pts[j]) < 1e-9:
                return _golden_spiral()
    
    return final_pts
