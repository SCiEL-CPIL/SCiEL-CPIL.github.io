"""Improved program: minimize squared max/min pairwise distance ratio for N=14 in 3D."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_dists_sq(pts):
    """Return matrix of squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)

def _score(pts):
    """Compute (max_dist / min_dist)^2. Returns inf if any points coincide."""
    d2 = _pairwise_dists_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    d2_off = d2[mask]
    if d2_off.size == 0:
        return np.inf
    min_d2 = np.min(d2_off)
    max_d2 = np.max(d2_off)
    if min_d2 <= 1e-12:
        return np.inf
    ratio = np.sqrt(max_d2 / min_d2)
    return ratio * ratio

def _normalize(pts):
    """Center at origin and scale so RMS distance from origin is 1."""
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-9:
        return pts
    return pts / rms

def _objective(x):
    """Objective for optimizer: normalized squared ratio."""
    pts = x.reshape(N, D)
    pts = _normalize(pts)
    return _score(pts)

def _generate_icosahedron():
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    # Standard icosahedron vertices
    verts = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ]
    verts = np.array(verts, dtype=float)
    verts = _normalize(verts)
    return verts

def _generate_initial_guesses():
    """Generate diverse initial configurations."""
    guesses = []
    
    # 1. Icosahedron + 2 random points
    ico = _generate_icosahedron()
    for _ in range(5):
        extra = np.random.randn(2, D)
        extra = _normalize(extra)
        pts = np.vstack([ico, extra])
        guesses.append(pts)
        
    # 2. Random points on sphere
    for _ in range(5):
        pts = np.random.randn(N, D)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        guesses.append(pts)
        
    # 3. Fibonacci sphere (approximation)
    i = np.arange(N)
    phi_fib = np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * i / (N - 1)
    r = np.sqrt(1 - z**2)
    theta = phi_fib * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y, z))
    guesses.append(pts)
    
    return guesses

def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    inits = _generate_initial_guesses()
    
    # Shuffle inits to ensure diversity
    np.random.shuffle(inits)
    
    for init in inits:
        if time.time() > deadline - 15:
            break
            
        try:
            # Use Nelder-Mead for robustness to non-smoothness (min/max switch)
            # L-BFGS-B is faster but can get stuck if gradient is zero due to symmetry or flat regions.
            # Nelder-Mead is derivative-free and handles the kinks better.
            res = minimize(
                _objective,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, D)
                
                # Optional: Polish with L-BFGS-B if Nelder-Mead gets close
                if best_score < 20.0 and time.time() < deadline - 20:
                     res2 = minimize(
                        _objective,
                        best_pts.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                     if res2.fun < best_score:
                         best_score = res2.fun
                         best_pts = res2.x.reshape(N, D)
                         
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = np.random.randn(N, D)
        best_pts = _normalize(best_pts)
    
    return best_pts
