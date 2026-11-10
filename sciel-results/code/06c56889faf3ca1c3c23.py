"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Deterministic global RNG so construct() is reproducible within a run.
_RNG = np.random.default_rng(12345)


def _pairwise_chordal_sq(pts):
    """Return (N, N) matrix of squared chordal distances."""
    G = pts @ pts.T
    return np.maximum(2.0 - 2.0 * G, 0.0)


def _log_energy(pts):
    """Compute log energy sum_{i<j} log(1/d_ij). Vectorized, numerically safe."""
    D2 = _pairwise_chordal_sq(pts)
    iu = np.triu_indices(N, k=1)
    d2 = D2[iu]
    d2 = np.maximum(d2, 1e-30)
    return float(np.sum(-0.5 * np.log(d2)))


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _riemannian_grad(pts):
    """
    Compute the Riemannian gradient of the log energy on the sphere.
    grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    This vector is already tangent to the sphere at x_i because x_i dot (x_i - x_j) = 1 - x_i.dot(x_j) 
    and the sum of these terms projected onto x_i is 0? 
    Actually, the Euclidean gradient g_i = sum_{j != i} (x_i - x_j)/d2_ij.
    The Riemannian gradient is the projection of g_i onto the tangent space at x_i:
    grad_riem_i = g_i - (g_i . x_i) x_i
    """
    D2 = _pairwise_chordal_sq(pts)
    D2_safe = np.where(np.eye(N) > 0, 1.0, D2)
    # M[i,j] = (x_i - x_j) / D2[i,j]
    M = (pts[:, None, :] - pts[None, :, :]) / D2_safe[:, :, None]
    M[np.diag_indices(N)] = 0.0
    # Euclidean gradient
    g = np.sum(M, axis=1)
    # Project to tangent space: g - (g . x) x
    dot = np.einsum('ij,ij->i', g, pts)
    grad_riem = g - dot[:, None] * pts
    return grad_riem


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    global _RNG
    _RNG = np.random.default_rng(12345)
    
    # Deadline guard: 150s limit, aim for 140s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_E = _log_energy(pts)
    
    # Riemannian Gradient Descent with Momentum and Adaptive Step Size
    # We use a simple momentum-based update:
    # v = beta * v + (1 - beta) * grad
    # x_new = x - lr * v
    # x_new = normalize(x_new)
    
    lr = 0.01
    beta = 0.9
    v = np.zeros_like(pts)
    
    # Phase 1: Steady gradient descent
    while time.time() < deadline - 20.0:
        grad = _riemannian_grad(pts)
        
        # Adaptive step size: if energy increases significantly, reduce lr
        # Simple heuristic: monitor norm of gradient
        grad_norm = np.linalg.norm(grad)
        
        v = beta * v + (1 - beta) * grad
        new_pts = pts - lr * v
        new_pts = _normalize(new_pts)
        
        new_E = _log_energy(new_pts)
        
        if new_E < best_E:
            best_E = new_E
            best_pts = new_pts.copy()
            
        if new_E < _log_energy(pts): # Accept if better than current
            pts = new_pts
            # Slightly increase lr if we are making good progress and lr is small
            if lr < 0.05:
                lr *= 1.01
        else:
            # Reject, reduce lr
            lr *= 0.9
            # Revert momentum to avoid getting stuck
            v *= 0.5
            
        # Decay learning rate slowly over time
        # This is a heuristic, we can also just keep it constant or decay linearly
        # Let's add a slow decay
        # elapsed = time.time() - start_time
        # lr = max(0.001, lr * 0.999)
        
    # Phase 2: Stochastic restarts to escape local minima
    # Use remaining time to try a few random restarts with short GD runs
    remaining_time = deadline - time.time()
    n_restarts = int(remaining_time / 10.0) # ~10s per restart
    
    for _ in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        pts_rand = _RNG.normal(0, 1, size=(N, 3))
        pts_rand = _normalize(pts_rand)
        
        # Short gradient descent
        lr_r = 0.01
        v_r = np.zeros_like(pts_rand)
        current_pts = pts_rand.copy()
        
        # Run for a few seconds
        restart_start = time.time()
        while time.time() - restart_start < 5.0:
            grad = _riemannian_grad(current_pts)
            v_r = beta * v_r + (1 - beta) * grad
            new_pts = current_pts - lr_r * v_r
            new_pts = _normalize(new_pts)
            
            new_E = _log_energy(new_pts)
            if new_E < _log_energy(current_pts):
                current_pts = new_pts
                if new_E < best_E:
                    best_E = new_E
                    best_pts = new_pts.copy()
            else:
                lr_r *= 0.9
                v_r *= 0.5
            
            # Break if converged
            if np.linalg.norm(grad) < 1e-6:
                break
                
    # Final check
    final_E = _log_energy(best_pts)
    if final_E < best_E:
        best_E = final_E

    return best_pts
# EVOLVE-BLOCK-END
