"""Minimal logarithmic energy of 150 points on the sphere."""
import numpy as np
import time
import math

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and gradient w.r.t. X (N x 3).
    """
    # Pairwise distances
    # |x_i - x_j|^2 = 2 - 2 x_i . x_j
    G = X @ X.T  # N x N
    np.fill_diagonal(G, 0.0)
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 1.0)  # avoid log(0) on diagonal
    d = np.sqrt(np.maximum(d2, 1e-30))

    # Energy: -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(d[iu[0], iu[1]]))

    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Since dE/dx_i = -sum_{j != i} d(log d_ij)/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # Actually E = -sum log(d_ij), dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum (x_i-x_j)/d_ij^2
    # But we want to MINIMIZE E, so gradient points in direction of increase.
    # For descent we use -gradient.
    # grad[i] = -sum_{j != i} (x_i - x_j) / d_ij^2
    #         = sum_{j != i} (x_j - x_i) / d_ij^2

    # Vectorized: for each i, sum over j of (x_j - x_i) / d_ij^2
    # d_ij^2 = d2[i,j]
    # (x_j - x_i) is a 3-vector
    # grad[i] = sum_j (X[j] - X[i]) / d2[i,j]
    #         = (sum_j X[j]/d2[i,j]) - X[i] * sum_j 1/d2[i,j]

    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    np.fill_diagonal(d2, 1.0)  # avoid div by zero
    inv_d2 = 1.0 / np.maximum(d2, 1e-30)
    np.fill_diagonal(inv_d2, 0.0)

    # sum_j X[j] / d2[i,j] = (inv_d2 @ X)
    sum_x = inv_d2 @ X  # N x 3
    sum_inv = inv_d2.sum(axis=1)  # N

    grad = sum_x - X * sum_inv[:, np.newaxis]

    return energy, grad


def _project_to_sphere(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _halton_init():
    """Initialize N points using a 3D Halton sequence mapped to the sphere."""
    def halton_sequence(n, base):
        seq = np.zeros(n)
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            j = i
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            seq[i - 1] = r
        return seq

    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)

    # Map [0,1]^3 to sphere using spherical coordinates
    # Use a better mapping: uniform on sphere via z = 2*h3 - 1, theta = 2*pi*h1
    z = 2.0 * h3 - 1.0
    theta = 2.0 * np.pi * h1
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _project_to_sphere(X)


def _fibonacci_init():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(theta), z])
    return _project_to_sphere(X)


def _riemannian_cg(X, max_iter=200, tol=1e-10):
    """
    Riemannian conjugate gradient on the sphere.
    Uses exponential map: Exp_x(v) = x cos(|v|) + (v/|v|) sin(|v|)
    """
    X = _project_to_sphere(X.copy())
    E, G = _energy_and_grad(X)
    
    # Initial search direction: negative projected gradient
    # Project gradient to tangent space: G_t = G - (G . x) x
    G_t = G - (G * X).sum(axis=1, keepdims=True) * X
    p = -G_t
    
    E_prev = E
    converged = False
    
    for it in range(max_iter):
        # Check time
        if time.time() > DEADLINE - 5.0:
            break
        
        # Line search along Riemannian direction
        # Try step sizes
        best_E = E
        best_X = X
        best_alpha = 0.0
        
        for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
            # Move along direction p by alpha
            # X_new = Exp_x(alpha * p)
            p_norm = np.linalg.norm(alpha * p, axis=1, keepdims=True)
            p_norm = np.maximum(p_norm, 1e-15)
            # Exp_x(v) = x * cos(|v|) + (v/|v|) * sin(|v|)
            v = alpha * p
            v_norm = np.linalg.norm(v, axis=1, keepdims=True)
            v_norm_safe = np.maximum(v_norm, 1e-15)
            cos_term = np.cos(v_norm_safe)
            sin_term = np.sin(v_norm_safe) / v_norm_safe
            X_new = X * cos_term + v * sin_term
            X_new = _project_to_sphere(X_new)
            
            E_new, _ = _energy_and_grad(X_new)
            if E_new < best_E:
                best_E = E_new
                best_X = X_new
                best_alpha = alpha
        
        if best_alpha == 0.0:
            # Try smaller steps
            for alpha in [0.001, 0.005]:
                v = alpha * p
                v_norm = np.linalg.norm(v, axis=1, keepdims=True)
                v_norm_safe = np.maximum(v_norm, 1e-15)
                cos_term = np.cos(v_norm_safe)
                sin_term = np.sin(v_norm_safe) / v_norm_safe
                X_new = X * cos_term + v * sin_term
                X_new = _project_to_sphere(X_new)
                E_new, _ = _energy_and_grad(X_new)
                if E_new < best_E:
                    best_E = E_new
                    best_X = X_new
                    best_alpha = alpha
        
        if best_alpha == 0.0:
            # Stuck, restart with steepest descent
            X = _project_to_sphere(X)
            E, G = _energy_and_grad(X)
            G_t = G - (G * X).sum(axis=1, keepdims=True) * X
            p = -G_t
            continue
        
        X = best_X
        E = best_E
        
        # Check convergence
        if abs(E_prev - E) < tol:
            converged = True
            break
        E_prev = E
        
        # Compute new gradient
        _, G_new = _energy_and_grad(X)
        G_t_new = G_new - (G_new * X).sum(axis=1, keepdims=True) * X
        
        # Beta (Polak-Ribiere)
        G_t_old = G - (G * X).sum(axis=1, keepdims=True) * X
        beta_num = np.sum(G_t_new * (G_t_new - G_t_old), axis=1)
        beta_den = np.sum(G_t_old * G_t_old, axis=1)
        beta = np.where(beta_den > 1e-15, np.maximum(beta_num / beta_den, 0.0), 0.0)
        
        p = -G_t_new + beta[:, np.newaxis] * p
        
        # Check if we should restart (if beta is too large or direction is bad)
        if np.any(beta > 1.0):
            p = -G_t_new
        
        G = G_new
    
    return X, E


def _cartesian_gradient_descent(X, max_iter=100, lr=0.001):
    """Simple projected gradient descent in Cartesian space."""
    X = _project_to_sphere(X.copy())
    E, G = _energy_and_grad(X)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        
        X_new = X - lr * G
        X_new = _project_to_sphere(X_new)
        E_new, G_new = _energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            G = G_new
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return X, E


def construct():
    start_time = time.time()
    
    # Try multiple initializations and pick the best
    best_X = None
    best_E = np.inf
    
    # Initialization 1: Halton sequence
    X1 = _halton_init()
    E1, _ = _energy_and_grad(X1)
    
    # Initialization 2: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X2 = _project_to_sphere(X2)
    E2, _ = _energy_and_grad(X2)
    
    candidates = [(X1, E1), (X2, E2)]
    
    # Sort by initial energy
    candidates.sort(key=lambda x: x[1])
    
    for X_init, E_init in candidates:
        if time.time() > DEADLINE - 10.0:
            break
        
        # Phase 1: Riemannian CG for global-ish optimization
        X_opt, E_opt = _riemannian_cg(X_init, max_iter=100, tol=1e-9)
        
        # Phase 2: Fine-tune with more Riemannian CG iterations
        if time.time() < DEADLINE - 10.0:
            X_opt, E_opt = _riemannian_cg(X_opt, max_iter=200, tol=1e-10)
        
        # Phase 3: Final polish with very tight tolerance
        if time.time() < DEADLINE - 5.0:
            X_opt, E_opt = _riemannian_cg(X_opt, max_iter=100, tol=1e-11)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # Final validation
    best_X = _project_to_sphere(best_X)
    final_E, _ = _energy_and_grad(best_X)
    
    # Check for any points too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_X[i] - best_X[j]) < 1e-9:
                # Nudge apart
                best_X[j] = best_X[j] + 1e-8 * np.random.randn(3)
                best_X = _project_to_sphere(best_X)
    
    return best_X


# Ensure the fibonacci_init function is correct (fix the bug)
def _fibonacci_init():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(X)
