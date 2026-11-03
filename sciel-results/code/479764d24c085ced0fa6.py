"""Rewritten program: local optimization of logarithmic energy on the unit sphere."""
import time
import math
import numpy as np

N = 100
DEADLINE = 115.0  # safety margin before 120s kill

def _energy(X):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-9:
                return -np.inf
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_fast(X):
    """Vectorized energy for speed (not used for final scoring, but for gradient steps)."""
    n = X.shape[0]
    # pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    return np.sum(np.log(1.0 / d))

def _gradient(X):
    """
    Gradient of energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (since d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2, and E = -sum log(dist),
     so dE/dx_i = sum_j (x_i - x_j)/|x_i - x_j|^2)
    """
    n = X.shape[0]
    grad = np.zeros_like(X)
    for i in range(n):
        diff = X[i] - X  # shape (n, 3)
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = 1.0  # avoid div by zero
        # contribution: (x_i - x_j) / |x_i - x_j|^2
        contrib = diff / (dist ** 2)[:, np.newaxis]
        contrib[i] = 0.0
        grad[i] = np.sum(contrib, axis=0)
    return grad

def _project_to_sphere(X):
    """Normalize each point to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_descent_sphere(X0, max_iters=5000, lr0=0.01, deadline=None):
    """
    Gradient descent on the sphere using retraction:
    x_{k+1} = normalize(x_k - lr * grad_k)
    """
    X = X0.copy()
    best_X = X.copy()
    best_E = _energy_fast(X)
    lr = lr0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        grad = _gradient(X)
        # Retraction: move in negative gradient direction, then project
        X_new = X - lr * grad
        X_new = _project_to_sphere(X_new)
        
        E_new = _energy_fast(X_new)
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
        
        if E_new < best_E + 1e-10:
            X = X_new
        else:
            # Line search: reduce learning rate
            lr *= 0.5
            X_new = X - lr * grad
            X_new = _project_to_sphere(X_new)
            E_new = _energy_fast(X_new)
            if E_new < best_E + 1e-10:
                X = X_new
            else:
                lr *= 0.5
                X_new = X - lr * grad
                X_new = _project_to_sphere(X_new)
                E_new = _energy_fast(X_new)
                if E_new < best_E + 1e-10:
                    X = X_new
                else:
                    # Accept anyway to make progress
                    X = X_new
                    if E_new < best_E:
                        best_E = E_new
                        best_X = X.copy()
        
        # Adaptive learning rate
        if E_new < best_E:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.9, 1e-6)
    
    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Fibonacci spiral
    X0 = _fibonacci_sphere(N)
    X0 = _project_to_sphere(X0)
    
    best_X = X0.copy()
    best_E = _energy_fast(X0)
    
    # Run gradient descent from the Fibonacci start
    X, E = _gradient_descent_sphere(X0, max_iters=3000, lr0=0.005, deadline=deadline)
    if E < best_E:
        best_E = E
        best_X = X.copy()
    
    # Try a few different initializations and learning rates
    np.random.seed(42)
    for trial in range(3):
        if time.time() > deadline:
            break
        # Perturbed Fibonacci
        X_trial = X0 + 0.05 * np.random.randn(N, 3)
        X_trial = _project_to_sphere(X_trial)
        X, E = _gradient_descent_sphere(X_trial, max_iters=2000, lr0=0.003, deadline=deadline)
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    # Final polish: tighter gradient descent
    if time.time() < deadline - 5:
        X, E = _gradient_descent_sphere(best_X, max_iters=5000, lr0=0.001, deadline=deadline)
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N)
        best_X = _project_to_sphere(best_X)
    
    return best_X
