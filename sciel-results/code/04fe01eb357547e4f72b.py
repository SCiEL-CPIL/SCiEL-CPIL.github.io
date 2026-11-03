"""
Strategy: Simulated Annealing (SA) global search followed by local Riemannian polish.
The current program uses local Riemannian gradient descent which often gets stuck in local minima.
SA explores the configuration space more broadly by accepting worse solutions with a probability
that decreases over time, allowing it to escape local minima.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# Precompute pair indices
_idx_i, _idx_j = np.triu_indices(N, k=1)
_n_pairs = len(_idx_i)

def _energy_and_tangent_grad(X):
    """
    Compute energy and tangent-space gradient for logarithmic energy on sphere.
    Uses dot-product based distance: |xi-xj|^2 = 2 - 2*xi.xj
    X: (N, 3) array of unit vectors
    Returns: (energy, grad_tangent) where grad_tangent is (N,3)
    """
    # Compute all pairwise dot products
    dot = X @ X.T
    # dist_sq[i,j] = 2 - 2*dot[i,j]
    dist_sq = 2.0 - 2.0 * dot
    
    # Extract upper triangle (excluding diagonal)
    d2 = dist_sq[_idx_i, _idx_j]
    d2 = np.maximum(d2, 1e-18)
    
    # Energy = sum log(1/d) = -sum log(d) = -0.5 * sum log(d^2)
    energy = -0.5 * math.fsum(np.log(d2))
    
    # Gradient: dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    diff = X[_idx_i] - X[_idx_j]  # (n_pairs, 3)
    inv_d2 = 1.0 / d2  # (n_pairs,)
    
    grad_euclidean = np.zeros((N, 3))
    np.add.at(grad_euclidean, _idx_i, diff * inv_d2[:, np.newaxis])
    np.add.at(grad_euclidean, _idx_j, -diff * inv_d2[:, np.newaxis])
    
    # Project to tangent space
    dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
    grad_tangent = grad_euclidean - dot_grad * X
    
    return energy, grad_tangent

def _exp_map(X, V, step):
    """
    Exponential map on sphere: move each point along geodesic in direction V.
    X: (N, 3) unit vectors
    V: (N, 3) tangent vectors (already projected)
    step: scalar step size
    Returns: (N, 3) new unit vectors
    """
    cos_s = math.cos(step)
    sin_s = math.sin(step)
    X_new = cos_s * X + sin_s * V
    # Renormalize for numerical safety
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_new = X_new / norms
    return X_new

def _local_minimize_riemannian(X_init, deadline, max_iters=2000):
    """
    Pure Riemannian gradient descent with adaptive step size on the sphere.
    """
    X = X_init.copy()
    current_energy, grad_tan = _energy_and_tangent_grad(X)
    best_X = X.copy()
    best_energy = current_energy
    
    step_size = 0.05
    patience = 0
    max_patience = 50
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        if patience > max_patience:
            break
        
        # Compute step direction
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size
        step = step_size / (1.0 + grad_norm)
        step = min(step, 0.1)
        
        # Exponential map step
        V = -grad_tan / (grad_norm + 1e-12)
        X_new = _exp_map(X, V, step)
        
        E_new, grad_new = _energy_and_tangent_grad(X_new)
        
        if E_new < current_energy:
            X = X_new
            current_energy = E_new
            grad_tan = grad_new
            patience = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_X = X.copy()
        else:
            patience += 1
            step_size *= 0.9
    
    return best_X, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial configuration: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial energy
    current_energy, _ = _energy_and_tangent_grad(X)
    best_X = X.copy()
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T = 5.0  # Initial temperature (energy scale is ~ -1000, so T needs to be large enough to accept moves)
    # Actually, let's look at the energy scale.
    # Energy is sum log(1/d). d ~ 1/sqrt(N) ~ 0.1. log(10) ~ 2.3.
    # N pairs ~ 5000. Total energy ~ -5000 * 2.3 ~ -11500?
    # Wait, the target is -1083.
    # Let's re-evaluate energy scale.
    # If points are uniform, d ~ 1/sqrt(N).
    # log(1/d) ~ 0.5 log(N).
    # Sum ~ N^2 * 0.5 log(N).
    # For N=100, N^2=10000. 0.5 log(100) ~ 2.3.
    # 10000 * 2.3 = 23000?
    # The prompt says -1083.
    # Let's check the formula: sum_{i<j} log(1/|xi-xj|).
    # If d is small, log(1/d) is large positive.
    # Wait, "minimize the logarithmic energy... -1083".
    # If energy is negative, then log(1/d) must be negative?
    # log(1/d) = -log(d). If d < 1, log(d) < 0, so -log(d) > 0.
    # So energy should be positive?
    # "logarithmic energy, the sum over all pairs of log(1/distance)".
    # If distance < 1, 1/distance > 1, log > 0.
    # Why is the published energy -1083?
    # Maybe the distance is defined differently? Or maybe it's log(distance)?
    # "logarithmic energy... log(1/distance)".
    # If the points are on the unit sphere, max distance is 2.
    # If d is close to 2, 1/d is 0.5, log(0.5) is negative.
    # To minimize energy (make it very negative), we want d to be as large as possible (close to 2).
    # This corresponds to maximizing the minimum distance, i.e., spherical codes.
    # So we want points to be as far apart as possible.
    # The "repulsive" force is log(1/d).
    # Gradient of log(1/d) = -1/d * dd/dx.
    # d = |xi-xj|.
    # This is a standard Riesz s-energy with s=0 (logarithmic).
    # Usually Riesz energy is sum |xi-xj|^{-s}. For s=0, it's log.
    # Minimizing sum log(1/d) is equivalent to maximizing sum log(d).
    # This is the same as maximizing the volume of the configuration or spreading them out.
    
    # Let's check the sign in the code.
    # energy = -0.5 * math.fsum(np.log(d2))
    # d2 = d^2. log(d2) = 2 log(d).
    # energy = - sum log(d).
    # This matches sum log(1/d).
    # If d is small, log(d) is large negative, -log(d) is large positive.
    # If d is large (close to 2), log(d) is small positive (log 2 ~ 0.69), -log(d) is small negative.
    # So to minimize energy (make it more negative), we want d to be large.
    # This confirms we want to maximize distances.
    
    # Temperature T should be on the order of the energy change per step.
    # A single point move might change energy by ~ O(1) or O(10).
    # Let's start with T=10.0 and decay it.
    
    T = 10.0
    alpha = 0.999  # Cooling rate
    
    # Number of iterations per temperature step
    # We have ~145 seconds.
    # Each iteration:
    # 1. Pick random point
    # 2. Generate random tangent vector
    # 3. Move point
    # 4. Compute new energy (full recalc is O(N^2))
    # 5. Accept/Reject
    # 
    # O(N^2) = 10,000 ops. Very fast in NumPy.
    # We can probably do ~100,000 iterations in 145s?
    # Let's estimate: 10k ops * 100k iters = 1e9 ops.
    # NumPy can do ~1e8-1e9 simple ops/sec.
    # So 100k iterations might take ~10-100s.
    # Let's aim for ~50,000 iterations.
    
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() > deadline - 20.0:  # Leave 20s for final polish
            break
            
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Generate random tangent vector
        # Random vector in R3, project to tangent space at X[idx]
        v = rng.standard_normal(3)
        v_tan = v - np.dot(v, X[idx]) * X[idx]
        norm_v = np.linalg.norm(v_tan)
        if norm_v < 1e-10:
            continue
        v_tan = v_tan / norm_v
        
        # Step size: proportional to sqrt(T)
        step = 0.1 * math.sqrt(T)
        
        # Move the point
        X_new = X.copy()
        X_new[idx] = _exp_map(X_new[idx:idx+1], v_tan.reshape(1,3), step)[0]
        
        # Compute new energy
        E_new, _ = _energy_and_tangent_grad(X_new)
        
        # Metropolis criterion
        dE = E_new - current_energy
        if dE < 0 or rng.random() < math.exp(-dE / T):
            X = X_new
            current_energy = E_new
            if current_energy < best_energy:
                best_energy = current_energy
                best_X = X.copy()
        
        # Cool down
        T *= alpha
        
        # If T is very small, we can break early or reduce iterations
        if T < 0.01:
            break
    
    # Final local polish
    if time.time() < deadline - 5.0:
        X_refined, E_refined = _local_minimize_riemannian(best_X, deadline)
        if E_refined < best_energy:
            best_energy = E_refined
            best_X = X_refined
    
    # Final normalization
    if best_X is None:
        best_X = X
    
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    # Validate
    if not np.all(np.isfinite(best_X)):
        # Fallback: Fibonacci sphere
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = math.pi * (1.0 + math.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_X
