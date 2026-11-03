"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(centres):
    """Compute the packing score (largest common radius)."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    dist_boundary = 0.5 - np.abs(centres)
    min_boundary = np.min(dist_boundary)
    if min_boundary < -1e-12:
        return 0.0
    d = pdist(centres)
    min_pair = np.min(d)
    r = min(min_pair / 2.0, min_boundary)
    return r

def _soft_min(d, beta):
    """Smooth min via log-sum-exp."""
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta

def _objective_and_grad(x, beta):
    """
    Objective with auxiliary variable t (epigraph formulation).
    Minimize -t subject to t <= min_pair/2 and t <= min_face.
    Using soft constraints: penalty for violations.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    i, j = np.triu_indices(N, 1)
    diff = c[i] - c[j]
    dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
    
    # Soft min of pairwise distances
    min_pair_soft = _soft_min(dist, beta)
    
    # Boundary distances
    face_dists = 0.5 - np.abs(c)
    min_face_soft = _soft_min(face_dists, beta)
    
    # The radius is min(min_pair/2, min_face)
    r_soft = _soft_min(np.array([min_pair_soft / 2.0, min_face_soft]), beta * 0.5)
    
    value = -r_soft
    
    # Gradient computation
    # For soft_min(d, beta) = d_min - log(sum(exp(-beta*(d-d_min))))/beta
    # grad = -sum(exp(-beta*(d-d_min)) / sum(exp(...)) * grad_d) / beta * beta = -weights * grad_d
    # where weights = exp(-beta*(d-d_min)) / sum(exp(...))
    
    # Gradient of soft_min of pairwise distances
    d_min = np.min(dist)
    weights = np.exp(-beta * (dist - d_min))
    weights /= np.sum(weights)
    
    grad_pair = np.zeros((N, 3))
    for k in range(len(i)):
        w = weights[k]
        grad_pair[i[k]] -= w * diff[k] / dist[k]
        grad_pair[j[k]] += w * diff[k] / dist[k]
    
    # Gradient of soft_min of face distances
    f_min = np.min(face_dists)
    weights_f = np.exp(-beta * (face_dists - f_min))
    weights_f /= np.sum(weights_f)
    
    grad_face = np.zeros((N, 3))
    for n in range(N):
        for k in range(3):
            grad_face[n, k] = -weights_f[n, k] * np.sign(c[n, k])
    
    # Gradient of outer soft_min: r_soft = soft_min([a, b]) where a=min_pair/2, b=min_face
    a = min_pair_soft / 2.0
    b = min_face_soft
    outer_beta = beta * 0.5
    outer_min = min(a, b)
    wa = np.exp(-outer_beta * (a - outer_min))
    wb = np.exp(-outer_beta * (b - outer_min))
    wsum = wa + wb
    wa /= wsum
    wb /= wsum
    
    # d(r_soft)/d(a) = -wa, d(r_soft)/d(b) = -wb
    # d(a)/d(c) = 0.5 * grad_pair, d(b)/d(c) = grad_face
    grad_total = 0.5 * wa * grad_pair + wb * grad_face
    
    # Objective is -r_soft, so gradient is -grad_total
    grad = -grad_total.flatten()
    
    return value, grad

def _build_initial_grid(N_local):
    """Build an initial grid configuration."""
    m = int(np.ceil(N_local ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N_local]
    return grid

def _spectral_init(N_local, rng):
    """Spectral initialization using eigenvectors of a random matrix."""
    # Create a random symmetric matrix and use its eigenvectors
    A = rng.randn(N_local, N_local)
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    # Use the top eigenvectors to create diverse initial positions
    # Project onto 3D using random linear combinations
    R = rng.randn(3, N_local)
    coords = R @ eigvecs
    coords = coords / np.max(np.abs(coords), axis=0, keepdims=True)
    coords = coords * 0.45  # Scale to fit in cube
    return coords.T

def _optimize_from_init(init, beta_schedule, max_iter_per_beta=200):
    """
    Optimize from an initial configuration using a beta schedule with proper gradients.
    """
    current = init.copy()
    
    for beta in beta_schedule:
        x0 = current.flatten()
        
        def obj_and_grad(x):
            val, grad = _objective_and_grad(x, beta)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': max_iter_per_beta * 3}
            )
            current = result.x.reshape(N, 3)
        except Exception:
            pass
    
    return current

def _local_search_polish(centres, max_iter=500):
    """Final polish using direct minimization of the exact (nonsmooth) objective."""
    c = centres.copy()
    
    for iteration in range(max_iter):
        # Compute current score
        d = pdist(c)
        min_pair = np.min(d)
        face_dists = 0.5 - np.abs(c)
        min_face = np.min(face_dists)
        r = min(min_pair / 2.0, min_face)
        
        # Find the constraint that is most active
        if min_pair / 2.0 < min_face:
            # Pairwise constraint is active
            i, j = np.triu_indices(N, 1)
            diff = c[i] - c[j]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            k = np.argmin(dist)
            di, dj = i[k], j[k]
            direction = -diff[k] / dist[k]
            # Move i and j apart slightly
            step = 1e-8
            c[di] += direction * step
            c[dj] -= direction * step
        else:
            # Boundary constraint is active
            face_dists = 0.5 - np.abs(c)
            min_idx = np.unravel_index(np.argmin(face_dists), face_dists.shape)
            n, k = min_idx
            # Move away from boundary
            sign = np.sign(c[n, k])
            if sign == 0:
                sign = 1
            c[n, k] += sign * 1e-8
        
        # Clip to cube
        c = np.clip(c, -0.5, 0.5)
    
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    # Best solution found so far
    best_centres = None
    best_score = 0.0
    
    # Initial solution: grid
    init = _build_initial_grid(N)
    init_score = _score(init)
    if init_score > best_score:
        best_centres = init.copy()
        best_score = init_score
    
    # Beta schedule: start with low beta (smooth), increase to high beta (sharp)
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    # Run optimization from grid
    refined = _optimize_from_init(init, beta_schedule, max_iter_per_beta=300)
    refined_score = _score(refined)
    if refined_score > best_score:
        best_centres = refined.copy()
        best_score = refined_score
    
    # Try spectral initializations
    n_spectral = 5
    for s in range(n_spectral):
        if time.time() > deadline:
            break
        
        init_s = _spectral_init(N, rng)
        init_s = np.clip(init_s, -0.49, 0.49)
        
        refined_s = _optimize_from_init(init_s, beta_schedule, max_iter_per_beta=200)
        refined_s_score = _score(refined_s)
        if refined_s_score > best_score:
            best_centres = refined_s.copy()
            best_score = refined_s_score
    
    # Try random restarts with jitter
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        init_r = _build_initial_grid(N)
        jitter = rng.uniform(-0.08, 0.08, size=init_r.shape)
        init_r += jitter
        init_r = np.clip(init_r, -0.49, 0.49)
        
        refined_r = _optimize_from_init(init_r, beta_schedule, max_iter_per_beta=150)
        refined_r_score = _score(refined_r)
        if refined_r_score > best_score:
            best_centres = refined_r.copy()
            best_score = refined_r_score
    
    # Final polish with high beta
    if time.time() < deadline - 10 and best_centres is not None:
        final = _optimize_from_init(best_centres, [800.0, 1500.0, 3000.0], max_iter_per_beta=400)
        final_score = _score(final)
        if final_score > best_score:
            best_centres = final.copy()
            best_score = final_score
        
        # Local search polish
        if time.time() < deadline - 5:
            polished = _local_search_polish(best_centres, max_iter=200)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_centres = polished.copy()
                best_score = polished_score
    
    # Ensure validity
    if best_centres is None:
        return _build_initial_grid(N)
    
    # Final clip to ensure inside cube
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
