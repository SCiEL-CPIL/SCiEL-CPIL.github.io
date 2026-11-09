"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 50
CUBE_HALF = 0.5
DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _pairwise_distances(centers):
    """Compute pairwise distances between all pairs of centers. Shape: (N, N)."""
    # Use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    return dists


def _min_pairwise_dist(centers):
    """Minimum pairwise distance."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def _min_boundary_dist(centers):
    """Minimum distance from any center to the cube boundary."""
    # Distance to nearest face: min over coordinates of (0.5 - |coord|)
    d = CUBE_HALF - np.abs(centers)
    return np.min(d)


def _score(centers):
    """Compute the common radius (score) for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check all centers are inside the cube
    if np.any(np.abs(centers) > CUBE_HALF):
        return 0.0
    mpd = _min_pairwise_dist(centers)
    mbd = _min_boundary_dist(centers)
    return min(mpd / 2.0, mbd)


def _soft_min_score(centers, beta):
    """
    Smooth surrogate: maximize a soft-min of (pairwise_dists/2, boundary_dists).
    We minimize the negative of a soft-max.
    
    soft_max(x, beta) = (1/beta) * log(sum(exp(beta * x)))
    We want to maximize min(terms), so we minimize -soft_max(terms, beta).
    """
    # Compute all pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Boundary distances
    bnd_terms = CUBE_HALF - np.abs(centers)
    
    # Combine all terms into a flat array
    pair_flat = pair_terms[pair_terms < np.inf]
    bnd_flat = bnd_terms.flatten()
    
    all_terms = np.concatenate([pair_flat, bnd_flat])
    
    # Soft max via log-sum-exp for numerical stability
    max_term = np.max(all_terms)
    exp_terms = np.exp(beta * (all_terms - max_term))
    soft_max = max_term + np.log(np.sum(exp_terms)) / beta
    
    return -soft_max


def _soft_min_grad(centers, beta):
    """
    Gradient of the soft-min score surrogate (negative soft-max).
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    bnd_terms = CUBE_HALF - np.abs(centers)
    
    pair_flat = pair_terms[pair_terms < np.inf]
    bnd_flat = bnd_terms.flatten()
    
    all_terms = np.concatenate([pair_flat, bnd_flat])
    
    max_term = np.max(all_terms)
    exp_terms = np.exp(beta * (all_terms - max_term))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft_max w.r.t. each term is just weights
    # d(soft_max)/d(centers) = sum_i weights_i * d(term_i)/d(centers)
    
    # For pairwise terms: d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # d(pair_term_ij)/d(c_i) = (c_i - c_j) / (2 * dist_ij)
    
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    # For each pair (i,j), contribution to grad[i] and grad[j]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            # weight for this pair term
            w = weights[0]  # placeholder, need to track indices
            # Actually, let's do this more carefully with explicit indexing
    
    # Let me redo this more carefully
    grad = np.zeros_like(centers)
    
    # Rebuild with explicit tracking
    # Pair terms are stored in upper triangle order
    pair_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_indices.append((i, j))
    
    # pair_flat corresponds to these indices
    for idx, (i, j) in enumerate(pair_indices):
        w = weights[idx]
        d = dists[i, j]
        if d < 1e-12:
            continue
        diff_ij = centers[i] - centers[j]
        # d(pair_term)/d(c_i) = diff_ij / (2*d)
        grad[i] += w * diff_ij / (2.0 * d)
        # d(pair_term)/d(c_j) = -diff_ij / (2*d)
        grad[j] += w * (-diff_ij / (2.0 * d))
    
    # Boundary terms: d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    bnd_offset = len(pair_indices)
    for i in range(N):
        for k in range(3):
            w = weights[bnd_offset + i * 3 + k]
            c = centers[i, k]
            if abs(c) < 1e-15:
                s = 0.0
            else:
                s = np.sign(c)
            grad[i, k] += w * (-s)
    
    # We're minimizing -soft_max, so gradient is -d(soft_max)/d(centers)
    return -grad


def _construct_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_random():
    """Random initialization inside the cube."""
    return (np.random.rand(N, 3) - 0.5) * 0.99


def _local_optimize(centers, beta, maxiter=1000):
    """Run L-BFGS-B optimization on the soft-min surrogate."""
    def objective(x):
        c = x.reshape(N, 3)
        return _soft_min_score(c, beta)
    
    def gradient(x):
        c = x.reshape(N, 3)
        return _soft_min_grad(c, beta).flatten()
    
    x0 = centers.flatten()
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=[(-CUBE_HALF, CUBE_HALF)] * (N * 3),
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x.reshape(N, 3)


def _refine(centers, time_limit):
    """
    Multi-stage refinement:
    1. Anneal beta from low to high, running L-BFGS-B at each stage.
    2. Final polish with very high beta.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Annealing schedule: increasing beta values
    betas = [10, 30, 60, 100, 200, 400, 800, 1500, 3000, 5000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
        try:
            current = _local_optimize(current, beta, maxiter=500)
        except Exception:
            break
        # Ensure validity
        if not np.all(np.isfinite(current)):
            current = centers.copy()
            break
        # Clip to cube (shouldn't be needed with bounds, but safety)
        current = np.clip(current, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    # Final high-precision polish
    if time.time() - start_time < time_limit - 2:
        try:
            current = _local_optimize(current, 10000, maxiter=2000)
        except Exception:
            pass
    
    return current


def construct():
    """
    Construct centers of N equal spheres in the unit cube to maximize common radius.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_centers = _construct_grid()
    best_score = _score(best_centers)
    
    # Try multiple initializations with different strategies
    initializations = []
    
    # 1. Grid
    initializations.append(_construct_grid())
    
    # 2. Random
    for _ in range(3):
        initializations.append(_construct_random())
    
    # 3. Grid with small perturbations
    grid = _construct_grid()
    for _ in range(3):
        perturbed = grid + np.random.randn(N, 3) * 0.01
        perturbed = np.clip(perturbed, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        initializations.append(perturbed)
    
    # 4. Face-centered-ish: place some near faces
    fcc_like = []
    # 5x5x2 grid
    for x in np.linspace(-0.4, 0.4, 5):
        for y in np.linspace(-0.4, 0.4, 5):
            for z in [-0.25, 0.25]:
                fcc_like.append([x, y, z])
    fcc_like = np.array(fcc_like[:N])
    initializations.append(fcc_like)
    
    # Run local optimization on each initialization
    time_per_init = (DEADLINE - 10) / len(initializations)
    
    for i, init in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        # Ensure valid
        init = np.clip(init, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        # Quick local optimization with moderate beta
        try:
            refined = _local_optimize(init, beta=50, maxiter=300)
            refined = np.clip(refined, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_centers = refined.copy()
        except Exception:
            continue
    
    # Final refinement of the best solution
    if time.time() < deadline - 5:
        try:
            final = _refine(best_centers, time_limit=DEADLINE - 3)
            final = np.clip(final, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s = _score(final)
            if s > best_score:
                best_score = s
                best_centers = final.copy()
        except Exception:
            pass
    
    # Final safety: ensure valid output
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    if not np.all(np.isfinite(best_centers)):
        best_centers = _construct_grid()
    
    return best_centers
