"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_min_logsumexp(d, beta):
    """
    Smooth surrogate for min(d) using log-sum-exp.
    Returns -log(sum(exp(-beta*d)))/beta which approximates min(d) as beta->inf.
    We want to maximize min distance, so we minimize this surrogate.
    """
    # d is a 1D array of distances
    # For numerical stability, shift by min
    d_min = np.min(d)
    # exp(-beta * (d - d_min)) is bounded by 1
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check boundaries: min distance to each face
    # Face distances: 0.5 - |c_ik| for each coordinate k
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    # Pairwise distances
    # Use broadcasting for pairwise differences
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _objective(c, beta):
    """
    Objective function to minimize: -soft_min of constraints.
    We want to maximize min(pairwise_dist/2, face_dist).
    So we minimize the negative of a smooth approximation of that min.
    
    Constraints:
    1. Pairwise distances: d_ij >= 2r
    2. Face distances: f_ik >= r
    
    We can define a smooth "radius" estimate:
    r_smooth = soft_min( [d_ij/2 for all i<j], [f_ik for all i,k] )
    Then minimize -r_smooth.
    """
    # Flatten c to 1D for scipy
    c = c.reshape(N, 3)
    
    # Compute pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    # Get upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Compute face distances
    face_dists = 0.5 - np.abs(c)
    face_flat = face_dists.flatten()
    
    # Combine all constraints into one array
    all_dists = np.concatenate([pair_dists, face_flat])
    
    # Soft min
    r_smooth = _soft_min_logsumexp(all_dists, beta)
    
    return -r_smooth


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    best_c = c0.copy()
    best_score = _score(best_c)
    
    # Multi-start optimization with beta annealing
    rng = np.random.default_rng(42)
    
    # Beta schedule: start low, increase over iterations
    beta_starts = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    # Number of restarts per beta stage
    n_restarts = 5
    
    for beta_idx, beta in enumerate(beta_starts):
        if time.time() > deadline:
            break
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Generate initial point
            if restart == 0:
                # Use current best
                c_init = best_c.copy()
            else:
                # Random perturbation of grid or random
                if restart == 1:
                    # Perturbed grid
                    c_init = c0 + rng.normal(0, 0.05, size=(N, 3))
                else:
                    # Random points in cube
                    c_init = rng.uniform(-0.4, 0.4, size=(N, 3))
            
            # Clip to valid range
            c_init = np.clip(c_init, -0.49, 0.49)
            
            # Optimize
            try:
                res = minimize(
                    _objective,
                    c_init.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                c_opt = res.x.reshape(N, 3)
                
                # Clip to ensure validity
                c_opt = np.clip(c_opt, -0.4999, 0.4999)
                
                score = _score(c_opt)
                if score > best_score:
                    best_score = score
                    best_c = c_opt.copy()
            except Exception:
                pass
        
        # If we've reached a good score, try a final polish with higher beta
        if best_score > 0.13:
            break
    
    # Final polish pass with high beta on the best configuration
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _objective,
                best_c.flatten(),
                args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_polished = res.x.reshape(N, 3)
            c_polished = np.clip(c_polished, -0.49999, 0.49999)
            
            score = _score(c_polished)
            if score > best_score:
                best_score = score
                best_c = c_polished.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
