"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

# Precompute upper triangle indices for pairwise distances
_IU = np.triu_indices(N, k=1)

def _pairwise_dists(c):
    """Compute all pairwise squared distances (faster for comparisons)."""
    diff = c[_IU[0]] - c[_IU[1]]
    return np.sum(diff * diff, axis=1)

def _min_pairwise_dist(c):
    """Return the minimum pairwise distance."""
    return np.sqrt(np.min(_pairwise_dists(c)))

def _min_boundary_dist(c):
    """Return the minimum distance from any centre to the nearest face."""
    abs_c = np.abs(c)
    return np.min(0.5 - abs_c)

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    r_pair = _min_pairwise_dist(c) / 2.0
    r_bound = _min_boundary_dist(c)
    return min(r_pair, r_bound)

def _initial_grid():
    """Create a well-distributed initial placement using Fibonacci sphere + jitter."""
    # Use a body-centered cubic lattice pattern for better initial spread
    # Place points on a 5x5x4 grid with some offset
    c = np.zeros((N, 3))
    idx = 0
    for z in range(4):
        for y in range(5):
            for x in range(5):
                if idx >= N:
                    break
                # Offset alternating layers for better packing
                offset = 0.1 if (z % 2 == 1) else 0.0
                c[idx, 0] = (x + offset - 2.0) / 4.0 * 0.9
                c[idx, 1] = (y + offset - 2.0) / 4.0 * 0.9
                c[idx, 2] = (z - 1.5) / 3.0 * 0.9
                idx += 1
                if idx >= N:
                    break
            if idx >= N:
                break
        if idx >= N:
            break
    
    # Add small random jitter to break symmetry
    c += np.random.randn(N, 3) * 0.01
    c = np.clip(c, -0.49, 0.49)
    return c

def _simulated_annealing(c, max_iters=30000, T0=0.008, T1=1e-8, alpha=0.9996, step_scale=0.02):
    """
    Improved simulated annealing with adaptive step size and multi-point moves.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    step = step_scale
    n_accepted = 0
    n_total = 0
    patience = 0
    
    for it in range(max_iters):
        # Adaptive step size: decrease if many rejections, increase if many acceptances
        if n_total > 0:
            accept_rate = n_accepted / n_total
            if accept_rate < 0.1:
                step *= 1.5
            elif accept_rate > 0.8:
                step *= 0.7
            step = np.clip(step, 0.001, 0.1)
        
        # Generate perturbation: move 1-3 random points
        n_move = 1 if random.random() < 0.5 else (2 if random.random() < 0.7 else 3)
        idxs = random.sample(range(N), n_move)
        
        c_new = c.copy()
        for idx in idxs:
            delta = np.random.randn(3) * step
            c_new[idx] = np.clip(c_new[idx] + delta, -0.4999, 0.4999)
        
        new_score = _score(c_new)
        
        # Metropolis acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            n_accepted += 1
            patience = 0
        else:
            delta_score = new_score - current_score
            if delta_score > -100 * T:  # Avoid underflow
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
                    n_accepted += 1
        
        n_total += 1
        
        # Track best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
            patience = 0
        else:
            patience += 1
        
        # Reset patience if we've been stuck too long (restart with perturbation)
        if patience > 5000:
            # Perturb several random points to escape local minimum
            n_perturb = random.randint(5, 15)
            pert_idxs = random.sample(range(N), n_perturb)
            for idx in pert_idxs:
                c[idx] = np.clip(c[idx] + np.random.randn(3) * 0.05, -0.4999, 0.4999)
            current_score = _score(c)
            patience = 0
            n_accepted = 0
            n_total = 0
        
        # Cool down
        T *= alpha
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine(c, max_iters=10000):
    """
    Local refinement: iteratively push the most constrained points apart.
    Uses vectorized operations for speed.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        # Find the limiting constraint
        r_pair = _min_pairwise_dist(c) / 2.0
        r_bound = _min_boundary_dist(c)
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            dists_sq = _pairwise_dists(c)
            min_idx = np.argmin(dists_sq)
            i, j = _IU[0][min_idx], _IU[1][min_idx]
            
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try multiple move sizes
            improved = False
            for move in [0.003, 0.001, 0.0005, 0.002, 0.0015]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.4999, 0.4999)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.4999, 0.4999)
                
                new_score = _score(c_new)
                if new_score > current_score + 1e-12:
                    c = c_new
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
            
            if not improved:
                # Try moving one point in a random direction
                for idx in [i, j]:
                    for _ in range(3):
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        for move in [0.002, 0.001]:
                            c_new = c.copy()
                            c_new[idx] = np.clip(c_new[idx] + direction * move, -0.4999, 0.4999)
                            new_score = _score(c_new)
                            if new_score > current_score + 1e-12:
                                c = c_new
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_c = c.copy()
                                break
        
        else:
            # Boundary is limiting
            abs_c = np.abs(c)
            boundary_dists = 0.5 - abs_c
            min_idx = np.argmin(boundary_dists)
            i = min_idx // 3
            axis = min_idx % 3
            
            sign = 1.0 if c[i, axis] > 0 else -1.0
            improved = False
            for move in [0.003, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.4999, 0.4999)
                new_score = _score(c_new)
                if new_score > current_score + 1e-12:
                    c = c_new
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
            
            if not improved:
                # Try moving in other directions
                for _ in range(3):
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    for move in [0.002, 0.001]:
                        c_new = c.copy()
                        c_new[i] = np.clip(c_new[i] + direction * move, -0.4999, 0.4999)
                        new_score = _score(c_new)
                        if new_score > current_score + 1e-12:
                            c = c_new
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_c = c.copy()
                            break
        
        # Occasional random kick to escape local minima
        if it % 200 == 0:
            n_kick = random.randint(1, 3)
            for _ in range(n_kick):
                idx = random.randint(0, N - 1)
                c_new = c.copy()
                c_new[idx] = np.clip(c_new[idx] + np.random.randn(3) * 0.008, -0.4999, 0.4999)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    np.random.seed(42)
    random.seed(42)
    
    # Start with a structured initial placement
    c = _initial_grid()
    
    # First round: aggressive simulated annealing
    c, score = _simulated_annealing(c, max_iters=25000, T0=0.006, T1=1e-8, alpha=0.9996, step_scale=0.02)
    
    # Local refinement
    c, score = _local_refine(c, max_iters=8000)
    
    # Second round: finer annealing
    c, score2 = _simulated_annealing(c, max_iters=20000, T0=0.003, T1=1e-9, alpha=0.9997, step_scale=0.01)
    if score2 > score:
        score = score2
        c, score = _local_refine(c, max_iters=5000)
    
    # Third round: very fine tuning
    c, score3 = _simulated_annealing(c, max_iters=15000, T0=0.001, T1=1e-10, alpha=0.9998, step_scale=0.005)
    if score3 > score:
        score = score3
        c, score = _local_refine(c, max_iters=3000)
    
    return c
# EVOLVE-BLOCK-END
