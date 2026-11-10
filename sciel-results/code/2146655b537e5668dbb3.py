"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(pts):
    """Compute the radius for a set of points in the unit cube [-0.5, 0.5]^3."""
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1]),
        0.5 - np.abs(pts[:, 2])
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use a chunked approach or just compute full matrix if N is small (50 is small)
    pts_flat = pts.reshape(-1)
    # Compute all pairwise distances
    # dist[i, j] = |pts[i] - pts[j]|
    # We want min_{i<j} dist[i,j]
    
    # Efficient calculation for N=50
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def _construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial configuration: Simple grid or random
    # Let's try a few different initializations
    best_pts = None
    best_score = -1.0
    
    # Initialization 1: Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_pts = grid[:N].copy()
    
    # Initialization 2: Random
    rng = np.random.default_rng(42)
    random_pts = rng.uniform(-0.5, 0.5, size=(N, 3))
    
    initializations = [init_pts, random_pts]
    
    # We can also try a BCC-like or FCC-like structure if N matches, but 50 is awkward.
    # For N=50, a 4x4x4 grid is 64 points, so we take 50.
    # A 5x5x2 grid is 50 points exactly.
    # Let's try a 5x5x2 grid.
    ticks_5 = (np.arange(5) + 0.5) / 5.0 - 0.5
    ticks_2 = (np.arange(2) + 0.5) / 2.0 - 0.5
    grid_552 = np.array([(x, y, z) for z in ticks_2 for y in ticks_5 for x in ticks_5])
    initializations.append(grid_552)
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
            
        # Optimization using gradient ascent on score (or minimize negative score)
        # Score is non-differentiable (min of distances).
        # We can use a smooth surrogate or just rely on Nelder-Mead or Powell which don't need gradients.
        # Given N=50, 150 dimensions. Nelder-Mead might be slow.
        # Let's try a simple repulsion-based local search or gradient-free optimization.
        
        # Powell's method is often robust for this.
        # However, it might be slow for 150 variables.
        
        # Alternative: Local search with small perturbations.
        # Or: Gradient ascent using the gradient of the soft-min approximation.
        
        # Let's try a simple iterative repulsion algorithm.
        # Move points away from each other and walls.
        
        pts = init.copy()
        step_size = 0.01
        
        for it in range(2000):
            if time.time() > deadline - 5:
                break
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Wall repulsion
            # Push away from walls
            for i in range(N):
                for d in range(3):
                    if pts[i, d] > 0:
                        forces[i, d] -= 1.0 # Push left
                    else:
                        forces[i, d] += 1.0 # Push right
                        # Actually, we want to push towards center if close to wall
                        # dist to wall = 0.5 - abs(pts[i,d])
                        # force should be proportional to 1/dist or constant?
                        # Let's use a simple force: if close to wall, push inwards.
                        
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        # Random jitter
                        diff = rng.normal(size=3)
                        dist = np.linalg.norm(diff)
                    force = diff / (dist ** 2) # Inverse square law
                    forces[i] += force
                    forces[j] -= force
            
            # Normalize forces to avoid huge steps
            norm = np.linalg.norm(forces, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            forces = forces / norm
            
            # Update positions
            pts += step_size * forces
            
            # Keep inside cube
            pts = np.clip(pts, -0.49, 0.49)
            
            # Check score
            current_score = _score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            # Reduce step size
            step_size *= 0.999
            
            if step_size < 1e-6:
                break
        
        # Final polish with a finer local search
        if time.time() < deadline - 5:
             # Try to optimize the best found so far with a smaller step size
             pts = best_pts.copy()
             step_size = 0.001
             for it in range(1000):
                 if time.time() > deadline - 1:
                     break
                 
                 forces = np.zeros_like(pts)
                 for i in range(N):
                     for j in range(i + 1, N):
                         diff = pts[i] - pts[j]
                         dist = np.linalg.norm(diff)
                         if dist < 1e-6:
                             diff = rng.normal(size=3)
                             dist = np.linalg.norm(diff)
                         force = diff / (dist ** 2)
                         forces[i] += force
                         forces[j] -= force
                 
                 # Wall force
                 for i in range(N):
                     for d in range(3):
                         dist_wall = 0.5 - abs(pts[i, d])
                         if dist_wall < 0.1: # Only if close
                             forces[i, d] += np.sign(pts[i, d]) * -1.0 # Push towards center? No, if positive, push left (negative).
                             # Wait, if pts[i,d] is positive (right half), wall is at 0.5.
                             # Force should be towards 0. So negative force.
                             # If pts[i,d] is negative (left half), wall is at -0.5.
                             # Force should be towards 0. So positive force.
                             # So force = -sign(pts[i,d]) * strength.
                             # My previous code: forces[i, d] += np.sign(pts[i, d]) * -1.0 is correct.
                 
                 norm = np.linalg.norm(forces, axis=1, keepdims=True)
                 norm[norm == 0] = 1.0
                 forces = forces / norm
                 
                 pts += step_size * forces
                 pts = np.clip(pts, -0.49, 0.49)
                 
                 current_score = _score(pts)
                 if current_score > best_score:
                     best_score = current_score
                     best_pts = pts.copy()
                 
                 step_size *= 0.999
                 if step_size < 1e-7:
                     break

    if best_pts is None:
        return init_pts # Fallback to first init
        
    return best_pts

def construct():
    return _construct()
# EVOLVE-BLOCK-END
