"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    # Min distance between pairs
    from scipy.spatial.distance import pdist
    d = pdist(pts, 'euclidean')
    min_pair = d.min() if len(d) > 0 else 0.0
    # Distance to boundary
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r


def _negative_radius(pts):
    """Objective to minimize (negative of radius)."""
    pts = np.array(pts).reshape(-1, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    if np.any(np.abs(pts) > 0.5):
        # Penalize being outside
        penalty = np.sum(np.maximum(0, np.abs(pts) - 0.5))
        return 1e10 + penalty
    r = _compute_radius(pts)
    return -r


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Start with a grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Use Nelder-Mead or similar optimization
    # Flatten points for optimization
    x0 = pts.flatten()
    
    # Try L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # We need a smooth objective. The min of distances is non-smooth.
    # Use a different approach: local search with perturbations
    
    # Simple local search: randomly perturb points and keep if radius improves
    best_r = _compute_radius(best_pts)
    current_pts = best_pts.copy()
    current_r = best_r
    
    np.random.seed(42)
    
    # Try a few optimization strategies
    # 1. Start with Nelder-Mead on small subsets? No, too slow for 200 points.
    # 2. Use gradient-free local search
    
    # Better: use scipy minimize with a smoothed objective or just Nelder-Mead on the whole thing
    # But Nelder-Mead on 400 variables is slow.
    
    # Let's try a simple iterative improvement:
    # For each point, try moving it slightly in random directions to increase min distance
    
    # Actually, let's use a more systematic approach:
    # Divide into blocks and optimize each block?
    
    # For now, let's do a simple local search with random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_pts = best_pts.copy()
        else:
            # Add small random noise
            current_pts = best_pts + np.random.randn(N, 2) * 0.01
            # Clip to bounds
            current_pts = np.clip(current_pts, -0.49, 0.49)
        
        current_r = _compute_radius(current_pts)
        
        # Local search iterations
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            orig = current_pts[idx].copy()
            
            # Try several random directions
            best_move = None
            best_move_r = current_r
            
            for _ in range(20):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                step = 0.005 * (0.5 + np.random.rand())
                new_pos = orig + direction * step
                new_pos = np.clip(new_pos, -0.49, 0.49)
                
                temp_pts = current_pts.copy()
                temp_pts[idx] = new_pos
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > best_move_r:
                    best_move_r = temp_r
                    best_move = new_pos
            
            if best_move is not None:
                current_pts[idx] = best_move
                current_r = best_move_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
    
    # Final check
    if time.time() > deadline:
        return best_pts
    
    # Ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.499, 0.499)
    
    return best_pts
# EVOLVE-BLOCK-END
