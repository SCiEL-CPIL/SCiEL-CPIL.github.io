"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    d = pdist(pts, 'euclidean')
    min_pair = d.min() if len(d) > 0 else 0.0
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill

    # Start with a grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)

    np.random.seed(42)

    # Local search with adaptive step sizes
    # Key improvement: use larger initial steps and anneal them down
    # Also: when stuck, try larger perturbations (restart-like behavior within local search)
    
    current_pts = best_pts.copy()
    current_r = best_r

    # Multi-scale local search
    # Phase 1: coarse optimization with larger steps
    # Phase 2: fine optimization with smaller steps
    
    # Phase 1: Coarse search
    step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        
        # Run local search with this step size
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            orig = current_pts[idx].copy()
            
            # Try several random directions
            best_move = None
            best_move_r = current_r
            
            n_trials = 30
            for _ in range(n_trials):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                # Use step size with some randomness
                step = step_size * (0.5 + np.random.rand())
                new_pos = orig + direction * step
                new_pos = np.clip(new_pos, -0.499, 0.499)
                
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
        
        # If we improved, continue with this step size for a bit more
        if current_r > best_r - 1e-10:
            # Do a few more iterations at this scale
            for it in range(200):
                if time.time() > deadline:
                    break
                
                idx = np.random.randint(N)
                orig = current_pts[idx].copy()
                
                best_move = None
                best_move_r = current_r
                
                for _ in range(20):
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)
                    step = step_size * (0.5 + np.random.rand())
                    new_pos = orig + direction * step
                    new_pos = np.clip(new_pos, -0.499, 0.499)
                    
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

    # Final fine-tuning phase
    if time.time() < deadline:
        current_pts = best_pts.copy()
        current_r = best_r
        
        for it in range(1000):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(N)
            orig = current_pts[idx].copy()
            
            best_move = None
            best_move_r = current_r
            
            for _ in range(15):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                step = 0.0005 * (0.5 + np.random.rand())
                new_pos = orig + direction * step
                new_pos = np.clip(new_pos, -0.499, 0.499)
                
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

    # Ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.499, 0.499)
    
    return best_pts
# EVOLVE-BLOCK-END
