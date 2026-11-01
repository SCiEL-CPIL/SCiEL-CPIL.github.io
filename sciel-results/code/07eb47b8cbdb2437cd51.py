def _greedy_seed(N, grid_size=150, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    # Create a grid of candidate points
    # Use a slightly larger grid to ensure enough points
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    candidates = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Add small random jitter to break symmetry and allow for "discrete" exploration
    candidates += rng.normal(0, 0.01, candidates.shape)
    
    # Filter candidates inside the unit circle
    dists = np.linalg.norm(candidates, axis=1)
    candidates = candidates[dists < 0.98]
    
    if len(candidates) < N:
        # Fallback if grid is too small
        return _lattice_seed(2.0 / (np.sqrt(N) + 1.0), rng)

    selected = []
    
    for _ in range(N):
        if len(candidates) == 0:
            break
            
        if len(selected) > 0:
            sel_arr = np.array(selected)
            diff = candidates[:, np.newaxis, :] - sel_arr[np.newaxis, :, :]
            dists_to_sel = np.linalg.norm(diff, axis=2)
            min_d_to_sel = np.min(dists_to_sel, axis=1)
        else:
            min_d_to_sel = np.full(len(candidates), np.inf)
            
        dist_to_bound = 1.0 - np.linalg.norm(candidates, axis=1)
        
        # Score: We want to maximize the minimum distance.
        # To avoid picking points that are too close to the boundary initially,
        # we can weight the boundary distance.
        # Simple approach: score = min_d_to_sel
        # Better approach: score = min(min_d_to_sel, dist_to_bound)
        # But min_d_to_sel is 0 for the first point? No, it's inf.
        # For the first point, we should pick one that maximizes dist_to_bound?
        # Or just pick a random one?
        # Let's pick the one that maximizes min_d_to_sel.
        # If min_d_to_sel is inf (first point), we pick the one with max dist_to_bound?
        # Actually, for the first point, any point is fine, but center is good.
        
        if len(selected) == 0:
             # Pick point closest to origin? Or random?
             # Let's pick the one with max dist_to_bound to be safe?
             # No, center is best for symmetry.
             # Let's pick the one with min norm.
             scores = -np.linalg.norm(candidates, axis=1)
        else:
             scores = min_d_to_sel
             
        best_idx = np.argmax(scores)
        best_pt = candidates[best_idx]
        selected.append(best_pt)
        
        # Remove candidates that are too close to the new point to speed up?
        # We can remove candidates with dist < threshold.
        # Threshold: current best min dist / 2?
        # Let's just keep all for simplicity, N=300, Grid=150x150=22500 points.
        # 300 * 22500 = 6.75M ops. Still fast.
        
    return np.array(selected)
