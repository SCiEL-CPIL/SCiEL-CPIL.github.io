"""Circle packing in a unit square: 200 equal circles, maximize radius."""
import numpy as np
import time

N = 200
DEADLINE_FRACTION = 0.85  # use 85% of the 120s budget


def _score(centres):
    """Compute the packing radius (minimum of half min-pair-distance and min wall distance)."""
    c = np.asarray(centres, dtype=np.float64)
    n = c.shape[0]
    # Wall distances
    wall = np.minimum(
        np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    )
    min_wall = np.min(wall)
    # Pair distances - compute all pairwise distances efficiently
    # Use chunked computation to avoid huge memory
    min_pair = np.inf
    for i in range(n):
        diffs = c[i + 1:] - c[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        if dists.size > 0:
            min_pair = min(min_pair, np.min(dists))
    return min(min_pair / 2.0, min_wall)


def _hexagonal_lattice(rows, cols, jitter_scale, rng):
    """Generate a jittered hexagonal lattice pattern for N circles."""
    # Hexagonal packing: stagger rows
    # For a hexagonal lattice, the spacing between centers is 2r
    # We want to place N points; use a roughly sqrt(N) x sqrt(N) grid with hex offset
    # Start with a rectangular grid then apply hexagonal staggering
    
    # Determine grid dimensions
    # For hexagonal packing, aspect ratio is sqrt(3)/2
    # Let's use rows and cols such that rows*cols >= N
    while rows * cols < N:
        cols += 1
    
    # Create base grid
    xs = np.linspace(-0.5, 0.5, cols)
    ys = np.linspace(-0.5, 0.5, rows)
    
    # Build hexagonal pattern
    points = []
    for r in range(rows):
        for c in range(cols):
            x = xs[c]
            y = ys[r]
            # Hexagonal offset: shift every other row
            if r % 2 == 1:
                x += (xs[1] - xs[0]) * 0.5 if cols > 1 else 0
            points.append([x, y])
    
    points = np.array(points[:N], dtype=np.float64)
    
    # Add jitter
    if jitter_scale > 0:
        points += rng.normal(0, jitter_scale, size=points.shape)
    
    # Clip to square
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    return points


def _nelder_mead_step(centres, rng, max_iter=500, tol=1e-10):
    """
    Apply Nelder-Mead simplex optimization to improve the packing.
    We optimize the negative of the packing radius.
    """
    from scipy.optimize import minimize
    
    def objective(x_flat):
        c = x_flat.reshape(-1, 2)
        return -_score(c)
    
    x0 = centres.flatten()
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iter,
            'xatol': tol,
            'fatol': 1e-12,
            'adaptive': True
        }
    )
    
    if result.success or result.fun < objective(x0):
        return result.x.reshape(-1, 2)
    return centres


def _local_refinement(centres, rng, time_budget):
    """
    Perform local refinement using a combination of:
    1. Gradient-free optimization (Nelder-Mead) on subsets
    2. Direct coordinate perturbation of worst-placed circles
    """
    best = centres.copy()
    best_score = _score(best)
    start = time.time()
    
    # Strategy: iteratively find the worst circle (smallest clearance) and try to move it
    for iteration in range(2000):
        if time.time() - start > time_budget:
            break
        
        c = best
        n = len(c)
        
        # Compute clearance for each circle
        # Wall clearance
        wall_clear = np.minimum(
            np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
        )
        
        # Pair clearance (approximate: for each circle, min distance to others / 2)
        # This is expensive, so do it in chunks
        pair_clear = np.full(n, np.inf)
        for i in range(n):
            if i == 0:
                diffs = c[1:] - c[0]
            else:
                # Only need distances from i to others not yet computed
                diffs_prev = c[:i] - c[i]
                d_prev = np.sqrt(np.sum(diffs_prev * diffs_prev, axis=1))
                if d_prev.size > 0:
                    pair_clear[i] = min(pair_clear[i], np.min(d_prev) / 2)
                    # Also update the previous circles
                    pair_clear[:i] = np.minimum(pair_clear[:i], d_prev / 2)
            
            if i < n - 1:
                diffs_next = c[i + 1:] - c[i]
                d_next = np.sqrt(np.sum(diffs_next * diffs_next, axis=1))
                if d_next.size > 0:
                    pair_clear[i] = min(pair_clear[i], np.min(d_next) / 2)
                    pair_clear[i + 1:] = np.minimum(pair_clear[i + 1:], d_next / 2)
        
        # Total clearance for each circle
        clearance = np.minimum(wall_clear, pair_clear)
        
        # Find the worst circle
        worst_idx = np.argmin(clearance)
        
        # Try to move the worst circle to improve its clearance
        # Use random perturbations
        improved = False
        for _ in range(50):
            new_c = c.copy()
            # Random small perturbation
            delta = rng.normal(0, 0.005, size=2)
            new_c[worst_idx] += delta
            
            # Check if still valid (inside square)
            if np.any(np.abs(new_c) > 0.5):
                continue
            
            new_score = _score(new_c)
            if new_score > best_score + 1e-12:
                best = new_c
                best_score = new_score
                improved = True
                break
        
        if not improved:
            # Try larger moves
            for _ in range(20):
                new_c = c.copy()
                delta = rng.normal(0, 0.02, size=2)
                new_c[worst_idx] += delta
                if np.any(np.abs(new_c) > 0.5):
                    continue
                new_score = _score(new_c)
                if new_score > best_score + 1e-12:
                    best = new_c
                    best_score = new_score
                    improved = True
                    break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    best_centres = None
    best_score = -np.inf
    
    # Try multiple hexagonal lattice configurations with different parameters
    configs = []
    for rows in range(12, 20):
        for cols in range(12, 20):
            for jitter in [0.0, 0.001, 0.005, 0.01, 0.02]:
                configs.append((rows, cols, jitter))
    
    # Shuffle configs
    rng.shuffle(configs)
    
    # Evaluate initial configurations
    for rows, cols, jitter in configs:
        if time.time() > deadline - 30:
            break
        
        centres = _hexagonal_lattice(rows, cols, jitter, rng)
        score = _score(centres)
        
        if score > best_score:
            best_score = score
            best_centres = centres.copy()
    
    # Ensure we have a valid initial configuration
    if best_centres is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
        best_score = _score(best_centres)
    
    # Local refinement with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        refined = _local_refinement(best_centres, rng, remaining_time * 0.7)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_centres = refined
            best_score = refined_score
    
    # Final attempt: try Nelder-Mead on the best configuration if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_centres is not None:
        try:
            nm_result = _nelder_mead_step(best_centres, rng, max_iter=200, tol=1e-10)
            nm_score = _score(nm_result)
            if nm_score > best_score:
                best_centres = nm_result
                best_score = nm_score
        except Exception:
            pass
    
    # Ensure valid output
    best_centres = np.asarray(best_centres, dtype=np.float64)
    assert best_centres.shape == (N, 2)
    assert np.all(np.isfinite(best_centres))
    assert np.all(np.abs(best_centres) <= 0.5)
    
    return best_centres
