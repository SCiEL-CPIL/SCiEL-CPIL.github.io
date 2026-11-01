"""Circle packing via discrete force relaxation with radius ramping."""
import numpy as np
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    if n < 2:
        return 0.0
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)

def _hex_init(spacing):
    """Hexagonal lattice initialization."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radius, kind='stable')[:N]]
    return points

def _ring_init(n_rings=8):
    """Concentric rings initialization."""
    pts = [np.array([0.0, 0.0])]
    for ring in range(1, n_rings):
        r = ring * 0.12
        n_in_ring = min(6 * ring, N - len(pts))
        if n_in_ring <= 0:
            break
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        for a in angles:
            pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    pts = np.array(pts[:N])
    return pts

def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1, size=(N * 2, 2))
        r = np.hypot(x[:, 0], x[:, 1])
        valid = r <= 0.95
        pts.extend(x[valid].tolist())
    pts = np.array(pts[:N])
    return pts

def _relax(pts, target_r, max_iters=500, dt=0.05, damping=0.9):
    """
    Relax points to satisfy constraints for radius target_r.
    Uses a simple force-based model:
    - Repulsion between overlapping circles (springs)
    - Attraction to container boundary if too close (or repulsion if outside)
    
    Returns the relaxed points.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        forces = np.zeros_like(pts)
        
        # 1. Pairwise repulsion
        # We want dist >= 2*target_r
        # If dist < 2*target_r, push apart
        diffs = pts[i_idx] - pts[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Avoid division by zero
        valid = dists > 1e-12
        if np.any(valid):
            # Overlap amount
            overlap = 2 * target_r - dists[valid]
            # Force magnitude proportional to overlap
            # Only apply if overlapping
            active = overlap > 0
            if np.any(active):
                # Normalize direction
                dirs = diffs[valid][active] / dists[valid][active, None]
                # Force on i is +dir * mag, on j is -dir * mag
                mags = overlap[active] * 10.0 # Stiffness
                
                # Accumulate forces
                # i_idx and j_idx are for all pairs, so we need to filter
                # It's more efficient to just loop or use np.add.at
                # For N=50, looping is fast enough and simpler to get right
                # But let's stick to vectorized if possible.
                
                # Create force vectors for active pairs
                f_vecs = dirs * mags[:, None]
                
                # Add to forces[i_idx[valid][active]]
                np.add.at(forces, i_idx[valid][active], f_vecs)
                # Add to forces[j_idx[valid][active]]
                np.add.at(forces, j_idx[valid][active], -f_vecs)

        # 2. Boundary constraint
        # We want |pts| <= 1 - target_r
        # If |pts| > 1 - target_r, push in
        radii = np.linalg.norm(pts, axis=1)
        max_r = 1.0 - target_r
        outside = radii > max_r
        if np.any(outside):
            # Push in towards origin
            # Force magnitude proportional to violation
            violations = radii[outside] - max_r
            dirs = pts[outside] / radii[outside, None]
            f_vecs = -dirs * violations * 10.0
            forces[outside] += f_vecs
        
        # Integrate
        pts += forces * dt
        # Damping is implicit in the force calculation if we don't store velocity,
        # but this is essentially a gradient descent step.
        # To improve stability, we can clamp movement
        # Actually, let's just use a simple projection step after force?
        # No, force relaxation is better.
        
        # Check convergence?
        # If max force is small, stop
        if np.max(np.linalg.norm(forces, axis=1)) < 1e-6:
            break
            
    return pts

def _optimize_radius(init_pts, target_r, time_limit):
    """
    Try to achieve a packing with radius >= target_r starting from init_pts.
    Returns (pts, score, success)
    """
    t0 = time.time()
    pts = init_pts.copy()
    
    # Coarse to fine relaxation
    # Start with a smaller radius to settle positions, then increase
    # This helps escape local minima where circles are "stuck" in a bad config
    current_r = target_r * 0.8
    
    # Phase 1: Settle at lower radius
    steps = 100
    for _ in range(steps):
        if time.time() - t0 > time_limit * 0.2:
            break
        pts = _relax(pts, current_r, max_iters=10, dt=0.01)
        
    # Phase 2: Ramp up radius
    num_ramps = 10
    for i in range(num_ramps):
        if time.time() - t0 > time_limit * 0.8:
            break
        frac = (i + 1) / num_ramps
        r_target = current_r + (target_r - current_r) * frac
        
        # Relax at this target radius
        # Increase iterations as we get closer to target
        iters = 50 + i * 20
        pts = _relax(pts, r_target, max_iters=iters, dt=0.02)
        
    # Final polish at target radius
    if time.time() - t0 < time_limit:
        pts = _relax(pts, target_r, max_iters=200, dt=0.01)
        
    score = _compute_score(pts)
    success = score >= target_r - 1e-8
    return pts, score, success

def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    inits = [
        _hex_init(0.25),
        _hex_init(0.22),
        _hex_init(0.28),
        _ring_init(8),
        _ring_init(7),
        _random_init(42),
        _random_init(123),
        _random_init(7),
    ]
    
    # Filter valid inits
    valid_inits = []
    for init in inits:
        if len(init) == N:
            valid_inits.append(init)
    
    if not valid_inits:
        return _hex_init(0.25)

    # Strategy:
    # 1. For each init, do a quick relaxation at a moderate radius to get a "good" shape.
    # 2. Then, perform a binary search on radius using the best shape as a starting point
    #    for each trial, or re-initialize from the best shape.
    
    # Step 1: Get decent starting points
    r_initial = 0.115
    for init in valid_inits:
        if time.time() - t_start > DEADLINE - 40:
            break
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_opt = min(time_remaining * 0.1, 5.0)
        if time_for_opt < 1.0:
            break
            
        pts, score, _ = _optimize_radius(init, r_initial, time_for_opt)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = valid_inits[0]
        best_score = _compute_score(best_pts)
        
    # Step 2: Binary search for max radius
    # We know the best known is ~0.1258
    # Start low and go high
    r_low = best_score
    r_high = 0.130
    
    # We will use the best_pts found so far as the seed for the binary search steps
    # If a step fails, we might try a different seed or just accept the lower bound
    
    num_bs_steps = 12
    for step in range(num_bs_steps):
        if time.time() - t_start > DEADLINE - 5:
            break
            
        r_mid = (r_low + r_high) / 2.0
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_opt = min(time_remaining * 0.6, 10.0)
        if time_for_opt < 1.0:
            break
            
        # Try to achieve r_mid starting from best_pts
        # Also try from a fresh init if best_pts is stuck?
        # For now, just use best_pts.
        
        pts_new, score_new, success = _optimize_radius(best_pts, r_mid, time_for_opt)
        
        if success:
            r_low = r_mid
            if score_new > best_score:
                best_score = score_new
                best_pts = pts_new.copy()
        else:
            r_high = r_mid
            # Even if it failed, maybe it improved the score?
            if score_new > best_score:
                best_score = score_new
                best_pts = pts_new.copy()
                
    # Final polish
    if time.time() - t_start < DEADLINE - 2:
        time_remaining = DEADLINE - (time.time() - t_start)
        # Try to polish at the current best score + epsilon
        target = best_score + 0.0001
        pts_final, score_final, _ = _optimize_radius(best_pts, target, time_remaining)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final.copy()
            
    return best_pts
