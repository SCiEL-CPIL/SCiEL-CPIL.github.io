# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _generate_hexagonal_16():
    """
    Generate a hexagonal cluster of 16 points.
    Center (1) + Ring 1 (6) + Ring 2 (9 points).
    """
    points = [[0.0, 0.0]]
    r1 = 1.0
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2 points in hex lattice (axial to cartesian)
    # Axial coords for ring 2:
    # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-2)? No.
    # Let's use the standard hex lattice generation for radius 2.
    hex_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Convert axial to cartesian (pointy top)
            x = 1.0 * (q + r/2.0)
            y = np.sqrt(3)/2 * r
            dist = np.sqrt(x*x + y*y)
            # We want points that are exactly in the second shell.
            # In a hex lattice, the distance from center in "lattice steps" is max(|q|, |r|, |q+r|)
            # But we want Euclidean distance.
            # Shell 1: dist 1.0 (6 pts)
            # Shell 2: dist sqrt(3) ~ 1.732 (6 pts) and dist 2.0 (6 pts)
            # Let's just collect all non-center, non-shell1 points with dist < 2.5
            if dist > 1.1 and dist < 2.1:
                # Check if it's a valid lattice point
                # q, r are integers.
                # The 12 points in shell 2 are:
                # (2,0)->(2,0), (1,1)->(1.5, 0.866), (0,2)->(1, 1.732), (-1,1)->(0.5, 0.866), (-2,0)->(-2,0), (-1,-1)->(-1.5, -0.866), (0,-2)->(-1, -1.732), (1,-1)->(0.5, -0.866), 
                # Wait, (2,0) is dist 2. (1,1) is dist sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25+0.75)=sqrt(3).
                # There are 6 points at dist sqrt(3) and 6 points at dist 2.
                hex_pts.append([x, y])
    
    # Remove duplicates if any (shouldn't be any)
    hex_pts = np.array(hex_pts)
    # We need 9 points.
    # To be symmetric, we should pick a subset.
    # The 12 points form a regular 12-gon? No, 6 at sqrt(3) and 6 at 2.
    # The angles are:
    # Dist 2: 0, 60, 120, 180, 240, 300
    # Dist sqrt(3): 30, 90, 150, 210, 270, 330
    # To pick 9 symmetrically, we can drop 3 points from the dist 2 set? Or mix?
    # Dropping 3 from dist 2 leaves 3 at dist 2 and 6 at dist sqrt(3).
    # This creates a shape with 3-fold symmetry?
    # Let's just sort by angle and pick every 12th? No, we need 9.
    # Let's pick the 9 closest to the origin? That would be the 6 at sqrt(3) and 3 at dist 2.
    dists = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists)
    selected = hex_pts[idx[:9]]
    points.extend(selected.tolist())
    pts = np.array(points)
    return pts

def _generate_concentric_rings_16():
    """
    Generate 16 points as two concentric regular octagons.
    """
    n_outer = 8
    n_inner = 8
    r_outer = 1.0
    r_inner = 0.6 # Optimized later
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_outer
    
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    pts = np.vstack([pts_outer, pts_inner])
    return pts

def _generate_random_16():
    return np.random.randn(16, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    d = 2
    
    # Hard deadline: 600s wall-clock, leave 10s margin
    deadline = time.time() + 590
    
    best_points = None
    best_score = -1.0
    
    rng = np.random.default_rng(42)
    
    initial_configs = [
        _generate_hexagonal_16(),
        _generate_concentric_rings_16(),
        _generate_random_16()
    ]
    
    # Normalize initial configs
    for i in range(len(initial_configs)):
        init = initial_configs[i]
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale
        initial_configs[i] = init
        
    # Langevin Dynamics Optimization
    # Objective: Maximize (d_min / d_max)^2
    # Surrogate: Smooth approximation using soft-min/soft-max or just raw min/max with gradient estimation.
    # Since min/max are non-differentiable, we use a "soft" version or finite differences.
    # A better approach for Langevin: Use the score function directly with finite difference gradients.
    # Or, use a smooth surrogate:
    # S = (soft_min(dists) / soft_max(dists))^2
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # soft_max(x) = 1/beta * log(sum(exp(beta*x)))
    # This is differentiable.
    
    beta = 20.0 # Sharpness of the soft min/max
    
    def _soft_score_and_grad(points):
        dists = spd.pdist(points)
        # Soft min and max
        # Clip to avoid overflow
        dists_clipped = np.clip(dists, 1e-15, 1e15)
        
        # Soft min
        # f_min = -1/beta * log(sum(exp(-beta * d)))
        # grad f_min w.r.t d = - exp(-beta*d) / sum(exp(-beta*d))
        
        # Soft max
        # f_max = 1/beta * log(sum(exp(beta * d)))
        # grad f_max w.r.t d = exp(beta*d) / sum(exp(beta*d))
        
        # R = (f_min / f_max)^2
        # dR/dd = 2 * (f_min / f_max) * ( (1/f_max) * df_min/dd - (f_min/f_max^2) * df_max/dd )
        
        # This is computationally heavy to do analytically for the chain rule through pdist.
        # Instead, we will use a finite difference approach for the gradient in the Langevin step,
        # or simply use the "score" as the potential and rely on the noise to explore,
        # but that's just random walk.
        
        # Better: Use the analytic gradient of the raw min/max if we track which pair is min/max.
        # But min/max changes discontinuously.
        
        # Alternative Strategy: 
        # Use a "repulsive" force model.
        # Force on point i:
        # 1. Repel all other points (to increase d_min)
        # 2. Attract all other points (to decrease d_max)
        # This is a balance.
        
        # Let's stick to a simpler, robust method: 
        # Random Restarts + Nelder-Mead was the previous approach.
        # The inspiration suggests Simulated Annealing.
        # Let's implement Simulated Annealing with a specific neighborhood structure.
        
        pass

    # Actually, let's implement a robust Simulated Annealing as suggested by inspirations, 
    # which is structurally different from Nelder-Mead.
    
    def _sa_optimize(init_points, time_budget):
        current_pts = init_points.copy()
        current_score = _score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        T = 0.1 # Initial temperature
        cooling_rate = 0.999
        min_T = 1e-6
        
        start_time = time.time()
        
        # Pre-compute indices for pdist to speed up? No, pdist is fast.
        
        while time.time() < start_time + time_budget and T > min_T:
            # Generate neighbor
            # Perturb a random subset of points
            n_perturb = np.random.choice([1, 2, 3, 4, 5, 8, 16], p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
            idx = rng.choice(n, size=n_perturb, replace=False)
            
            new_pts = current_pts.copy()
            # Add Gaussian noise
            noise_scale = T * 0.1
            new_pts[idx] += rng.normal(0, noise_scale, size=(n_perturb, d))
            
            # Re-center to prevent drift
            new_pts = new_pts - new_pts.mean(axis=0)
            
            new_score = _score(new_pts)
            
            delta = new_score - current_score
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            
            T *= cooling_rate
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return best_pts, best_score

    # Run SA on each initial config
    for init in initial_configs:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        time_per_run = min(remaining - 5, 180) # 3 minutes per run
        
        pts, score = _sa_optimize(init, time_per_run)
        
        if score > best_score:
            best_score = score
            best_points = pts
    
    # If time remains, do more random restarts with SA
    while time.time() < deadline - 20:
        init = _generate_random_16()
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale
        
        remaining = deadline - time.time()
        time_per_run = min(remaining - 5, 60) # 1 minute per random restart
        
        pts, score = _sa_optimize(init, time_per_run)
        
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
