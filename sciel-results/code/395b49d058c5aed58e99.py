import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def compute_score(points):
    # Returns the radius r
    if points.shape != (N, 2):
        return 0
    if not np.all(np.isfinite(points)):
        return 0
    
    # Check if inside unit circle
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        # Invalid, but we can calculate a "penalized" score or just 0
        # For optimization, we use soft penalty. For final score, it's 0 if invalid.
        # But the evaluator says "A centre on or outside ... makes the candidate invalid."
        # So we must ensure validity.
        pass

    # Pairwise distances
    idx_i, idx_j = np.triu_indices(N, k=1)
    diff = points[idx_i] - points[idx_j]
    dists = np.linalg.norm(diff, axis=1)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    min_boundary_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if min_boundary_dist < 0:
        return 0 # Or negative? The contract says invalid.
        
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_score = 0
    best_points = None
    
    # Objective function with gradient
    def obj_and_grad(x, lam):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        idx_i, idx_j = np.triu_indices(N, k=1)
        diff = p[idx_i] - p[idx_j]
        dists = np.linalg.norm(diff, axis=1)
        dists_safe = np.maximum(dists, 1e-8)
        
        norms = np.linalg.norm(p, axis=1)
        norms_safe = np.maximum(norms, 1e-8)
        boundary_dists = 1.0 - norms
        
        # Log-sum-exp stability
        all_dists = np.concatenate([dists, boundary_dists])
        neg_lam_dists = -lam * all_dists
        max_neg = np.max(neg_lam_dists)
        if not np.isfinite(max_neg):
            return 1e10, np.zeros_like(x)
            
        exp_terms = np.exp(neg_lam_dists - max_neg)
        sum_exp = np.sum(exp_terms)
        val = max_neg + np.log(sum_exp)
        
        # Gradient
        grad = np.zeros_like(p)
        weights = exp_terms / sum_exp
        
        # Pairwise
        # diff is p_i - p_j
        unit_vec = diff / dists_safe[:, np.newaxis]
        coeff = weights[:len(dists)] * (-lam)
        
        np.add.at(grad, idx_i, coeff[:, np.newaxis] * unit_vec)
        np.subtract.at(grad, idx_j, coeff[:, np.newaxis] * unit_vec)
        
        # Boundary
        unit_p = p / norms_safe[:, np.newaxis]
        coeff_b = weights[len(dists):] * (-lam)
        # d(1-||p||)/dp = -unit_p
        # Term: coeff_b * (-unit_p) = -coeff_b * unit_p
        # Wait, let's re-verify sign.
        # L = log(sum exp(-lam d))
        # dL/dp = -lam * sum(w * dd/dp)
        # dd/dp for boundary = -unit_p
        # dL/dp = -lam * w * (-unit_p) = lam * w * unit_p
        # coeff_b is w * (-lam).
        # So dL/dp = -coeff_b * unit_p.
        grad += -coeff_b[:, np.newaxis] * unit_p
        
        return val, grad

    # Initialization strategies
    
    inits = []
    
    # 1. Hexagonal Lattice
    # Generate a dense hex grid and pick N closest to center
    spacing = 0.15 # Initial guess, will be adjusted
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points strictly inside
        mask = radius < 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Sort by radius to get most central N
            idx = np.argsort(radius, kind="stable")[:N]
            inits.append(points[idx].copy())
            break
        spacing *= 0.95
        if spacing < 0.05: break
        
    # 2. Concentric Rings
    # Try to mimic the optimal structure
    # 1 center, 6, 12, 18, 13
    ring_counts = [1, 6, 12, 18, 13]
    # Estimate radii
    # Total area 50 * pi * r^2 = pi * 1^2 => r ~ 1/sqrt(50) ~ 0.14
    # But packing efficiency is lower.
    # Let's just use evenly spaced radii for the rings
    ring_radii = np.linspace(0.2, 0.9, len(ring_counts))
    
    pts = []
    for i, count in enumerate(ring_counts):
        r = ring_radii[i]
        if count == 1:
            pts.append([0, 0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Add a small random offset to break symmetry if needed, or keep symmetric
            # Symmetric is often better for local optimization starting point
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.append(np.column_stack([x, y]))
            
    inits.append(np.vstack(pts))
    
    # 3. Random
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.uniform(-0.9, 0.9, (N, 2))
        norms = np.linalg.norm(pts, axis=1)
        # Project outside points back inside
        outside = norms > 0.9
        if np.any(outside):
            pts[outside] = pts[outside] / norms[outside, np.newaxis] * 0.9
        inits.append(pts)

    # Optimization Loop
    for init in inits:
        if time.time() > deadline:
            break
            
        x = init.flatten()
        
        # Stepwise annealing of lambda
        for lam in [10, 50, 100, 500]:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    lambda x: obj_and_grad(x, lam),
                    x,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                x = res.x
            except Exception:
                pass
                
        # Evaluate final score
        p = x.reshape(-1, 2)
        
        # Check validity
        if not np.all(np.isfinite(p)):
            continue
            
        norms = np.linalg.norm(p, axis=1)
        if np.any(norms > 1.0 + 1e-6):
            # If slightly outside, shrink?
            # The evaluator rejects invalid.
            # Let's try to project back if very close?
            # No, better to reject and rely on other inits.
            # Or, we can clamp: p = p / np.maximum(norms, 1.0)[:, np.newaxis]
            # But this might break distances.
            # Let's just check score.
            pass
            
        score = compute_score(p)
        
        if score > best_score:
            best_score = score
            best_points = p.copy()

    if best_points is None:
        # Fallback to first init
        best_points = inits[0]
        
    return best_points
