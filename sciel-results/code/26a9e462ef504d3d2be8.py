import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Use broadcasting for pairwise distances to speed up
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def _initial_hexagonal():
    """
    Generate an initial configuration based on a hexagonal lattice clipped to the unit circle.
    This is a very good starting point for circle packing.
    """
    points = []
    # Hexagonal lattice parameters
    # For packing radius r, distance between centers is 2r.
    # We want to fit 50 circles.
    # Approximate area of 50 circles with radius r: 50 * pi * r^2
    # Area of unit circle: pi
    # r^2 ~ 1/50 => r ~ 0.14.
    # Distance d ~ 0.28.
    
    # Let's generate a grid of hexagonal points and select the best 50 inside the circle.
    # Or better, generate points and let the optimizer move them.
    
    # Simple heuristic: place points on concentric rings with hexagonal symmetry initially.
    # But the prompt asks for a different strategy. Force-directed from random/hex is better.
    # Let's create a "hex-like" random initialization.
    
    # Generate many candidate points in the unit disk
    # We will pick the 50 that are most spread out.
    
    # Actually, a simpler robust initialization:
    # Place 1 at center.
    # Place 6 around it at distance d.
    # Place 12 around that...
    # This is the ring structure again.
    
    # Let's try a "Random + Repulsion" initialization.
    # Start with random points, apply a few steps of force-directed relaxation to get a decent spread.
    
    return None # Placeholder, we will use a specific heuristic in construct

def _force_directed_relaxation(points, steps=50, dt=0.05, k_repel=1.0, k_attract=0.1):
    """
    Perform force-directed relaxation to maximize minimum distance.
    We simulate repulsion between all pairs and a weak attraction to the center to keep them in.
    """
    pts = points.copy()
    for _ in range(steps):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-6:
                    # Jitter if overlapping
                    d_vec = np.random.uniform(-1, 1, 2)
                    d = np.linalg.norm(d_vec)
                
                # Repulsive force: inverse square or linear? 
                # For maximizing min distance, a strong repulsion at short range is good.
                # Force magnitude ~ 1/d^2 or (target_d - d) if d < target_d.
                # Let's use a soft potential: V = 1/d. Force = 1/d^2.
                f_mag = k_repel / (d * d)
                f_vec = f_mag * (d_vec / d)
                
                forces[i] += f_vec
                forces[j] -= f_vec
            
            # Weak attraction to center to prevent drift
            r_i = np.linalg.norm(pts[i])
            if r_i > 1e-6:
                f_attract = -k_attract * pts[i] / r_i
                forces[i] += f_attract
            else:
                forces[i] += np.random.uniform(-1, 1, 2) * 0.01
        
        # Update positions
        pts += dt * forces
        
        # Constrain to unit circle
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 0.95: # Keep slightly inside to allow movement
                pts[i] *= 0.95 / norms[i]
                
    return pts

def _local_polish(points, time_limit=10.0):
    """
    Use L-BFGS-B to polish the configuration.
    Objective: maximize min(d_ij/2, 1-|c_i|).
    We use a smooth approximation of the min function: log-sum-exp trick.
    """
    start_time = time.time()
    x0 = points.flatten()
    
    def fun_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        
        pts = x.reshape(N, 2)
        
        # Compute margins
        # Boundary margins: 1 - |c_i|
        norms = np.linalg.norm(pts, axis=1)
        f = 1.0 - norms
        
        # Pairwise margins: |c_i - c_j| / 2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        g = dists / 2.0
        
        all_margins = np.concatenate([f.ravel(), g.ravel()])
        
        # We want to maximize min(all_margins).
        # Minimize -min(all_margins).
        # Smooth approximation: -(-logsumexp(-beta * all_margins) / beta)
        # Or simply: - (max_margin - (1/beta) * log(sum(exp(beta*(max_margin - margin)))) )
        # This approximates the max of the margins? No, we want min.
        # min(x) = -max(-x).
        # So we want to maximize min(margins).
        # Let y = -margins. We want to minimize max(y).
        # Smooth max(y) = (1/beta) * log(sum(exp(beta * y))).
        # So objective = (1/beta) * log(sum(exp(beta * (-margins)))).
        
        beta = 50.0
        neg_margins = -all_margins
        max_neg = np.max(neg_margins)
        shifted = beta * (neg_margins - max_neg)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp_shifted))
        
        obj = max_neg + (1.0 / beta) * log_sum_exp
        
        # Gradient
        p = exp_shifted / np.sum(exp_shifted)
        grad = np.zeros_like(x)
        
        # Boundary part
        # d obj / d x = (1/beta) * d log_sum_exp / d x
        # d log_sum_exp / d x_i = sum_j p_j * d(-margin_j)/d x_i
        # For boundary i: margin = 1 - |x_i|. d(-margin)/d x_i = x_i / |x_i|.
        for i in range(N):
            w = p[i]
            if norms[i] > 1e-10:
                grad[2*i:2*i+2] += w * (pts[i] / norms[i])
        
        # Pairwise part
        # For pair (i,j) in g: margin = |x_i - x_j| / 2.
        # d(-margin)/d x_i = -(x_i - x_j) / (2 * |x_i - x_j|)
        # Note: g.ravel() order corresponds to upper triangle? 
        # Our g is full matrix. all_margins includes all i!=j.
        # In all_margins, index N + i*N + j corresponds to g[i,j].
        # Wait, g is N x N. ravel is row-major.
        # Index in all_margins:
        # 0..N-1: boundary
        # N..N*N-1: pairwise.
        # Offset in pairwise: (i*N + j)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                idx = N + i*N + j
                w = p[idx]
                if w < 1e-15:
                    continue
                d = dists[i, j]
                if d > 1e-10:
                    vec = pts[i] - pts[j]
                    # d(-margin)/d x_i = -vec / (2d)
                    grad[2*i:2*i+2] += w * (-vec / (2.0 * d))
                    # d(-margin)/d x_j = vec / (2d)
                    grad[2*j:2*j+2] += w * (vec / (2.0 * d))
                    
        return obj, grad

    try:
        result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x.reshape(N, 2)
    except Exception:
        pass
    
    return points

def construct():
    deadline = time.time() + DEADLINE
    
    best_r = 0.0
    best_points = None
    
    # Strategy: Multiple random restarts with force-directed relaxation, then local polish.
    # This is a "population" of initial states, but we process them sequentially to save memory/complexity.
    
    num_restarts = 10
    restart_budget = 5.0 # seconds per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0: # Leave time for final polish
            break
            
        # 1. Initialization
        # Random points in unit disk, but ensure minimum initial separation to help FDR.
        # We can use a simple rejection sampling or just random.
        # Let's use a "hexagonal-ish" random start:
        # Place points on a grid, add noise.
        
        # Better: Start with a known good configuration (rings) and add noise.
        # This combines the benefits of both: good topology from rings, flexibility from noise.
        
        # Ring counts: 1, 6, 12, 18, 13
        ring_counts = [1, 6, 12, 18, 13]
        ring_radii = [0.0, 0.25, 0.5, 0.75, 0.95]
        
        pts = np.zeros((N, 2))
        idx = 0
        for i, count in enumerate(ring_counts):
            r = ring_radii[i]
            for k in range(count):
                angle = 2 * np.pi * k / count + np.random.uniform(-0.1, 0.1)
                pts[idx, 0] = r * np.cos(angle)
                pts[idx, 1] = r * np.sin(angle)
                idx += 1
        
        # Add some global noise
        pts += np.random.uniform(-0.05, 0.05, (N, 2))
        
        # 2. Force Directed Relaxation
        # This helps untangle and spread points.
        pts_relaxed = _force_directed_relaxation(pts, steps=20, dt=0.02, k_repel=0.5, k_attract=0.05)
        
        # 3. Local Polish
        # Calculate time remaining for this polish
        polish_limit = min(restart_budget, deadline - time.time() - 5.0)
        if polish_limit < 1.0:
            continue
            
        pts_polished = _local_polish(pts_relaxed, time_limit=polish_limit)
        
        # 4. Evaluate
        r = _compute_radius(pts_polished)
        
        if r > best_r:
            best_r = r
            best_points = pts_polished
            
        # Optional: Binary search polish on the best candidate
        # We can do a final binary search on the radius r to see if we can push it higher
        # by treating it as a feasibility problem?
        # Actually, the local polish already maximizes the min margin.
        # A binary search on r with a constraint solver (like SLSQP) might be faster/more robust.
        # But L-BFGS-B on the smooth min is usually very effective.

    # Final check and safety
    if best_points is None:
        # Fallback to simple rings
        ring_counts = [1, 6, 12, 18, 13]
        ring_radii = [0.0, 0.25, 0.5, 0.75, 0.95]
        best_points = np.zeros((N, 2))
        idx = 0
        for i, count in enumerate(ring_counts):
            r = ring_radii[i]
            for k in range(count):
                angle = 2 * np.pi * k / count
                best_points[idx, 0] = r * np.cos(angle)
                best_points[idx, 1] = r * np.sin(angle)
                idx += 1
        best_r = _compute_radius(best_points)
        
    # Ensure valid output
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.999:
            best_points[i] *= 0.999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = _compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
