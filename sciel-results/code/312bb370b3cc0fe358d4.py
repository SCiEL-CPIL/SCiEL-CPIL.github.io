"""
Strategy: Kinetic Annealing with Force-Based Repulsion and Newton Polish.
Instead of minimizing a smooth scalar objective via gradient descent (which can
get stuck in flat regions or local minima due to the 'soft' nature of the
objective), we simulate a physical system where circles repel each other and
are pushed away from the boundary. We use a temperature-like decay to allow
larger moves early and smaller moves later. Finally, we identify the active
constraints (pairs and boundary contacts at the minimum distance) and solve
the local linear system to precisely maximize the minimum distance.
"""
import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 300
DEADLINE = 140.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the maximum radius r for given centers."""
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _compute_forces(points, r_target, temp):
    """
    Compute force vectors for kinetic annealing.
    We want to push points apart if they are closer than 2*r_target,
    and push them inwards if they are closer to boundary than r_target.
    temp controls the magnitude of random perturbation to escape local minima.
    """
    forces = np.zeros_like(points)
    
    # Boundary forces
    # d_i = 1 - |p_i|
    # If d_i < r_target, push inwards (towards origin)
    r_pts = np.hypot(points[:, 0], points[:, 1])
    d_b = 1.0 - r_pts
    # Gradient of d_b w.r.t p is -p/|p|. We want to increase d_b.
    # Force direction: -p/|p|
    # Magnitude: proportional to violation (r_target - d_b)
    violations_b = np.maximum(0, r_target - d_b)
    # Normalize direction
    norms = np.maximum(r_pts, 1e-9)
    dir_b = -points / norms[:, np.newaxis]
    forces += dir_b * violations_b[:, np.newaxis] * 10.0  # Scale factor
    
    # Pairwise forces
    # d_ij = |p_i - p_j|
    # If d_ij < 2*r_target, push apart.
    # Force on i from j: (p_i - p_j) / d_ij * (2*r_target - d_ij)
    # To make it O(N) or efficient, we compute all pairs.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e-9)
    
    # We only care about pairs that are close.
    # Threshold: 2.5 * r_target to allow some interaction range
    threshold = 2.5 * r_target
    mask = dists < threshold
    np.fill_diagonal(mask, False)
    
    # Compute forces only for masked pairs
    # F_i = sum_{j != i} (p_i - p_j)/d_ij * (2*r_target - d_ij)
    # This is equivalent to summing over all j.
    # To avoid O(N^2) memory if N is large, we can do it in batches or just use full matrix since N=300 is small.
    # 300x300 is 90k elements, very small.
    
    # Force contribution matrix
    # f_ij = (2*r_target - d_ij) / d_ij if d_ij < 2*r_target else 0
    # Actually, let's use a smoother potential: V = (2r - d)^2 / 2 for d < 2r.
    # Force = gradient of V w.r.t p_i = (2r - d) * (p_i - p_j) / d
    # This is repulsive.
    
    # Calculate scalar factor for each pair
    factors = np.zeros_like(dists)
    active = dists < 2.0 * r_target
    factors[active] = (2.0 * r_target - dists[active]) / dists[active]
    
    # Force on i is sum_j factors[i,j] * (p_i - p_j)
    # diff[i,j] = p_i - p_j
    # forces[i] = sum_j factors[i,j] * diff[i,j]
    # This is a matrix-vector product: forces = factors @ points? No.
    # forces[i] = sum_j factors[i,j] * (points[i] - points[j])
    #            = points[i] * sum_j factors[i,j] - sum_j factors[i,j] * points[j]
    # Let S_i = sum_j factors[i,j]
    # Let W_i = sum_j factors[i,j] * points[j]
    # forces[i] = S_i * points[i] - W_i
    
    S = np.sum(factors, axis=1)
    W = factors @ points
    
    forces += S[:, np.newaxis] * points - W
    
    # Add random perturbation for annealing
    if temp > 0:
        forces += np.random.normal(0, temp, size=points.shape)
        
    return forces

def _newton_polish(points, max_iter=20, tol=1e-9):
    """
    Identify active constraints and solve the linear system to maximize min distance.
    This is a local optimization step.
    """
    pts = points.copy()
    for _ in range(max_iter):
        # Calculate current min distance
        dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_b = np.min(dist_boundary)
        min_p = np.min(dists) / 2.0
        r = min(min_b, min_p)
        
        # Identify active constraints
        # Boundary: indices where dist_boundary is close to min_b
        b_active = np.where(dist_boundary < min_b + 1e-6)[0]
        
        # Pairs: indices (i,j) where dists[i,j]/2 is close to min_p
        p_active_mask = dists < 2.0 * min_p + 1e-6
        np.fill_diagonal(p_active_mask, False)
        p_active_idx = np.argwhere(p_active_mask)
        
        # If no active constraints, we are in the interior, can move freely?
        # But we want to maximize r. If interior, we should push out?
        # The gradient of r w.r.t pts depends on which constraint is active.
        # If both are active, we solve for the direction that increases both.
        
        # Construct linear system A dx = b
        # We want to find a displacement dx that increases the active distances.
        # For boundary i: grad(d_i) = -p_i/|p_i|. We want grad(d_i) . dx >= 0.
        # For pair (i,j): grad(d_ij) = (p_i - p_j)/d_ij. We want grad(d_ij) . dx >= 0.
        
        # This is a feasibility problem. We can use a simple iterative projection or solve a least squares.
        # Let's try to find dx that maximizes the minimum increase.
        # Simpler: Just move in the direction of the average gradient of active constraints.
        
        grads = []
        for i in b_active:
            norm = np.linalg.norm(pts[i])
            if norm > 1e-9:
                grads.append(-pts[i] / norm)
        
        for i, j in p_active_idx:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-9:
                # Gradient w.r.t p_i
                grads.append(d / dist)
                # Gradient w.r.t p_j
                grads.append(-d / dist)
        
        if not grads:
            break
            
        grads = np.array(grads)
        
        # We want dx such that grads @ dx > 0 for all.
        # Find a vector that has positive dot product with all grads.
        # This is equivalent to finding a direction in the intersection of hemispheres.
        # A simple heuristic: sum of normalized grads?
        # Or solve min ||dx||^2 s.t. grads @ dx >= 1.
        # Using scipy.optimize.linprog? Too slow/complex.
        
        # Let's use a simple gradient ascent step with adaptive size.
        # Direction: average of gradients
        dir_vec = np.mean(grads, axis=0)
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-9:
            break
        dir_vec /= dir_norm
        
        # Determine step size
        # We can't exceed the distance to the next constraint.
        # Simple: try a small step and check if r increased.
        # Binary search for step size?
        
        best_step = 0.0
        best_r = r
        # Try a few steps
        for factor in [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]:
            step = 0.01 * factor # Base step 0.01
            new_pts = pts + dir_vec * step
            # Check validity
            new_r = _score(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_step = step
                best_pts = new_pts
        
        if best_step == 0.0:
            # Try random directions if stuck
            improved = False
            for _ in range(10):
                rand_dir = np.random.normal(0, 1, size=2)
                rand_dir /= np.linalg.norm(rand_dir)
                # Expand to N points
                # This is getting complicated. Let's just break if no improvement.
                pass
            break
        
        pts = best_pts
        
        if best_r - r < tol:
            break
            
    return pts, _score(pts)

def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * math.cos(angle)
            points[i, 1] = r * math.sin(angle)
    
    best_points = points.copy()
    best_score = _score(points)
    
    np.random.seed(42)
    
    # Kinetic Annealing Loop
    # Start with a rough estimate of r
    # Hex lattice spacing ~0.1, so r ~0.05
    r_target = best_score
    
    num_anneal_steps = 200
    current_temp = 0.05
    cooling_rate = 0.95
    
    for step in range(num_anneal_steps):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20: # Leave time for polish
            break
            
        # Compute forces
        forces = _compute_forces(best_points, r_target, current_temp)
        
        # Update positions
        # Step size proportional to temperature and force magnitude
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-9:
            break
        step_size = 0.01 * current_temp / max_force
        # Limit step size to avoid jumping over constraints
        step_size = min(step_size, 0.05)
        
        new_points = best_points + forces * step_size
        
        # Clamp to disk
        r_pts = np.hypot(new_points[:, 0], new_points[:, 1])
        mask = r_pts > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / r_pts[mask, np.newaxis] * (1.0 - 1e-9)
            
        # Calculate new score
        new_score = _score(new_points)
        
        # Accept if better, or with some probability if worse (Metropolis)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
            r_target = new_score
        else:
            # Accept with probability exp((new_score - best_score) / temp)
            # Scores are small, so delta is small.
            # Use a scaled temperature for acceptance
            delta = best_score - new_score
            if delta < 1e-6:
                prob = 1.0
            else:
                prob = math.exp(-delta / (current_temp * 0.1))
                
            if np.random.random() < prob:
                best_points = new_points
                # Don't update r_target if worse, keep it as the target to push towards
                # Actually, r_target should be the current best to guide forces
                # But if we accept a worse state, we should probably lower r_target slightly?
                # No, r_target is the desired radius. We want to reach it.
                # If we are at a worse state, the forces will push us back towards the best config.
                
        # Cool down
        current_temp *= cooling_rate
        
        # Periodically polish
        if step % 20 == 0 and step > 0:
            polished, p_score = _newton_polish(best_points)
            if p_score > best_score:
                best_score = p_score
                best_points = polished
                r_target = best_score
    
    # Final Polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        polished, p_score = _newton_polish(best_points, max_iter=50)
        if p_score > best_score:
            best_score = p_score
            best_points = polished
    
    # Final check: ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6
    
    return best_points

# Dummy to satisfy import if needed, though not used
def _unused_least_squares_helper():
    pass

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
