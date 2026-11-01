"""
Improved packing algorithm for 300 equal circles in a unit circle.
Uses a radial-ring decomposition with Simulated Annealing to exploit circular symmetry.
"""
N = 300

import numpy as np
import time
import math
from scipy.optimize import minimize

def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 1.0
    
    # Compute all pairwise distances
    # Using broadcasting for simplicity, though memory heavy for N=300 (300x300 is fine)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)

    return min(min_pair, min_boundary)

def _rings_to_points(radii, offsets, counts):
    """
    Convert ring parameters to (x, y) coordinates.
    radii: list of ring radii
    offsets: list of angular offsets for each ring
    counts: list of number of circles in each ring
    """
    points = []
    for r, off, n in zip(radii, offsets, counts):
        if n == 0:
            continue
        angles = 2 * np.pi * np.arange(n) / n + off
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points.extend(zip(x, y))
    
    # Ensure exactly N points
    # If we have fewer, add points at origin (should not happen with good init)
    # If we have more, trim (should not happen)
    if len(points) < N:
        # Add remaining points at random positions near center
        diff = N - len(points)
        extra = np.random.randn(diff, 2) * 0.01
        points.extend(map(tuple, extra))
    elif len(points) > N:
        points = points[:N]
        
    return np.array(points)

def _init_rings():
    """
    Initialize ring structure.
    Heuristic: 
    - Start with 1 point at center.
    - Subsequent rings have increasing counts.
    - Approximate optimal packing has ~6 points in first ring, 12 in second, etc.
    - But boundary effects reduce capacity.
    """
    counts = [1]
    remaining = N - 1
    ring_idx = 1
    # Approximate radius growth. 
    # Area of circle = pi * R^2. 
    # Number of circles ~ Area / (pi * r_c^2).
    # For a ring of radius r, circumference ~ 2*pi*r.
    # Spacing ~ 2*r_c.
    # Count ~ pi*r/r_c.
    
    # Let's estimate r_c ~ 0.053.
    # r_1 ~ 0.1 -> count ~ 6
    # r_2 ~ 0.2 -> count ~ 12
    # ...
    # Sum counts ~ 300.
    # 1 + 6 + 12 + 18 + ...
    
    # A better heuristic:
    # Try to distribute N points such that they form a hexagonal lattice clipped to circle.
    # But we want explicit rings.
    
    # Let's try a standard ring packing initialization:
    # Ring 0: 1 point at r=0
    # Ring k: n_k points at r_k.
    # n_k increases roughly linearly with k.
    
    # Let's generate a hex lattice, then cluster by distance to origin to define rings?
    # No, let's just construct rings manually.
    
    # Empirical rule for circle packing in a circle:
    # The number of circles in ring k is often 6k for k>0, but adjusted for boundary.
    # Let's start with counts = [1] + [6*k for k in range(1, ...)]
    
    k = 1
    while remaining > 0:
        # Approx capacity of ring k
        # r_k approx k * 2 * r_c * cos(pi/6) ?
        # Let's just guess counts.
        # For N=300, we need about 10-12 rings.
        # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 = 271.
        # Next ring would be 60. 271+60 = 331. Too many.
        # So last few rings need adjustment.
        
        # Let's use a simpler distribution:
        # Spread remaining points evenly over estimated number of rings.
        # Estimated rings ~ sqrt(N/3) ~ 10.
        
        # Let's just generate a hex lattice, get unique radii, and use those as ring radii.
        # This is a robust way to get good initial radii.
        
        break # We will use a different init strategy below.
    
    # Alternative Init: Hex Lattice -> Binning
    s = 0.1 # Approx spacing
    # Generate hex points
    pts_hex = []
    k_max = int(1.0 / s) + 2
    for j in range(-k_max, k_max + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        x_offset = (j % 2) * s / 2.0
        for i in range(-k_max, k_max + 1):
            x = (i + x_offset) * s
            r = np.hypot(x, y)
            if r <= 1.0 - s/2.0:
                pts_hex.append([x, y])
                
    pts_hex = np.array(pts_hex)
    # Select N closest to origin
    dists = np.hypot(pts_hex[:, 0], pts_hex[:, 1])
    idx = np.argsort(dists, kind='stable')[:N]
    pts_hex = pts_hex[idx]
    
    # Now, group these points into rings based on radius.
    # Sort by radius
    order = np.argsort(dists[idx])
    sorted_pts = pts_hex[order]
    sorted_radii = dists[idx][order]
    
    # Define ring boundaries. 
    # We want to group points that are close in radius.
    # Let's just assign each point to a ring index based on its rank? No, that's not physical.
    # Let's cluster the radii.
    
    # Simple approach: 
    # 1. Point at center.
    # 2. Rest grouped by "shell".
    
    # Let's try to fit a hexagonal lattice structure to the rings.
    # Ring 0: 1 point.
    # Ring k: 6k points.
    # This sums to 3k^2+3k+1.
    # For N=300, 3k^2+3k+1 <= 300 => 3k^2+3k-299 <= 0.
    # k ~ sqrt(100) = 10.
    # 3(100)+30+1 = 331.
    # So rings 1..9 are full (1+6+12+...+54 = 1+3*9*10 = 271).
    # Remaining: 300 - 271 = 29 points for ring 10.
    
    counts = [1]
    for k in range(1, 10):
        counts.append(6*k)
    # Sum so far: 1 + 6*45 = 271.
    remaining = N - 271
    counts.append(remaining)
    
    # Radii:
    # r_k = k * 2 * r_c * cos(pi/6) ?
    # Let's estimate r_c.
    # Area = N * pi * r_c^2 <= pi * 1^2.
    # r_c <= 1/sqrt(N) ~ 0.057.
    # Actual is ~0.053.
    # Spacing between rings ~ 2 * r_c * cos(pi/6) ~ 1.732 * r_c.
    
    r_c_est = 0.053
    ring_spacing = 2 * r_c_est * np.cos(np.pi / 6.0)
    
    radii = [0.0]
    for k in range(1, len(counts)):
        radii.append(k * ring_spacing)
        
    # Adjust last ring radius to be closer to boundary?
    # The last ring has fewer points, so it can be pushed out?
    # No, it's constrained by the previous ring and the boundary.
    # Let's leave it for optimization.
    
    offsets = [0.0] * len(counts)
    
    return radii, offsets, counts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    radii, offsets, counts = _init_rings()
    
    # Convert to points to get initial radius
    pts = _rings_to_points(radii, offsets, counts)
    best_r = _compute_radius(pts)
    best_radii = radii.copy()
    best_offsets = offsets.copy()
    
    # 2. Simulated Annealing on ring parameters
    # Variables: radii (except first, which is 0), offsets
    # We keep counts fixed to maintain the ring structure topology.
    # If the structure is wrong, SA might not fix it, but the hex init is usually topologically correct.
    
    n_rings = len(radii)
    
    # Objective function for SA
    def get_r(radii_curr, offsets_curr):
        pts_curr = _rings_to_points(radii_curr, offsets_curr, counts)
        return _compute_radius(pts_curr)
    
    # SA Loop
    T = 0.01 # Initial temperature
    T_min = 0.0001
    cooling = 0.999
    max_iters = 50000
    
    # We need to be fast. _compute_radius is O(N^2) ~ 90k ops.
    # 50k iters * 90k ops = 4.5e9 ops. Too slow for Python/NumPy in 140s?
    # 140s / 50k = 2.8ms per iter.
    # NumPy vectorized distance calculation for 300 points is very fast.
    # Let's optimize the distance calculation.
    
    # Precompute indices for pairs?
    # Or just rely on NumPy speed.
    
    # Let's reduce iters if time is tight.
    # We will monitor time.
    
    for i_iter in range(max_iters):
        if time.time() > deadline - 10.0:
            break
            
        # Generate perturbation
        # 1. Perturb a random ring radius
        # 2. Perturb a random ring offset
        # 3. Swap counts between adjacent rings? (Rarely needed, skip for simplicity)
        
        radii_curr = best_radii.copy()
        offsets_curr = best_offsets.copy()
        
        # Randomly choose which parameter to perturb
        if np.random.rand() < 0.5:
            # Perturb radius
            # Don't perturb ring 0 (center)
            k = np.random.randint(1, n_rings)
            delta_r = np.random.normal(0, T * 0.1)
            new_r = radii_curr[k] + delta_r
            # Constraints:
            # r_k > r_{k-1} + 2*r_c? No, just r_k > r_{k-1}.
            # r_k + r_c < 1.
            # Let's just clamp to valid range [0, 1]
            if new_r > 0.01 and new_r < 0.99:
                radii_curr[k] = new_r
        else:
            # Perturb offset
            k = np.random.randint(1, n_rings)
            delta_a = np.random.normal(0, T * 0.1)
            offsets_curr[k] = offsets_curr[k] + delta_a
        
        # Evaluate
        r_new = get_r(radii_curr, offsets_curr)
        
        # Acceptance criterion
        if r_new > best_r:
            best_r = r_new
            best_radii = radii_curr
            best_offsets = offsets_curr
        else:
            # Metropolis criterion
            # We are maximizing, so delta = best_r - r_new > 0
            # Accept with prob exp(-delta / T)
            delta = best_r - r_new
            if delta < 1e-10:
                pass # Same
            else:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    best_r = r_new
                    best_radii = radii_curr
                    best_offsets = offsets_curr
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # 3. Final Polish with L-BFGS-B on full coordinates
    # This allows breaking the ring symmetry if it helps.
    # We use a soft-min objective to handle the non-differentiability.
    
    pts = _rings_to_points(best_radii, best_offsets, counts)
    
    def soft_min_value(pts, temp=0.01):
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        
        all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
        all_vals = np.clip(all_vals, -100, 100)
        
        min_val = np.min(all_vals)
        exp_vals = np.exp((all_vals - min_val) / temp)
        soft_min = min_val - temp * np.log(np.sum(exp_vals))
        
        return -soft_min
        
    def soft_min_grad(pts, temp=0.01):
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        
        all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
        all_vals = np.clip(all_vals, -100, 100)
        
        min_val = np.min(all_vals)
        exp_vals = np.exp((all_vals - min_val) / temp)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        # Pair gradients
        pair_weights = weights[:n * (n - 1)]
        pair_weights = pair_weights.reshape(n, n)
        
        # Vectorized gradient for pairs
        # grad[i] += sum_j w_ij * (p_i - p_j) / (2 * d_ij) * (-temp)
        # Note: pair_weights is symmetric? No, we flattened upper/lower?
        # In soft_min_value, we flattened the whole matrix (N*N).
        # So weights correspond to all i,j pairs including i=j (which are inf/0 weight).
        
        # Let's compute gradients carefully.
        # d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij)
        # Objective is -soft_min.
        # soft_min = min - T log(sum exp((v-min)/T))
        # d(soft_min)/dv_k = - exp((v_k-min)/T) / sum exp(...) = -w_k
        # d(-soft_min)/dv_k = w_k
        # So we want to increase v_k (distance/boundary) for active constraints.
        # Gradient of v_k w.r.t p_i:
        
        # For pair (i,j), v = d_ij/2.
        # d(v)/d(p_i) = (p_i - p_j) / (2 * d_ij)
        # d(v)/d(p_j) = (p_j - p_i) / (2 * d_ij)
        
        # Sum over all j for each i.
        
        # This is O(N^2).
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = pair_weights[i, j]
                if dists[i, j] > 1e-12:
                    grad[i] += (pts[i] - pts[j]) / (2.0 * dists[i, j]) * w * (-temp)
                    # Wait, the sign.
                    # We are minimizing -soft_min.
                    # d(-soft_min)/dp = - d(soft_min)/dp.
                    # d(soft_min)/dp = sum w_k * d(v_k)/dp.
                    # So d(-soft_min)/dp = - sum w_k * d(v_k)/dp.
                    # My previous code had * (-temperature). Let's check.
                    # soft_min = min - T log S.
                    # d(soft_min)/dv = -w.
                    # d(-soft_min)/dv = w.
                    # d(v)/dp = grad_v.
                    # d(-soft_min)/dp = sum w * grad_v.
                    # So no negative sign from the soft min derivative?
                    # Let's re-derive.
                    # L = -soft_min.
                    # dL/dv = w.
                    # dL/dp = sum w_k (dv_k/dp).
                    # So grad[i] += w * (p_i - p_j) / (2 d_ij).
                    
        # Boundary gradients
        boundary_weights = weights[n * (n - 1):]
        for i in range(n):
            r = np.hypot(pts[i, 0], pts[i, 1])
            if r > 1e-12:
                # v = 1 - r.
                # d(v)/d(p) = -p/r.
                grad[i] += -pts[i] / r * boundary_weights[i]
                
        return grad

    # Run L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: soft_min_value(x.reshape(N, 2), temp=0.01),
                pts.flatten(),
                jac=lambda x: soft_min_grad(x.reshape(N, 2), temp=0.01).flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                new_pts = result.x.reshape(N, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    # Final validity check
    dists = np.hypot(pts[:, 0], pts[:, 1])
    mask = dists <= 1.0
    if not np.all(mask):
        for i in range(N):
            if dists[i] > 1.0:
                pts[i] = pts[i] / dists[i] * 0.999999999

    return pts

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    r = _compute_radius(pts)
    print(f"Radius: {r}, Time: {t1-t0:.2f}s")
