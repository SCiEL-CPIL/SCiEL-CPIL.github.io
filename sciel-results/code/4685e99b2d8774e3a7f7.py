import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: Radial Ring Decomposition
    # 1. Determine the number of rings and distribution of circles per ring
    # 2. For a given ring structure, solve for optimal radii and angular offsets
    # 3. Refine with local search
    
    # Initial guess for ring structure
    # Total area of 300 circles of radius r is 300 * pi * r^2
    # Area of unit circle is pi
    # r^2 ~ 1/300 -> r ~ 0.0577 (upper bound, actual is ~0.053)
    # Packing density ~ 0.9 -> r ~ 0.055
    
    # Let's try to find a good ring structure.
    # Common structures for 300 circles:
    # Center + rings.
    # Let's parameterize by number of rings M.
    # Circles per ring k_j. Sum k_j + 1 (center) = 300.
    # To maximize radius, we want to utilize the space efficiently.
    # Inner rings have less circumference, so fewer circles.
    
    # We will search over a few plausible ring distributions.
    # A good heuristic for k_j is proportional to j (radius).
    
    def generate_ring_configurations(n_total):
        configs = []
        # Try different numbers of rings
        for m_rings in range(10, 25):
            # Distribute n_total circles into m_rings + 1 (center)
            # Let's say ring 0 is center (1 circle), rings 1..m_rings have k_j circles
            # k_j should grow roughly linearly with j
            # We want sum(k_j) = n_total - 1
            # k_j = floor(c * j)
            # Try different scales
            
            # Simple approach: distribute evenly with bias to outer rings
            # k_j = round( (n_total - 1) * j / sum(1..m_rings) )
            # Adjust to ensure sum is correct
            
            base = np.arange(1, m_rings + 1)
            # Normalize to sum to n_total - 1
            target_sum = n_total - 1
            scale = target_sum / np.sum(base)
            k_vals = np.round(base * scale).astype(int)
            
            # Adjust rounding errors
            diff = target_sum - np.sum(k_vals)
            # Add/remove from the largest rings to fix sum
            if diff > 0:
                # Add to rings with largest fractional parts or just largest rings
                # Simple: add to the last 'diff' rings
                idxs = np.argsort(k_vals)[::-1][:diff]
                k_vals[idxs] += 1
            elif diff < 0:
                idxs = np.argsort(k_vals)[:abs(diff)]
                k_vals[idxs] -= 1
                # Ensure no negative counts
                k_vals = np.clip(k_vals, 1, None)
                # Re-adjust if clipping changed sum
                new_diff = target_sum - np.sum(k_vals)
                if new_diff != 0:
                     # Just force a valid config, e.g. add to center? No, center is fixed 1.
                     # Add to outer rings
                     if new_diff > 0:
                         k_vals[-1] += new_diff
                     else:
                         k_vals[-1] += new_diff # This might make it 0, bad.
                         # Better to just skip this config if invalid
                         k_vals = None
                         continue
            
            if k_vals is None or np.any(k_vals < 1):
                continue
                
            configs.append(k_vals)
            
        # Also add a few specific known good patterns if N is close
        # For N=300, a hexagonal-like structure is good.
        # Hex numbers: 1, 7, 19, 37, 61, 91, 127, 169, 217, 271, 331
        # 271 is close to 300.
        # Rings for hex packing: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54...
        # Sum: 1+6+12+18+24+30+36+42+48+54 = 271.
        # We have 29 circles left.
        # We can add a partial 11th ring or distribute.
        # Let's generate a hex-based config
        hex_rings = [6 * i for i in range(1, 10)] # 6, 12, ..., 54
        sum_hex = sum(hex_rings) + 1 # 271
        remaining = N - sum_hex # 29
        # Add remaining to the outer ring or create a new partial ring
        # Option 1: Add to outer ring
        hex_rings[9] += remaining
        configs.append(np.array(hex_rings))
        
        # Option 2: Distribute remaining among last few rings
        hex_rings2 = [6 * i for i in range(1, 10)]
        for i in range(remaining):
            hex_rings2[-1 - (i % 3)] += 1 # Spread among last 3
        configs.append(np.array(hex_rings2))

        return configs

    configs = generate_ring_configurations(N)
    
    best_radius = 0.0
    best_pts = None
    
    def compute_radius(pts):
        if len(pts) != N:
            return 0.0
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    for k_vals in configs:
        if time.time() > deadline - 20:
            break
            
        # k_vals: array of circle counts for rings 1..M
        # Total circles = 1 (center) + sum(k_vals)
        # Radii r_j for j=1..M
        # Center at (0,0)
        
        # Optimization variables: radii R_1, ..., R_M
        # Objective: Maximize min(
        #   min_j (R_{j+1} - R_j - r_c),  # Radial gap between rings (approx)
        #   min_j (2 * R_j * sin(pi / k_j) / 2 - r_c), # Tangential gap within ring
        #   1 - R_M - r_c # Boundary
        # )
        # This is a simplified model. Real optimization needs to handle angular shifts.
        
        # Let's use a direct numerical optimization for the ring positions.
        # Variables: Radii (M variables).
        # For a given set of radii, we can calculate the maximum possible radius r_c.
        # r_c is limited by:
        # 1. Boundary: R_M + r_c <= 1  => r_c <= 1 - R_M
        # 2. Inner ring tangential: k_j * r_c <= pi * R_j  => r_c <= pi * R_j / k_j
        #    (Actually 2 * r_c <= 2 * R_j * sin(pi/k_j) => r_c <= R_j * sin(pi/k_j))
        # 3. Radial gap: R_{j+1} - R_j >= 2 * r_c  => r_c <= (R_{j+1} - R_j) / 2
        #    Also R_1 >= 2 * r_c (from center) => r_c <= R_1 / 2
        
        # We want to maximize r_c subject to these constraints.
        # This is equivalent to minimizing the maximum violation, or just optimizing R_j to equalize the constraints.
        
        # Let's define a function that, given Radii, returns the max possible r_c.
        # Then we optimize Radii to maximize this r_c.
        
        def max_radius_for_radii(R):
            # R is array of radii for rings 1..M
            # Constraints for r_c:
            limits = []
            # Center to Ring 1
            limits.append(R[0] / 2.0)
            # Ring j to Ring j+1
            for j in range(len(R) - 1):
                limits.append((R[j+1] - R[j]) / 2.0)
            # Ring j tangential
            for j in range(len(R)):
                k_j = k_vals[j]
                if k_j > 0:
                    limits.append(R[j] * np.sin(np.pi / k_j))
            # Boundary
            limits.append(1.0 - R[-1])
            
            return np.min(limits)
        
        # We want to find R that maximizes max_radius_for_radii(R).
        # This is a max-min problem.
        # We can use scipy.optimize.minimize to minimize -max_radius_for_radii(R).
        # However, max_radius_for_radii is non-smooth (min).
        # We can use a smooth approximation or just try to equalize the active constraints.
        
        # Alternative: Binary search on r_c.
        # For a given r_c, check if there exist R_j such that:
        # R_1 >= 2 r_c
        # R_{j+1} - R_j >= 2 r_c
        # R_j sin(pi/k_j) >= r_c
        # R_M + r_c <= 1
        
        # This is a system of linear inequalities!
        # R_1 - 2 r_c >= 0
        # R_{j+1} - R_j - 2 r_c >= 0
        # R_j sin(pi/k_j) - r_c >= 0
        # -R_M - r_c >= -1
        
        # We can solve this feasibility problem using linear programming or simple iteration.
        # Since the constraints are monotonic, we can find the minimal R_j for a given r_c.
        
        def min_radii_for_r(r_c):
            # Returns minimal radii R_j that satisfy constraints for given r_c
            # R_1 >= max(2 r_c, r_c / sin(pi/k_1))
            # R_j >= max(R_{j-1} + 2 r_c, r_c / sin(pi/k_j))
            # Check if R_M + r_c <= 1
            
            R = np.zeros(len(k_vals))
            # Ring 1
            R[0] = max(2.0 * r_c, r_c / np.sin(np.pi / k_vals[0]))
            
            for j in range(1, len(k_vals)):
                R[j] = max(R[j-1] + 2.0 * r_c, r_c / np.sin(np.pi / k_vals[j]))
            
            # Check boundary
            if R[-1] + r_c > 1.0:
                return None
            return R

        # Binary search for max r_c
        lo = 0.0
        hi = 1.0 / (2 * np.sqrt(N/np.pi)) # Rough upper bound
        # Actually hi can be 0.5 (1 circle) but for 300 it's small.
        # Let's use 0.1 as a safe high bound
        hi = 0.1
        
        # Refine hi
        # Area bound: N * pi * r^2 <= pi * 1 * 0.9 (packing density)
        r_area = np.sqrt(0.9 / N)
        hi = min(hi, r_area * 1.1)
        
        for _ in range(30):
            mid = (lo + hi) / 2.0
            R_sol = min_radii_for_r(mid)
            if R_sol is not None:
                lo = mid
            else:
                hi = mid
                
        r_c_est = lo
        R_sol = min_radii_for_r(r_c_est)
        
        if R_sol is None:
            continue
            
        # Now we have optimal radii R_j and radius r_c for this ring structure.
        # We can construct the points.
        # Angular offsets: To maximize gaps, we should stagger rings.
        # For ring j with k_j circles, angle step = 2*pi/k_j.
        # Offset angle for ring j can be chosen to avoid alignment with neighbors.
        # A simple heuristic: offset_j = j * pi / k_j or similar.
        # Actually, for optimal packing, we might want to solve for angles too.
        # But let's start with a static heuristic and then refine.
        
        pts = [(0.0, 0.0)] # Center
        for j, R_j in enumerate(R_sol):
            k_j = k_vals[j]
            if k_j == 0:
                continue
            angles = np.linspace(0, 2*np.pi, k_j, endpoint=False)
            # Stagger: shift by half step for alternating rings?
            # Or just use 0.
            # Let's try offset = j * (2*np.pi / k_j) * 0.5 ? No, that's not consistent.
            # Let's just use 0 for now, or a small random offset to break symmetry?
            # Actually, symmetry is good for stability.
            # Let's try offset = 0.
            angles += 0.0
            
            x = R_j * np.cos(angles)
            y = R_j * np.sin(angles)
            pts.extend(zip(x, y))
            
        pts = np.array(pts)
        
        # Verify count
        if len(pts) != N:
            continue
            
        r_val = compute_radius(pts)
        
        # Refine with local search
        # We can use the ring structure as a starting point for a general local search.
        # But to keep it fast, we might just do a few steps of gradient ascent on the min-distance.
        
        # Let's try a simple local optimization:
        # Move points to increase the minimum distance.
        # This is similar to the seed but starting from a better structured init.
        
        if r_val > best_radius:
            best_radius = r_val
            best_pts = pts.copy()
            
        # Local refinement
        # Use a simple repulsion force model
        # F_i = sum_{j!=i} (c_i - c_j) / |c_i - c_j|^3 + (c_i / |c_i|^3) * lambda_boundary
        # We want to maximize min dist.
        
        # Let's do a few iterations of gradient ascent on the log-sum-exp of distances?
        # Or just simple repulsion.
        
        # Time check
        if time.time() > deadline - 50:
            continue
            
        # Simple local search
        for iter in range(50):
            if time.time() > deadline - 10:
                break
                
            # Compute forces
            # Pairwise repulsion
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, 1e-10)
            
            # Force from pairs
            # F_ij = (c_i - c_j) / dist^3
            # Only consider close pairs to save time?
            # For 300 points, 300^2 = 90k pairs. 50 iters = 4.5M ops. Fast.
            
            forces = np.zeros_like(pts)
            # dist^3
            dist3 = dist**3
            # Avoid division by zero
            dist3[dist3 < 1e-10] = 1e-10
            
            # Sum of (c_i - c_j) / dist^3
            # This is O(N^2)
            # We can vectorize
            # forces[i] = sum_j (pts[i] - pts[j]) / dist[i,j]^3
            # = sum_j pts[i]/dist^3 - sum_j pts[j]/dist^3
            # = pts[i] * sum_j 1/dist^3 - sum_j pts[j]/dist^3
            
            inv_dist3 = 1.0 / dist3
            np.fill_diagonal(inv_dist3, 0.0)
            
            sum_inv = np.sum(inv_dist3, axis=1)
            sum_pts_inv = np.sum(pts * inv_dist3[:, np.newaxis], axis=1)
            
            forces = pts * sum_inv[:, np.newaxis] - sum_pts_inv
            
            # Boundary force
            # Push back if r > 1 - r_c
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            # Soft boundary: if r_pts > 1 - r_c, push inward
            # Let's use a hard constraint projection or a strong force
            # Force = -c_i / |c_i|^3 * strength if outside
            # Or just project
            # Let's project to circle of radius 1 - r_c
            max_r = 1.0 - r_val
            mask = r_pts > max_r
            if np.any(mask):
                # Project
                pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * max_r
                # Zero out force for these? Or keep?
                # Let's just project and continue
            
            # Update
            # Learning rate
            lr = 0.01 * (1.0 - iter/50.0)
            pts += lr * forces
            
            # Project again
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            mask = r_pts > max_r
            if np.any(mask):
                pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * max_r
            
            new_r = compute_radius(pts)
            if new_r > r_val:
                r_val = new_r
            else:
                # If it got worse, revert or reduce lr?
                # Let's just keep the best
                pass
        
        if r_val > best_radius:
            best_radius = r_val
            best_pts = pts.copy()
            
    # Final validation
    if best_pts is None:
        # Fallback to random
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_final[mask, np.newaxis] * 0.9999

    return best_pts
