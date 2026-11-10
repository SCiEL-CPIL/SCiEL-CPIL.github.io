import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        # Use cdist for speed
        dist = cdist(points, points)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # --- Stage 1: Ring-based representation ---
    # We assume the packing consists of K rings.
    # Each ring k has:
    #   - radius r_k
    #   - count n_k
    #   - phase offset phi_k
    # The points in ring k are at angles phi_k + 2*pi*i/n_k for i=0..n_k-1.
    
    # Heuristic for ring counts:
    # The number of circles that fit in a ring of radius r is roughly floor(2*pi*r / (2*r_circle)).
    # Since r_circle is unknown initially, we estimate based on area.
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # Max r ~ sqrt(pi / N) / sqrt(2) approx? No, packing density ~ 0.9069.
    # r ~ sqrt(0.9069 * pi / (N * pi)) = sqrt(0.9069/N) ~ 0.055.
    # Let's guess an initial ring structure.
    
    # Common ring structures for N=300:
    # Center: 1
    # Ring 1: 6
    # Ring 2: 12
    # Ring 3: 18
    # ...
    # Sum of 1 + 6*k.
    # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 + 60 = 331 (too many)
    # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 = 271
    # Need 300. So maybe 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 + 29 (partial last ring)?
    # Or adjust counts.
    
    # Let's define a fixed partition for simplicity in the DE stage, 
    # but allow the radii and phases to vary.
    # A good partition for 300:
    # 1, 6, 12, 18, 24, 30, 36, 42, 48, 45, 38? 
    # Let's just use a heuristic to generate counts that sum to N.
    
    def get_ring_counts(N_total):
        counts = [1]
        current = 1
        k = 1
        while current < N_total:
            # Ideal count for next ring is 6*k
            ideal = 6 * k
            # Don't exceed remaining
            if current + ideal <= N_total:
                counts.append(ideal)
                current += ideal
                k += 1
            else:
                # Last ring takes the rest
                remaining = N_total - current
                if remaining > 0:
                    counts.append(remaining)
                break
        return counts

    counts = get_ring_counts(N)
    K = len(counts)
    
    # Variables for DE:
    # For each ring k:
    #   r_k (radius)
    #   phi_k (phase)
    # Total dims = 2*K
    
    # Bounds:
    # r_k must be such that points are inside.
    # Max radius is 1 - r_circle. r_circle is unknown, but < 0.06.
    # Let's set bounds loosely.
    # r_min for ring k: roughly distance to previous ring.
    # r_max: 1.0
    
    # We will optimize the "gap" or just the radii directly.
    # Let's optimize radii r_k and phases phi_k.
    
    def ring_params_to_points(params):
        # params: [r0, phi0, r1, phi1, ..., rK-1, phiK-1]
        points = []
        for k in range(K):
            r_k = params[2*k]
            phi_k = params[2*k + 1]
            n_k = counts[k]
            if n_k == 1:
                # Center point, phase doesn't matter, but r should be 0?
                # If k==0, r0 should be 0.
                points.append([0.0, 0.0])
            else:
                angles = phi_k + 2 * np.pi * np.arange(n_k) / n_k
                xs = r_k * np.cos(angles)
                ys = r_k * np.sin(angles)
                points.extend(zip(xs, ys))
        
        pts = np.array(points)
        return pts

    def ring_objective(params):
        # Returns a scalar to minimize. We want to maximize min_dist/2.
        # So we minimize -min_dist/2 or a penalty for violations.
        
        # Time check
        if time.time() > deadline - 20:
            return 1e6
        
        pts = ring_params_to_points(params)
        
        # Check validity: all points inside unit circle
        radii = np.linalg.norm(pts, axis=1)
        if np.any(radii > 1.0):
            # Penalty for being outside
            return 1e6 + np.sum(np.maximum(0, radii - 1.0))
        
        # Calculate min pair distance
        # For N=300, cdist is fast enough
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        min_d = np.min(dist)
        
        # We want to maximize min_d / 2
        # Also consider boundary distance
        min_boundary = np.min(1.0 - radii)
        score = min(min_d / 2.0, min_boundary)
        
        # To maximize score, minimize -score
        # But we also need to ensure no overlaps (min_d > 0 is guaranteed by geometry if distinct, 
        # but we want to push them apart).
        # A better objective: minimize (1/score) or -score.
        # Let's use -score.
        
        return -score

    # Bounds for DE
    # r_k: 
    # Ring 0 (center): r=0 fixed? Or optimize? If we optimize r0, it might move off center.
    # Let's fix r0=0 and only optimize phi0 (irrelevant) and r_k, phi_k for k>=1.
    # Actually, let's just let DE find the best radii.
    # Lower bound for r_k: 0.
    # Upper bound for r_k: 1.0.
    
    # However, a better initial guess helps DE.
    # Estimated radius of circles: r_c ~ 0.055
    # Ring k radius ~ (k + 0.5) * r_c * 2 * something?
    # Distance between rings ~ 2 * r_c * sin(pi/n) approx 2*r_c for large n.
    # So r_k ~ k * 2 * r_c.
    
    r_est = 0.055
    bounds = []
    x0 = []
    for k in range(K):
        if k == 0:
            # Center
            bounds.append((0.0, 0.0)) # Fix r0
            bounds.append((0.0, 2*np.pi)) # phi0
            x0.append(0.0)
            x0.append(0.0)
        else:
            # Estimate radius
            r_init = k * 2 * r_est
            bounds.append((0.1, 1.0))
            bounds.append((0.0, 2*np.pi))
            x0.append(r_init)
            x0.append(2 * np.pi * np.random.random()) # Random phase

    # Run Differential Evolution
    # popsize=15, maxiter=100, tol=1e-4
    # Time budget: ~60 seconds for DE
    
    de_start = time.time()
    de_deadline = de_start + 60.0
    
    try:
        # Use seed for reproducibility? No, random is fine.
        result_de = differential_evolution(
            ring_objective, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-4, 
            seed=42,
            init='latinhypercube',
            polish=False, # We will do our own polish
            callback=None
        )
    except Exception as e:
        # Fallback to hex lattice if DE fails
        pass

    if time.time() < de_deadline and 'result_de' in locals() and result_de.success:
        best_params = result_de.x
        best_ring_points = ring_params_to_points(best_params)
        best_ring_score = compute_score(best_ring_points)
    else:
        # Fallback: Generate hex lattice
        spacing = 2.0 * np.sqrt(0.907 / N)
        k_max = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        pts = np.array(pts)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind="stable")[:N]
        best_ring_points = pts[idx].copy()
        best_ring_score = compute_score(best_ring_points)

    # --- Stage 2: Full coordinate polish ---
    # Start from best_ring_points
    # Use L-BFGS-B to maximize the minimum distance.
    # Objective: minimize -min_dist/2 with penalty for boundary violation.
    
    def full_objective(x):
        if time.time() > deadline - 5:
            return 1e6
        pts = x.reshape(-1, 2)
        
        # Boundary penalty
        radii = np.linalg.norm(pts, axis=1)
        boundary_viol = np.sum(np.maximum(0, radii - 1.0)) ** 2 * 100
        
        # Pairwise distance penalty
        # We want to maximize min dist.
        # Approximate by sum of inverse squares of close pairs?
        # Or just use the min directly? Min is non-differentiable.
        # Use a smooth approximation or just minimize the sum of distances for the closest pairs.
        
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        
        # Find the k smallest distances
        # For N=300, 300*299/2 ~ 45k pairs. Sorting is fast.
        flat_dist = dist.flatten()
        # Get indices of 10 smallest
        k_smallest = 10
        part_idx = np.argpartition(flat_dist, k_smallest)[:k_smallest]
        smallest_dists = flat_dist[part_idx]
        
        # Objective: minimize sum of smallest distances? No, we want to increase them.
        # Minimize -sum(smallest_dists) or -min(smallest_dists).
        # Min is better for "max-min" but non-smooth.
        # Let's try minimizing the sum of the 10 smallest distances.
        # This pushes all close pairs apart.
        
        score_term = -np.sum(smallest_dists)
        
        # Also add a term to keep them inside
        # radii > 1.0 is bad.
        # We already have boundary_viol.
        
        return score_term + boundary_viol

    # Initial guess for gradient method
    x0_full = best_ring_points.flatten()
    
    # Run L-BFGS-B
    # Time budget: remaining time
    polish_start = time.time()
    polish_deadline = polish_start + (deadline - polish_start - 5)
    
    try:
        # Bounds for L-BFGS-B: x in [-1, 1], y in [-1, 1]
        bounds_full = [(-1.0, 1.0), (-1.0, 1.0)] * N
        
        # We need to ensure we don't take too long.
        # maxiter=100
        result_full = minimize(
            full_objective, 
            x0_full, 
            method='L-BFGS-B', 
            bounds=bounds_full,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result_full.success:
            new_points = result_full.x.reshape(-1, 2)
            # Project back inside if necessary (L-BFGS-B should handle bounds, but check)
            radii = np.linalg.norm(new_points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                new_points[mask_out] /= radii[mask_out, np.newaxis]
            
            new_score = compute_score(new_points)
            if new_score > best_ring_score:
                best_points = new_points
                best_score = new_score
            else:
                best_points = best_ring_points
                best_score = best_ring_score
        else:
            best_points = best_ring_points
            best_score = best_ring_score
    except Exception:
        best_points = best_ring_points
        best_score = best_ring_score

    # Final check and return
    # Ensure valid shape
    if best_points.shape != (N, 2):
        # Fallback
        spacing = 0.1
        pts = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 0.9:
                    pts.append((x, y))
        pts = np.array(pts[:N])
        return pts

    return best_points

# Test block (optional, for local debugging)
if __name__ == "__main__":
    import time
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    # Calculate score
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_d = np.min(dist)
    radii = np.linalg.norm(pts, axis=1)
    min_b = np.min(1.0 - radii)
    score = min(min_d / 2.0, min_b)
    print(f"Score: {score}")
    print(f"Min pair dist: {min_d}")
    print(f"Min boundary dist: {min_b}")
