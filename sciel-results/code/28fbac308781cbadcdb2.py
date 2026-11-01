"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def compute_score(points):
        """Compute the packing radius for a set of points."""
        if len(points) != N:
            return -1.0
        if not np.all(np.isfinite(points)):
            return -1.0
        r_orig = np.linalg.norm(points, axis=1)
        if np.any(r_orig > 1.0):
            return -1.0
        boundary_margin = 1.0 - r_orig
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        return min(boundary_margin.min(), min_pair_dist / 2.0)

    def soft_min_score(points, beta=50.0):
        """Compute a smooth approximation of the negative packing radius."""
        if len(points) != N:
            return 1.0
        if not np.all(np.isfinite(points)):
            return 1.0
        
        r_orig = np.linalg.norm(points, axis=1)
        boundary_margin = 1.0 - r_orig
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # We want to maximize min(boundary_margin, min_pair_dist / 2)
        # This is equivalent to minimizing max(-boundary_margin, -min_pair_dist/2)
        # Using soft min: -1/beta * log(sum(exp(-beta * d_i))) approximates min(d_i)
        
        d_pairs = dist / 2.0
        d_bound = boundary_margin
        
        # Combine into one vector of "distances"
        # To avoid huge arrays, we can just compute the soft min of the two minima?
        # No, the min is over all pairs AND boundary.
        # Let's compute soft min of all constraints.
        
        # Optimization: Only consider pairs that are "close" to the current minimum?
        # For L-BFGS, we need a smooth function.
        # Let's use a log-sum-exp on the negative distances.
        
        # Vector of all pair distances / 2
        # And vector of all boundary margins
        # We want min( min(d_pairs), min(d_bound) )
        # This is min( min(d_pairs), min(d_bound) )
        
        # Let's just compute the standard min for the score check, but for optimization
        # we use a surrogate.
        
        # Surrogate: -1/beta * log( sum(exp(-beta * d_pairs)) + sum(exp(-beta * d_bound)) )
        # This approximates min(all d's)
        
        # To prevent overflow, subtract max
        # But exp(-beta * d) is small if d is large.
        
        # Let's use a simpler approach:
        # f(x) = -min(d)
        # Approximate min(d) using log-sum-exp of -d?
        # min(d) = - max(-d)
        # max(-d) approx -1/beta log sum exp(beta * (-d)) ? No.
        
        # Let's stick to the known good strategy from Knowledge:
        # "Replaces non-smooth min-objective with smooth soft-min surrogate (log-sum-exp)"
        
        # We minimize: (1/beta) * log( sum_{i<j} exp(-beta * dist_ij/2) + sum_i exp(-beta * (1 - ||c_i||)) )
        # This is a smooth lower bound on the min distance.
        
        # For speed, we can't compute all pairs if N is large, but N=300 is fine (45k pairs).
        
        d_pairs_flat = d_pairs[~np.eye(N, dtype=bool)]
        d_bound_flat = d_bound
        
        all_d = np.concatenate([d_pairs_flat, d_bound_flat])
        
        # Clip to avoid issues
        all_d = np.clip(all_d, 1e-10, 1.0)
        
        # Log-sum-exp trick
        m = np.min(all_d)
        # We want approx min(all_d)
        # min(all_d) = -1/beta * log( sum( exp(-beta * d) ) ) is NOT correct.
        # min(all_d) is approximated by -1/beta * log( sum( exp(-beta * d) ) ) as beta -> inf?
        # If d_min is the smallest, exp(-beta * d_min) is the largest term.
        # sum ~ exp(-beta * d_min).
        # log(sum) ~ -beta * d_min.
        # -1/beta * log(sum) ~ d_min.
        # Yes, this works.
        
        val = -1.0 / beta * np.log(np.sum(np.exp(-beta * all_d)))
        
        return -val # We want to maximize val, so minimize -val

    def generate_ring_initialization():
        """Generate points in concentric rings."""
        points = []
        
        # Center
        points.append([0.0, 0.0])
        remaining = N - 1
        ring_idx = 1
        current_radius = 0.0
        
        while remaining > 0:
            # Estimate how many points fit in the next ring
            # Approximate radius of this ring
            # Area of annulus ~ 2 * pi * r * dr
            # We want to fill area efficiently.
            
            # Heuristic: m_k ~ 6 * k
            m = 6 * ring_idx
            if remaining < m:
                m = remaining
            
            # Calculate radius for this ring
            # Assume previous ring radius was r_prev
            # r_curr ~ r_prev + 2*rho
            # But we don't know rho yet.
            # Let's just place them at equal angular spacing and determine radius later?
            # Or use a fixed spacing.
            
            # Let's use a simple spiral or fixed radius increment.
            # Better: Solve for optimal radii given the counts.
            
            # For now, just accumulate counts and assign later.
            points_count = m
            remaining -= points_count
            ring_idx += 1
            
            # We'll store the counts in a list
            if not hasattr(generate_ring_initialization, 'counts'):
                generate_ring_initialization.counts = []
            generate_ring_initialization.counts.append(points_count)
            
        # Now we have a list of counts per ring (excluding center)
        # Let's refine the counts to be more uniform?
        # Standard hexagonal packing has 1, 6, 12, 18...
        # Let's check if our counts match that.
        # My loop: 6, 12, 18... Yes.
        
        # Now calculate optimal radii for these counts.
        counts = generate_ring_initialization.counts
        del generate_ring_initialization.counts
        
        # Radii r_0=0 (center)
        # r_k for ring k (1-indexed in counts, 0-indexed here)
        # Constraints:
        # 1. r_k + rho <= 1
        # 2. r_{k+1} - r_k >= 2*rho
        # 3. 2 * r_k * sin(pi / m_k) >= 2*rho  => r_k >= rho / sin(pi/m_k)
        
        # We want to maximize rho.
        # Let's solve for rho by binary search.
        
        def is_valid_rho(rho):
            r_prev = 0.0
            for i, m in enumerate(counts):
                # Constraint from tangential spacing
                r_min_tang = rho / np.sin(np.pi / m)
                # Constraint from radial spacing
                r_min_rad = r_prev + 2.0 * rho
                r_curr = max(r_min_tang, r_min_rad)
                
                if r_curr + rho > 1.0:
                    return False
                r_prev = r_curr
            return True

        lo = 0.0
        hi = 0.1 # Max possible rho is 1/sqrt(N) approx, 0.05 is realistic
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if is_valid_rho(mid):
                lo = mid
            else:
                hi = mid
        
        rho = lo
        r_prev = 0.0
        radii = [0.0] # Center
        for i, m in enumerate(counts):
            r_min_tang = rho / np.sin(np.pi / m)
            r_min_rad = r_prev + 2.0 * rho
            r_curr = max(r_min_tang, r_min_rad)
            radii.append(r_curr)
            r_prev = r_curr
        
        # Generate points
        all_points = [[0.0, 0.0]]
        for i, m in enumerate(counts):
            r = radii[i+1]
            angles = np.linspace(0, 2*np.pi, m, endpoint=False)
            # Add a small phase shift to alternate rings to improve packing?
            # For hexagonal, shift by pi/m
            phase = np.pi / m if i % 2 == 1 else 0.0
            angles += phase
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            for j in range(m):
                all_points.append([x[j], y[j]])
        
        return np.array(all_points)

    # 1. Generate initial points
    points = generate_ring_initialization()
    
    # 2. Optimize using L-BFGS-B with soft-min objective
    # Flatten points for optimization
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Ensure points stay inside circle during optimization?
        # L-BFGS-B doesn't handle non-linear constraints well.
        # We will rely on the penalty in the objective or projection.
        # Let's add a penalty for going outside.
        r = np.linalg.norm(pts, axis=1)
        penalty = np.sum(np.maximum(0, r - 1.0)**2)
        
        score = soft_min_score(pts, beta=50.0)
        return score + 10.0 * penalty

    # Run optimization
    # Limit iterations to ensure speed
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if time.time() > deadline - 10.0:
        # If we are running out of time, return current best
        pass
    
    best_points = res.x.reshape(-1, 2)
    
    # 3. Final Polish: A few steps of local optimization with exact min
    # Use Nelder-Mead on the exact score for a few steps
    def exact_neg_score(x):
        pts = x.reshape(-1, 2)
        s = compute_score(pts)
        if s < 0:
            return 1.0
        return -s

    # Run a short Nelder-Mead
    res2 = minimize(exact_neg_score, best_points.flatten(), method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if res2.fun < exact_neg_score(best_points.flatten()):
        best_points = res2.x.reshape(-1, 2)
    
    # Ensure validity
    r = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if r[i] > 1.0:
            best_points[i] *= 0.999 / r[i]
            
    return best_points
# EVOLVE-BLOCK-END
