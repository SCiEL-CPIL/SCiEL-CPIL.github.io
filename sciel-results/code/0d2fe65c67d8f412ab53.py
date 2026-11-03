import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing to find a near-optimal packing of N circles in a unit circle.
    """
    
    def energy(points):
        """
        Calculate an energy value to minimize.
        We want to maximize the minimum distance between any two circles and the distance to the boundary.
        Using a sum of inverse squares encourages spreading points out.
        E = sum(1/d_ij^2) + sum(1/(1-|c_i|)^2)
        """
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        dists_sq[range(N), range(N)] = np.inf
        
        # Inverse squared distances
        # Clip to avoid huge values if points coincide
        dists_sq = np.maximum(dists_sq, 1e-6)
        e_pairs = np.sum(1.0 / dists_sq)
        
        # Boundary distances
        radii_sq = np.sum(points * points, axis=1)
        # Distance to boundary is 1 - |c_i|. Squared is (1 - sqrt(r^2))^2.
        # It is easier to penalize if |c_i| > 1 - r_min.
        # Let's use the distance to the boundary directly.
        radii = np.sqrt(radii_sq)
        dists_boundary = 1.0 - radii
        # If a point is outside, dists_boundary is negative. We want to penalize this heavily.
        # If inside, we want to penalize small distances to boundary.
        # Use (1/r^2) where r is distance to boundary.
        # To handle negative distances (outside), we can square the distance and add a penalty.
        # Simpler: Just use 1 / max(dists_boundary, 1e-6)^2. But if outside, this becomes small? No.
        # If outside, dists_boundary < 0. We want high energy.
        # Let's define boundary penalty as:
        # if dist > 0: 1/dist^2
        # if dist <= 0: 1e6 (huge penalty)
        
        # Vectorized version:
        # We want to minimize energy.
        # Let's use a robust potential.
        # U = sum(1/d_ij^2) + sum(1/max(1-|c_i|, 1e-6)^2)
        # If 1-|c_i| is negative, max makes it 1e-6, so 1/1e-12 = 1e12. Huge penalty. Good.
        
        dists_boundary_safe = np.maximum(dists_boundary, 1e-6)
        e_boundary = np.sum(1.0 / (dists_boundary_safe * dists_boundary_safe))
        
        return e_pairs + e_boundary

    def score(points):
        """
        Calculate the actual score: max common radius.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_pairwise / 2.0, min_boundary)

    # Initialization: Random points within unit circle
    # Use a uniform distribution in the square and reject outside circle?
    # Or just generate random angles and radii.
    # Uniform in circle: r = sqrt(random), theta = random * 2*pi
    r_init = np.sqrt(np.random.rand(N))
    theta_init = np.random.rand(N) * 2 * np.pi
    points = np.column_stack((r_init * np.cos(theta_init), r_init * np.sin(theta_init)))
    
    current_energy = energy(points)
    best_energy = current_energy
    best_points = points.copy()
    best_score = score(best_points)
    
    # SA Parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    # We will run until time limit
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        # Generate a new candidate
        # Perturb a random point
        idx = np.random.randint(N)
        delta = np.random.normal(0, 0.1, 2) # Step size could be adaptive
        
        new_points = points.copy()
        new_points[idx] += delta
        
        # Keep points inside the circle? 
        # If we let them go outside, the energy penalty will push them back.
        # But if they go far, it might take too long to come back.
        # Let's project them back to the circle if they are outside, but with a penalty?
        # Actually, the energy function handles outside points with huge penalty.
        # However, for efficiency, it's better to keep them roughly inside.
        # Let's just let the energy function do the work.
        
        new_energy = energy(new_points)
        
        # Acceptance criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            points = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
                # Update best score only occasionally to save time? 
                # Score calculation is O(N^2). Energy is O(N^2).
                # It's fine to update score every time we find a new best energy.
                # But energy and score are not perfectly correlated (energy is sum, score is min).
                # A lower energy usually implies a higher score, but not strictly.
                # We should track the best *score* separately.
                # However, calculating score is expensive.
                # Let's calculate score every 1000 steps or when best_energy updates.
                s = score(best_points)
                if s > best_score:
                    best_score = s
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min # Or restart? Let's just stay at min for now, or reheat?
            # Reheating can help escape local minima.
            # If we are stuck, we could reheat.
            # For simplicity, we'll just continue with low T.
            # Actually, if T is very low, we only accept improvements.
            # This becomes a local search.
            # To maintain global search capability, we should perhaps reset T if no improvement for a long time?
            # Let's add a simple restart mechanism if stuck.
            
    # Final check
    # Ensure the returned points are valid (inside circle)
    # The SA might leave points slightly outside if T was too low or energy penalty wasn't enough?
    # The energy penalty is huge (1e12), so it's very unlikely.
    # But to be safe, project points inside.
    radii = np.linalg.norm(best_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / radii[mask][:, np.newaxis] * 0.999
        
    return best_points

# EVOLVE-BLOCK-END
