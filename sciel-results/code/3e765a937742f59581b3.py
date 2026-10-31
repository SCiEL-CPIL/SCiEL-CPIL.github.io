"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Pre-compute indices for pairwise distances
    iu = np.triu_indices(N, 1)
    
    def get_distances(pts):
        """Compute all pairwise distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists[iu]
        
    def score(pts):
        """Calculate the squared ratio of max to min pairwise distance."""
        d = get_distances(pts)
        if len(d) == 0:
            return 1e10
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        """Center and scale points so that max pairwise distance is 1."""
        pts = pts - np.mean(pts, axis=0)
        d = get_distances(pts)
        max_d = d.max()
        if max_d > 1e-12:
            pts = pts / max_d
        return pts

    def is_feasible(pts, R_min, R_max=1.0, eps=1e-9):
        """
        Check if all pairwise distances are within [R_min, R_max].
        R_max is fixed to 1.0 because we normalize points to have max dist 1.
        """
        d = get_distances(pts)
        if np.any(d < R_min - eps) or np.any(d > R_max + eps):
            return False
        return True

    def local_search(pts, R_min, max_iters=5000, step_scale=0.05, rng=None):
        """
        Randomized local search to find a feasible configuration for given R_min.
        We start from pts and try to move points to satisfy min_dist >= R_min.
        Strategy: 
        1. Identify violating pairs (d < R_min).
        2. Move points apart.
        3. Also ensure max_dist <= 1 (by normalizing or penalizing).
        
        Since we normalize after every move, we only need to worry about min_dist >= R_min.
        The max_dist will be 1 by construction (unless all points collapse, which we prevent).
        """
        if rng is None:
            rng = np.random.default_rng()
            
        current_pts = pts.copy()
        current_pts = normalize(current_pts)
        
        # If already feasible, return
        if is_feasible(current_pts, R_min):
            return current_pts, True
            
        best_pts = current_pts.copy()
        best_violation = get_distances(current_pts).min() - R_min
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Check current violation
            d = get_distances(current_pts)
            min_d = d.min()
            
            if min_d >= R_min - 1e-9:
                # Feasible found
                return current_pts, True
                
            # Find the pair with minimum distance
            min_idx = np.argmin(d)
            # Convert flat index in upper triangle to (i, j)
            # This is tricky to do fast without precomputed mapping.
            # For N=16, we can just find min in the full matrix for simplicity in LS, 
            # or use the iu indices.
            
            # Let's use full matrix for LS simplicity as N is small
            full_dists = np.sqrt(np.sum((current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :])**2, axis=2))
            np.fill_diagonal(full_dists, np.inf)
            
            i, j = np.unravel_index(np.argmin(full_dists), full_dists.shape)
            
            # Move i and j apart
            # Direction: from j to i for i, from i to j for j
            diff = current_pts[i] - current_pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                # Random kick
                direction = rng.normal(size=2)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
                
            # Step size: proportional to how much we need to move
            # We need to increase dist by at least (R_min - dist)
            # Moving each point by step/2 increases dist by step (roughly)
            needed = R_min - dist
            step = max(0.01, needed * 0.5) # Conservative step
            
            # Add some noise to escape local minima
            noise = rng.normal(0, step_scale * 0.1, size=(2, 2))
            
            move_i = direction * step + noise[0]
            move_j = -direction * step + noise[1]
            
            new_pts = current_pts.copy()
            new_pts[i] += move_i
            new_pts[j] += move_j
            
            # Normalize
            new_pts = normalize(new_pts)
            
            # Check if this move improved the minimum distance
            new_d = get_distances(new_pts)
            new_min_d = new_d.min()
            
            # Accept if it improves the min distance (hill climbing on min dist)
            # Or if it's feasible
            if new_min_d > min_d - 1e-9: # Accept if not significantly worse
                current_pts = new_pts
                if new_min_d > best_violation:
                    best_violation = new_min_d
                    best_pts = new_pts.copy()
                    
        return best_pts, False

    def binary_search_max_R(initial_pts, rng):
        """
        Binary search for the maximum R such that a feasible configuration exists.
        Feasible means: min_dist >= R and max_dist <= 1 (normalized).
        """
        R_low = 0.0
        R_high = 1.0
        
        # Initial check: is R_high feasible?
        pts_test, feasible = local_search(initial_pts, R_high, max_iters=100, rng=rng)
        if feasible:
            return R_high, pts_test
            
        # Binary search
        best_R = 0.0
        best_pts = initial_pts.copy()
        
        for _ in range(40): # ~2^-40 precision
            if time.time() > deadline - 10:
                break
                
            R_mid = (R_low + R_high) / 2
            
            # Try to find a feasible configuration for R_mid
            # We use the best found so far as a starting point, or random restarts
            # Strategy: Try from best_pts, then random
            found = False
            candidate_pts = None
            
            # Try 3 times: once from best, twice from random
            for attempt in range(3):
                if attempt == 0:
                    start_pts = best_pts.copy()
                else:
                    start_pts = rng.normal(0, 1, (N, 2))
                    
                candidate_pts, found = local_search(start_pts, R_mid, max_iters=200, rng=rng)
                
                if found:
                    break
                    
            if found:
                R_low = R_mid
                best_R = R_mid
                best_pts = candidate_pts.copy()
            else:
                R_high = R_mid
                
        return best_R, best_pts

    # Main Logic
    rng = np.random.default_rng(42)
    
    # Initial Guess: Hexagonal lattice
    pts = np.zeros((N, 2))
    side = 4
    count = 0
    for row in range(side):
        for col in range(side):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts[count, 0] = x
            pts[count, 1] = y
            count += 1
            if count == N:
                break
        if count == N:
            break
    pts = normalize(pts)
    
    best_R = 0.0
    best_pts = pts.copy()
    best_score = score(pts)
    
    # Perform binary search for max R
    # We will run multiple "rounds" of binary search with different random seeds/restarts
    # to avoid getting stuck in local minima of the feasibility landscape
    
    num_rounds = 5
    
    for r in range(num_rounds):
        if time.time() > deadline - 20:
            break
            
        # Use a different starting point for each round
        if r == 0:
            start_pts = pts.copy()
        else:
            start_pts = rng.normal(0, 1, (N, 2))
            start_pts = normalize(start_pts)
            
        # Run binary search
        R_found, pts_found = binary_search_max_R(start_pts, rng)
        
        if R_found > best_R:
            best_R = R_found
            best_pts = pts_found.copy()
            best_score = (1.0 / best_R) ** 2 if best_R > 0 else 1e10
            
            # Refine the best found with a high-precision local search
            # Try to push R a bit higher from this solution
            # We can do a few more steps of binary search around best_R
            # But for now, just record it.
            
    # Final Polish: Try to improve the best solution found
    if best_R > 0:
        # Use the best solution as a start for a final, intensive local search
        # aiming for a slightly higher R
        R_target = best_R * 1.01 # Try 1% higher
        refined_pts, refined = local_search(best_pts, R_target, max_iters=5000, rng=rng)
        if refined:
            best_R = R_target
            best_pts = refined_pts
            best_score = (1.0 / best_R) ** 2
            
    # Return the best points found
    # Ensure it's a valid array
    return best_pts

# EVOLVE-BLOCK-END
