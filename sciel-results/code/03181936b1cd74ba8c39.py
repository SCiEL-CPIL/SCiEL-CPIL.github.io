"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, aim to finish by 110s
    deadline = start_time + 110.0 

    # --- Stage 1: Structured Ring-Based Initialization ---
    # The "sunflower" spiral is good, but rings allow for better local density control.
    # We initialize with a hexagonal-like packing in concentric rings.
    # 
    # Strategy:
    # 1. Estimate number of rings.
    # 2. For each ring, place points evenly spaced.
    # 3. Adjust radii to maximize local spacing.
    
    # A good heuristic for ring packing:
    # Radius of ring k is roughly proportional to sqrt(k).
    # Number of points in ring k is proportional to k.
    
    # Let's use a deterministic initialization based on "Vogel's model" (Fermat spiral)
    # but then refine the radial positions to be more uniform in a "ring" sense if needed.
    # Actually, the prompt suggests a "different design or modelling family".
    # The current program uses L-BFGS-B on a soft-min.
    # The new strategy is Simulated Annealing on a hard-constraint score.
    
    # Initialization: Golden Angle Spiral (Sunflower)
    # This is a standard, robust initialization for circle packing.
    i = np.arange(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    # Scale to fit in unit circle with a safety margin
    # r_max = 1 - r_est. Let's estimate r_est ~ 0.05
    # r_i = sqrt(i/N) * (1 - 0.05)
    r_spiral = np.sqrt(i / N) * 0.95
    theta = i * golden_angle
    x = np.column_stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)])
    
    # --- Stage 2: Simulated Annealing ---
    # Objective: Maximize r_min = min( min_pair_dist/2, min_boundary_dist )
    # Since we can't easily maximize r_min directly with SA (it's not smooth),
    # we use a penalty-based score or simply track the best r_min found.
    # 
    # Move Strategy:
    # 1. Pick a random subset of points (or all points) to move.
    # 2. Apply a small random displacement (Gaussian noise).
    # 3. Calculate the new r_min.
    # 4. Accept if new r_min > old r_min (improvement).
    # 5. Otherwise, accept with probability exp((new_score - old_score) / T).
    #    Here, score is r_min. But r_min is discrete-ish.
    #    Better: Use a smooth score for SA, then refine?
    #    No, the directive says "hard-constrained optimization ... penalty-free barrier ... sum of logarithmic distances".
    #    
    # Let's implement the "Barrier" objective mentioned in the directive:
    # Objective to maximize: S = sum( log(dist_ij) ) + sum( log(1 - |p_i|) )
    # This function is smooth and encourages all distances to be large,
    # effectively pushing the system towards a uniform packing where the minimum is maximized.
    # It is "penalty-free" in the sense that it doesn't use max(0, violation),
    # but rather a potential that goes to -infinity as distance -> 0.
    
    def compute_barrier_score(x):
        """
        Computes the sum of log distances to nearest neighbors and boundary.
        Actually, summing log of ALL pairwise distances is O(N^2) but very stable.
        S = 0.5 * sum_{i != j} log(||p_i - p_j||) + sum_i log(1 - ||p_i||)
        We must ensure distances are > 0 and norms < 1.
        If a point is on the boundary or overlapping, log -> -inf.
        We clip values to avoid -inf during computation.
        """
        p = x.reshape(N, 2)
        
        # Boundary term
        norms = np.linalg.norm(p, axis=1)
        # Clip norms to be slightly less than 1 to avoid log(0)
        # If norm >= 1, the configuration is invalid. We want the score to drop sharply.
        # Use a small epsilon.
        eps = 1e-9
        valid_norms = np.clip(norms, 0.0, 1.0 - eps)
        # If any norm > 1, we should penalize heavily. 
        # The log barrier naturally does this if we define the domain correctly,
        # but here we are outside the domain. 
        # Let's use: log(1 - norm) if norm < 1, else -large_penalty.
        boundary_term = np.zeros(N)
        mask_in = norms < 1.0
        boundary_term[mask_in] = np.log(1.0 - norms[mask_in])
        boundary_term[~mask_in] = -100.0  # Heavy penalty for being outside
        
        # Pairwise term
        # Calculate pairwise distances
        # To save memory, we can do this in chunks or use the full matrix for N=300.
        # N=300 -> 300x300 matrix is 90k floats. Very small.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Mask diagonal (distance 0)
        np.fill_diagonal(dists, 1.0) # Set to 1 so log(1)=0, doesn't affect sum
        
        # Clip distances to eps to avoid log(0) for overlaps
        dists_clipped = np.clip(dists, eps, None)
        
        # Sum of logs. 
        # We only need the sum, not the matrix.
        # sum_{i,j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
        # We can just sum the whole matrix and divide by 2 (diagonal is 0 contribution).
        pair_term = 0.5 * np.sum(np.log(dists_clipped))
        
        return pair_term + np.sum(boundary_term)

    # We want to MAXIMIZE this barrier score.
    # However, the barrier score is dominated by the closest pairs.
    # Maximizing the sum of logs is equivalent to maximizing the geometric mean of distances.
    # This is a known heuristic for packing.
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature (scale of score difference)
    T_min = 1e-6
    cooling_rate = 0.995  # Cool slowly
    step_size = 0.05      # Initial step size for random moves
    
    current_x = x.copy()
    current_score = compute_barrier_score(current_x)
    
    best_x = current_x.copy()
    best_score = current_score
    
    # We need to track the ACTUAL packing radius (min clearance) for the final answer,
    # because the barrier score is a proxy.
    
    def get_actual_radius(x):
        p = x.reshape(N, 2)
        norms = np.linalg.norm(p, axis=1)
        min_bound = 1.0 - np.max(norms)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair = 0.5 * np.min(dists)
        
        return min(min_bound, min_pair)
    
    best_actual_r = get_actual_radius(best_x)
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 5000  # Restart or stop if stuck?
    
    while time.time() < deadline:
        # Check time every 100 iterations to avoid overhead
        if iteration % 100 == 0 and time.time() > deadline - 5:
            break
            
        # Generate a random move
        # Pick a random point (or a few) to move
        # Moving one point at a time is slow for N=300 but robust.
        # Moving a few points is faster. Let's pick 1 point for simplicity and robustness of SA.
        idx = np.random.randint(0, N)
        
        # Random displacement
        # Step size should decrease or stay constant? 
        # In SA, step size is often tied to T or kept constant.
        # Let's scale step size with T to refine the search as it cools.
        move_mag = step_size * T
        delta = np.random.normal(0, move_mag, 2)
        
        new_x = current_x.copy()
        new_x[idx] += delta
        
        # Ensure new_x is within bounds? 
        # The barrier score will handle it (penalty if outside).
        # But if it goes way outside, the score drops massively.
        
        new_score = compute_barrier_score(new_x)
        
        # Acceptance Criterion
        if new_score > current_score:
            # Always accept improvement
            current_x = new_x
            current_score = new_score
            no_improve_count = 0
        else:
            # Accept with probability exp((new - current) / T)
            # If new_score is -inf (invalid), prob is 0.
            if new_score == -np.inf or current_score == -np.inf:
                accept = False
            else:
                delta_e = new_score - current_score
                # Prevent overflow in exp
                if delta_e / T < -50:
                    accept = False
                else:
                    accept = np.random.rand() < np.exp(delta_e / T)
            
            if accept:
                current_x = new_x
                current_score = new_score
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Update Best
        # Checking actual radius every iteration is expensive (O(N^2)).
        # Check every 10 iterations.
        if iteration % 10 == 0:
            actual_r = get_actual_radius(current_x)
            if actual_r > best_actual_r:
                best_actual_r = actual_r
                best_x = current_x.copy()
                no_improve_count = 0 # Reset stagnation counter on actual improvement
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If T is too low, we might get stuck. 
            # We could restart with higher T if stagnation is high.
            if no_improve_count > 2000:
                # Reheat
                T = 0.1
                no_improve_count = 0
                # Jitter the best solution
                best_x = best_x + np.random.normal(0, 0.01, (N, 2))
                current_x = best_x.copy()
                current_score = compute_barrier_score(current_x)
        
        iteration += 1
        
    # Final refinement:
    # The SA might leave points slightly overlapping or not perfectly optimized.
    # We can run a few steps of gradient ascent on the barrier score or 
    # simply return the best found.
    # Given the time budget, the best found during SA is usually good.
    # However, the "barrier" score maximization tends to equalize distances.
    # Let's do a final local polish using a simple gradient step on the barrier score.
    
    # Simple gradient polish
    # Gradient of sum log(dist) w.r.t p_i:
    # d/dp_i sum_{j != i} log(||p_i - p_j||) = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    # d/dp_i log(1 - ||p_i||) = -p_i / (||p_i|| (1 - ||p_i||))
    
    def barrier_gradient(x):
        p = x.reshape(N, 2)
        grad = np.zeros_like(p)
        
        # Boundary gradient
        norms = np.linalg.norm(p, axis=1)
        # Avoid div by zero
        norms_safe = np.where(norms < 1e-9, 1e-9, norms)
        # Only valid if inside
        mask_in = norms < 1.0
        # grad_bound_i = -p_i / (norm_i * (1 - norm_i))
        grad[mask_in] -= p[mask_in] / (norms_safe[mask_in, np.newaxis] * (1.0 - norms[mask_in, np.newaxis]))
        
        # Pairwise gradient
        # For each i, sum over j != i of (p_i - p_j) / ||p_i - p_j||^2
        # We can vectorize this.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        # Avoid div by 0 on diagonal
        np.fill_diagonal(dists_sq, 1.0)
        
        # Term: (p_i - p_j) / dist_sq
        # This is a matrix of vectors.
        # inv_dists_sq = 1 / dists_sq
        # contrib = diff * inv_dists_sq[:, :, np.newaxis]
        # grad = sum(contrib, axis=1)
        
        # This creates a N x N x 2 array. 300*300*2 = 180k floats. Fine.
        inv_dists_sq = 1.0 / dists_sq
        # Zero out diagonal contribution
        np.fill_diagonal(inv_dists_sq, 0.0)
        
        # contrib[i, j] = (p_i - p_j) / dist_sq_ij
        # We want sum_j contrib[i, j]
        # diff is (p_i - p_j)
        contrib = diff * inv_dists_sq[:, :, np.newaxis]
        grad += np.sum(contrib, axis=1)
        
        return grad.flatten()

    # Run a few iterations of gradient ascent
    lr = 0.01
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        g = barrier_gradient(best_x)
        # Normalize gradient to avoid huge steps
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-6:
            g = g / g_norm
        best_x += lr * g
        # Ensure validity? The gradient might push outside. 
        # We'll check the actual radius and revert if it gets worse.
        r_new = get_actual_radius(best_x)
        if r_new < best_actual_r:
            # Revert
            best_x = best_x - lr * g
            # If still bad, break
            if get_actual_radius(best_x) < best_actual_r - 1e-9:
                break
        else:
            best_actual_r = r_new

    return best_x

# EVOLVE-BLOCK-END
