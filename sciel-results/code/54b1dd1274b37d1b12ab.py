"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Stage 1: Robust Initialization ---
    # Use a golden angle spiral initialization. This provides a reasonably uniform
    # distribution of points that avoids the symmetry traps of ring-based layouts
    # and the clustering of random layouts.
    
    i = np.arange(N)
    # Golden angle
    phi = np.pi * (3.0 - np.sqrt(5.0))
    # Scale radius to fit inside unit circle with margin for circle radius
    # We don't know the final r yet, so we pack them into a slightly smaller disk
    # to allow room for expansion.
    r_max = 1.0 - 0.05 
    r_spiral = np.sqrt(i / N) * r_max
    theta = i * phi
    
    points = np.column_stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)])
    
    # Initial radius estimate
    # Area bound: r <= 1/sqrt(N) ~ 0.0577
    # Best known ~ 0.0531
    # Start with a conservative estimate to ensure initial validity
    r_current = 0.050
    
    # --- Stage 2: Simulated Annealing Optimization ---
    # Objective: Maximize r such that all circles fit.
    # Equivalent to minimizing the "energy" E = sum(exp(-(d_ij - 2r)/T)) + boundary_violations
    # We will use a soft-minimum approximation for the minimum distance constraint.
    # Let E = sum_{i<j} exp(-(d_ij - 2r)/sigma) + sum_i exp((|p_i| + r - 1)/sigma)
    # As sigma -> 0, this penalizes violations heavily.
    # We will anneal sigma and r simultaneously? 
    # Better approach: Fix r and try to minimize violations. If violations ~ 0, increase r.
    # Or: Optimize points to minimize a combined objective:
    # F = -alpha * r + beta * sum(exp(-(d_ij - 2r)/sigma))
    # This is tricky.
    
    # Standard SA approach for packing:
    # 1. Start with valid configuration for r_start.
    # 2. Move a random point.
    # 3. Calculate change in "minimum distance" or "sum of overlaps".
    # 4. Accept if better, or with probability exp(-deltaE/T).
    
    # For N=300, calculating full min-distance is O(N^2). 
    # We can optimize by only checking neighbors, but for simplicity and robustness 
    # in Python/NumPy, we will use a vectorized energy calculation.
    
    # Energy function:
    # We want to maximize the minimum clearance.
    # Let's define a "smooth minimum" of distances:
    # D_smooth = -sigma * log(sum(exp(-d_ij/sigma)))
    # This is the log-sum-exp trick for min.
    # We want to maximize D_smooth and boundary clearance.
    # Objective to minimize: -D_smooth - B_smooth + penalty if invalid?
    
    # Actually, a simpler and very effective heuristic for circle packing is:
    # Minimize E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |p_i| + r - 1)^2
    # This is the same as the previous program's penalty, but SA handles the non-convexity better.
    
    # Let's use the "soft repulsion" potential which is smooth everywhere:
    # E = sum_{i<j} exp(-(d_ij - 2r)/sigma) + sum_i exp((|p_i| + r - 1)/sigma)
    # Minimizing this pushes points apart and inward.
    
    # Strategy:
    # 1. Start with r = 0.050.
    # 2. Run SA to minimize E for fixed r.
    # 3. Check if E is small enough (valid).
    # 4. Increase r slightly.
    # 5. Repeat.
    
    best_points = points.copy()
    best_r = 0.050
    best_score = 0.0
    
    # Update best score
    def calc_score(pts, r):
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.max(norms)
        return min(min_pair, min_bound)
    
    best_score = calc_score(best_points, best_r)
    
    # SA Parameters
    T = 0.005  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    sigma = 0.005  # Width of the exponential potential
    
    # Pre-compute indices for efficiency? 
    # cdist is fast enough for 300x300 in numpy.
    
    def get_energy(pts, r):
        # Pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # We only care about pairs where d < 2r + some margin
        # But exp(-(d-2r)/sigma) decays fast. 
        # To save time, we can mask.
        # However, for N=300, full cdist is ~1ms. 
        # Let's just compute the sum.
        
        # Avoid underflow/overflow
        # If d > 2r, term is exp(negative) -> small.
        # If d < 2r, term is exp(positive) -> large.
        
        # Clip arguments to avoid numerical issues
        args_pair = -(dists - 2*r) / sigma
        # Mask diagonal
        args_pair[np.eye(N, dtype=bool)] = -np.inf
        
        # Sum of exp
        # To prevent overflow, subtract max?
        # max_arg = np.max(args_pair)
        # E_pair = np.sum(np.exp(args_pair - max_arg)) * np.exp(max_arg)
        # But we just need the value for comparison, not the exact scale necessarily.
        # However, SA needs consistent energy differences.
        
        # Let's use a stable log-sum-exp for the sum? No, we just need the sum.
        # If values are huge, we might overflow.
        # Let's clip args_pair to [-50, 50] to keep exp bounded.
        args_pair = np.clip(args_pair, -50, 50)
        E_pair = np.sum(np.exp(args_pair))
        
        # Boundary
        norms = np.linalg.norm(pts, axis=1)
        args_bound = (norms + r - 1.0) / sigma
        args_bound = np.clip(args_bound, -50, 50)
        E_bound = np.sum(np.exp(args_bound))
        
        return E_pair + E_bound

    # Initial energy
    E_curr = get_energy(best_points, best_r)
    
    # SA Loop
    # We will iterate, moving one point at a time.
    # Number of iterations: ~ 50,000 to 100,000 depending on time.
    
    # Time check inside loop
    iter_count = 0
    while time.time() < deadline - 10.0:
        # Pick a random point
        idx = np.random.randint(N)
        orig_pt = best_points[idx].copy()
        
        # Propose a move: random displacement
        # Step size should be related to r
        step_size = 0.01 * r_current
        delta = np.random.normal(0, step_size, 2)
        new_pt = orig_pt + delta
        
        # Ensure new point is not wildly outside
        if np.linalg.norm(new_pt) > 1.5:
            continue
            
        best_points[idx] = new_pt
        
        # Calculate new energy
        E_new = get_energy(best_points, best_r)
        delta_E = E_new - E_curr
        
        # Acceptance criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            E_curr = E_new
            # Accepted
        else:
            # Rejected
            best_points[idx] = orig_pt
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If stuck at T_min, we might want to restart with higher T or increase r
            # For simplicity, just keep going. The moves will become small.
            
        # Check if we can increase r
        # Every few iterations, try to see if current config is valid for a slightly larger r
        # Or simply: if E_curr is very low, increase r.
        
        iter_count += 1
        if iter_count % 5000 == 0:
            # Try to increase r
            # Check current validity
            score_now = calc_score(best_points, best_r)
            if score_now > best_score:
                best_score = score_now
                # Update best_r to the actual achieved radius
                # This is a bit expensive (cdist), but we do it every 5000 iters.
                # Actually, calc_score does cdist.
                # Let's update best_r to score_now if it's valid?
                # The score IS the radius.
                # So if score_now > best_r, we can set best_r = score_now?
                # No, best_r is the target for the energy calculation.
                # If the current points support a radius larger than best_r, 
                # we should increase best_r to push them apart further.
                
                if score_now > best_r * 1.01: # 1% improvement
                    best_r = score_now
                    # Reset temperature to explore the new landscape
                    T = 0.005
                    # Re-calculate energy with new r
                    E_curr = get_energy(best_points, best_r)
                    
    # Final polish: 
    # The SA might have left points in a slightly suboptimal position for the current best_r.
    # We can do a few rounds of "relaxation" where we only accept improving moves (greedy).
    
    T = 1e-9
    for _ in range(5000):
        if time.time() > deadline - 5.0:
            break
        idx = np.random.randint(N)
        orig_pt = best_points[idx].copy()
        step_size = 0.005 * best_r
        delta = np.random.normal(0, step_size, 2)
        new_pt = orig_pt + delta
        
        if np.linalg.norm(new_pt) > 1.5:
            continue
            
        best_points[idx] = new_pt
        E_new = get_energy(best_points, best_r)
        delta_E = E_new - E_curr
        
        if delta_E < 0:
            E_curr = E_new
        else:
            best_points[idx] = orig_pt

    # Final Score Calculation
    final_score = calc_score(best_points, best_r)
    
    # Ensure validity: if any point is outside, project it?
    # The score calculation handles it by taking min(1-|p|), which becomes negative if outside.
    # But the task says "A centre on or outside... makes the candidate invalid".
    # So we MUST ensure all points are strictly inside (or on boundary? "inside" usually means <=).
    # "A centre on or outside ... makes the candidate invalid".
    # So we need |p_i| < 1.
    # Let's project any points that are too close to the edge back inward.
    
    norms = np.linalg.norm(best_points, axis=1)
    # If any norm >= 1, scale it down.
    # We need a safety margin.
    # Let's ensure max norm <= 1 - 1e-6.
    mask = norms >= (1.0 - 1e-6)
    if np.any(mask):
        # Scale these points to be at 1 - 2e-6
        target_norm = 1.0 - 2e-6
        scale = target_norm / norms[mask]
        best_points[mask] *= scale[:, np.newaxis]
        
    # Re-calculate score after projection
    final_score = calc_score(best_points, best_r)
    
    return best_points

# EVOLVE-BLOCK-END
