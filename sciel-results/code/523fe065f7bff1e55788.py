"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Stage 1: Structured Ring Initialization ---
    # This strategy uses a ring-based packing to create a highly structured, 
    # non-lattice starting point. It is denser and more regular than a simple 
    # spiral or random placement, providing a better basin of attraction for 
    # the subsequent gradient-based optimization.
    
    # We aim for a target radius r_target. 
    # Area bound: r <= 1/sqrt(300) ~ 0.0577.
    # Best known ~ 0.0531.
    # Let's initialize for r_init = 0.052 to have a valid start.
    
    r_init = 0.052
    points = np.zeros((N, 2))
    
    # Place one circle at the center
    points[0] = [0.0, 0.0]
    idx = 1
    
    # Place circles in concentric rings
    # For a circle of radius R, the number of circles of radius r that fit 
    # on the circumference is approximately floor(2*pi*R / (2*r)) = floor(pi*R/r)
    # However, we also need to ensure the rings themselves don't overlap.
    # Distance between ring centers should be at least 2*r.
    # Let ring radius R_k = k * 2*r_init.
    # Max k such that R_k + r_init <= 1 => k <= (1 - r_init) / (2*r_init)
    
    max_ring_index = int((1.0 - r_init) / (2.0 * r_init))
    
    for k in range(1, max_ring_index + 1):
        R = k * 2.0 * r_init
        # Check if this ring fits inside the container
        if R + r_init > 1.0:
            break
            
        # Number of circles on this ring
        # The chord length between adjacent centers is 2*R*sin(pi/n)
        # We need 2*R*sin(pi/n) >= 2*r_init => sin(pi/n) >= r_init/R
        # => pi/n >= arcsin(r_init/R) => n <= pi / arcsin(r_init/R)
        # Also n must be integer.
        
        ratio = r_init / R
        if ratio > 1.0:
            n = 1
        else:
            # Calculate max n
            max_n = int(np.floor(np.pi / np.arcsin(ratio)))
            # Ensure at least 1 circle if it fits
            n = max(1, max_n)
            
        # Check if we have enough spots left
        if idx + n > N:
            n = N - idx
            
        # Place n circles on this ring
        if n > 0:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            # Add a small random jitter to break symmetry and help optimization
            angles += np.random.uniform(-0.1, 0.1, n)
            
            x = R * np.cos(angles)
            y = R * np.sin(angles)
            
            points[idx:idx+n, 0] = x
            points[idx:idx+n, 1] = y
            idx += n
            
        if idx >= N:
            break
            
    # If we didn't fill all N points (due to ring limits), place remaining randomly
    # inside the disk, ensuring they don't overlap too much initially.
    # This is a fallback, but the ring packing should fill most of the disk.
    if idx < N:
        # Simple random placement in the remaining space
        # Just scatter them and let the optimizer fix overlaps
        while idx < N:
            r_rand = np.sqrt(np.random.rand()) * (1.0 - r_init)
            theta_rand = np.random.rand() * 2 * np.pi
            points[idx, 0] = r_rand * np.cos(theta_rand)
            points[idx, 1] = r_rand * np.sin(theta_rand)
            idx += 1

    # --- Stage 2: Gradient-Based Optimization ---
    # Objective: Maximize the minimum clearance.
    # Let d_min be the minimum distance between any pair of circles.
    # Let b_min be the minimum distance from any circle center to the boundary (1 - |p_i|).
    # We want to maximize r = min(d_min / 2, b_min).
    
    # Since maximizing a minimum is non-smooth, we use a soft approximation:
    # Maximize the sum of log(2*r_target - d_ij) for pairs that are too close?
    # Or simpler: Minimize the maximum violation.
    
    # A robust approach for "max-min" is to introduce a variable r and maximize r
    # subject to constraints d_ij >= 2r and |p_i| + r <= 1.
    # This is a Non-Linear Programming (NLP) problem.
    # scipy.optimize.minimize can handle SLSQP or trust-constr.
    
    # However, for N=300, the number of constraints is O(N^2) = 90,000.
    # This might be slow for SLSQP.
    
    # Alternative: Use a penalty method.
    # Minimize:  -r + lambda * (sum of penalty terms)
    # Penalty terms:
    # 1. max(0, 2r - d_ij)^2 for all pairs i,j
    # 2. max(0, |p_i| + r - 1)^2 for all i
    
    # To make it differentiable and efficient, we can optimize for a fixed r_target
    # and check if it's feasible. If feasible, increase r_target.
    # This is a "bisection" or "incremental" approach on r.
    
    # Let's try a direct optimization of a smooth surrogate for the minimum distance.
    # f(x) = -log(sum(exp(-d_ij / T))) approximates -min(d_ij) as T->0.
    # But we have boundary constraints too.
    
    # Let's stick to the Penalty Method with L-BFGS-B, which is fast for unconstrained
    # (or box-constrained) problems.
    
    # Variables: x (flattened points, size 2N) and r (size 1). Total 2N+1 = 601 variables.
    # This is manageable for L-BFGS-B.
    
    def objective_and_grad(vars):
        """
        vars: [x1, y1, ..., xN, yN, r]
        Returns: objective value and gradient.
        Objective: -r + penalty
        """
        if time.time() > deadline - 5.0:
            return 0, np.zeros_like(vars) # Timeout guard inside function? 
            # Actually, minimize doesn't easily interrupt. We rely on the outer loop.
            
        pts = vars[:2*N].reshape(N, 2)
        r = vars[2*N]
        
        # Boundary penalty
        norms = np.linalg.norm(pts, axis=1)
        boundary_viol = np.maximum(0, norms + r - 1.0)
        boundary_pen = np.sum(boundary_viol ** 2)
        
        # Pairwise penalty
        # For N=300, cdist is 300x300.
        # We only care about pairs where d < 2r + epsilon.
        # But for gradient, we need all? No, penalty is 0 if d >= 2r.
        # Computing full cdist is O(N^2) = 90k operations. Fast in numpy.
        
        dists = cdist(pts, pts)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        
        # We want penalty for d < 2r
        # violation = max(0, 2r - d)
        violations = np.maximum(0, 2*r - dists)
        pair_pen = np.sum(violations ** 2)
        
        obj = -r + 100.0 * (boundary_pen + pair_pen) # Weight 100 to prioritize feasibility
        
        # Gradient
        grad = np.zeros_like(vars)
        
        # Gradient w.r.t r
        # d/dr (-r) = -1
        # d/dr (pair_pen) = sum( 2 * violations * d(2r-d)/dr ) = sum( 2 * violations * 2 ) for active pairs
        # d/dr (boundary_pen) = sum( 2 * b_viol * d(norms+r-1)/dr ) = sum( 2 * b_viol * 1 ) for active boundaries
        
        active_pairs = violations > 0
        grad_r = -1.0 + 100.0 * (2.0 * np.sum(violations[active_pairs]) + 2.0 * np.sum(boundary_viol[boundary_viol > 0]))
        # Note: sum(violations) already includes zeros. So just sum(violations).
        grad_r = -1.0 + 100.0 * (2.0 * np.sum(violations) + 2.0 * np.sum(boundary_viol))
        
        grad[2*N] = grad_r
        
        # Gradient w.r.t points
        # d/dp_i (pair_pen) = sum_{j != i} 2 * violation_ij * d(2r - d_ij)/d p_i
        # d(2r - d_ij)/d p_i = - (p_i - p_j) / d_ij
        
        # This is O(N^2) to compute.
        # grad_pts = np.zeros_like(pts)
        # for i in range(N):
        #   for j in range(N):
        #     if i==j: continue
        #     if violations[i,j] > 0:
        #        vec = pts[i] - pts[j]
        #        d = dists[i,j]
        #        grad_pts[i] += 2 * violations[i,j] * (-vec / d)
        
        # Vectorized version:
        # violations is NxN.
        # We need to sum over j for each i.
        
        # Let V be the matrix of violations.
        # Let P be the matrix of points.
        # diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]  # (N, N, 2)
        # dists = np.linalg.norm(diff, axis=2)
        
        # This creates a large array (300x300x2). 180,000 floats. Fine.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists already computed, but let's ensure consistency.
        # We can reuse dists.
        
        # Factor = 2 * violations / dists
        # Handle division by zero (diagonal)
        factor = np.zeros_like(dists)
        mask = dists > 1e-9
        factor[mask] = 2.0 * violations[mask] / dists[mask]
        
        # grad_pts[i] = sum_j ( - factor[i,j] * diff[i,j] )
        # diff[i,j] = p_i - p_j
        # So grad_pts[i] = - sum_j factor[i,j] * (p_i - p_j)
        
        # Using einsum or matrix multiplication
        # grad_pts = - (factor @ diff)
        # factor is (N,N), diff is (N,N,2).
        # factor @ diff works if we view diff as (N, N, 2).
        
        grad_pts = -np.einsum('ij,ijk->ik', factor, diff)
        
        # Boundary gradient
        # d/dp_i (boundary_pen) = sum active 2 * b_viol * d(norms)/d p_i
        # d(norms)/d p_i = p_i / norms_i
        b_grad = np.zeros_like(pts)
        active_b = boundary_viol > 0
        if np.any(active_b):
            # norms is (N,)
            # We need p_i / norms_i for active i
            b_vecs = pts[active_b] / norms[active_b, np.newaxis]
            b_grad[active_b] = 2.0 * boundary_viol[active_b, np.newaxis] * b_vecs
        
        grad_pts += 100.0 * b_grad
        
        grad[:2*N] = grad_pts.flatten()
        
        return obj, grad

    # Initial guess for r
    r_start = 0.052
    
    # We will try to increase r in steps.
    # Start with a feasible r, then push it up.
    
    # First, ensure the initial points are valid for r_start.
    # If not, project them?
    # For now, assume the ring packing is valid for r_start.
    
    x0 = np.zeros(2*N + 1)
    x0[:2*N] = points.flatten()
    x0[2*N] = r_start
    
    # Run optimization
    # L-BFGS-B is good for this.
    # maxiter should be limited to fit in time.
    # Each iteration is O(N^2) ~ 90k ops. 
    # 1000 iterations ~ 90M ops. Should be < 1s in NumPy? Maybe 5-10s.
    # Let's allow up to 500 iterations per r step.
    
    best_x = x0.copy()
    best_r = r_start
    
    # Try to increase r
    current_r = r_start
    max_r = 0.0535 # Target slightly above best known
    
    # We'll do a few rounds of optimization with increasing r_target
    # Round 1: Optimize with r = 0.052
    # Round 2: Optimize with r = 0.0525
    # Round 3: Optimize with r = 0.053
    # Round 4: Optimize with r = 0.0532
    
    r_targets = [0.052, 0.0525, 0.053, 0.0532, 0.0534]
    
    for target_r in r_targets:
        if time.time() > deadline - 20.0:
            break
            
        # Set initial r to current best r, but allow it to move?
        # Actually, if we fix r in the objective as a variable, it might drift down.
        # Better: Fix r to target_r and only optimize points.
        # This turns it into a feasibility check / minimization of violations.
        
        def objective_fixed_r(vars_pts):
            pts = vars_pts.reshape(N, 2)
            r = target_r
            
            norms = np.linalg.norm(pts, axis=1)
            boundary_viol = np.maximum(0, norms + r - 1.0)
            boundary_pen = np.sum(boundary_viol ** 2)
            
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            violations = np.maximum(0, 2*r - dists)
            pair_pen = np.sum(violations ** 2)
            
            obj = boundary_pen + pair_pen
            
            # Gradient
            grad_pts = np.zeros_like(pts)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            factor = np.zeros_like(dists)
            mask = dists > 1e-9
            factor[mask] = 2.0 * violations[mask] / dists[mask]
            
            grad_pts -= np.einsum('ij,ijk->ik', factor, diff)
            
            active_b = boundary_viol > 0
            if np.any(active_b):
                b_vecs = pts[active_b] / norms[active_b, np.newaxis]
                grad_pts[active_b] += 2.0 * boundary_viol[active_b, np.newaxis] * b_vecs
                
            return obj, grad_pts.flatten()

        # Initial points for this stage: use the result from the previous stage
        # If it's the first stage, use the ring packing
        x_init = best_x[:2*N].copy()
        
        # Optimize
        res = minimize(
            objective_fixed_r, 
            x_init, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 1e-6:
            # Check if the solution is valid for this r
            final_pts = res.x.reshape(N, 2)
            dists = cdist(final_pts, final_pts)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            norms = np.linalg.norm(final_pts, axis=1)
            min_boundary = np.min(1.0 - norms)
            achieved_r = min(min_pair / 2.0, min_boundary)
            
            if achieved_r >= target_r - 1e-5:
                best_x[:2*N] = res.x
                best_x[2*N] = target_r
                best_r = target_r
            else:
                # Failed to reach target, keep previous best
                pass
        else:
            # Optimization failed, maybe keep previous
            pass

    # Final check and return
    final_pts = best_x[:2*N].reshape(N, 2)
    
    # Ensure validity
    norms = np.linalg.norm(final_pts, axis=1)
    if np.any(norms > 1.0):
        final_pts[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]
        
    return final_pts

# EVOLVE-BLOCK-END
