"""
Packing 50 equal circles in a unit circle.
Strategy: Parametric ring structure (1+6+12+18+13) optimized via 
Lennard-Jones dynamics (Velocity-Verlet) followed by local gradient 
refinement and binary search on radius.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Ring structure counts: 1 (center) + 6 + 12 + 18 + 13 = 50
RING_COUNTS = [1, 6, 12, 18, 13]

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Boundary constraint: min(1 - |p_i|)
    norms = np.linalg.norm(P, axis=1)
    r_bound = 1.0 - np.max(norms)
    
    # Pairwise constraint: min(|p_i - p_j|) / 2
    # Vectorized distance calculation
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _ring_params_to_points(params):
    """
    Convert parametric ring structure to point positions.
    params: [r1, r2, r3, r4, off1, off2, off3, off4]
    Ring 0 (center) is fixed at (0,0).
    """
    r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
    off1, off2, off3, off4 = params[4], params[5], params[6], params[7]
    
    points = [[0.0, 0.0]]
    
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6 + off1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + off2
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18 + off3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 points
    for i in range(13):
        angle = 2 * np.pi * i / 13 + off4
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _energy(params, r_target, softness=20.0):
    """
    Energy function for dynamics.
    We want to maximize the minimum distance, so we minimize the 'overlap' energy.
    If we fix r_target, we can check for violations.
    However, for dynamics, it's better to use a potential that pushes particles apart.
    
    We use a Lennard-Jones-like potential for pairs and a harmonic well for boundaries.
    But since we are optimizing the *structure* (params), we can define the energy
    as the sum of squared violations of the constraints for a given r_target.
    To make it smooth, we use a soft-min approximation.
    """
    P = _ring_params_to_points(params)
    
    # 1. Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # We want dist >= 2*r_target.
    # Violation v_ij = max(0, 2*r_target - dist_ij)
    # Smooth approximation: v_ij = (1/softness) * log(1 + exp(softness * (2*r_target - sqrt(dist_sq))))
    # To avoid log-sum-exp overflow issues and keep it fast, we use a simpler quadratic penalty for overlaps
    # and a linear penalty for near-overlaps.
    
    # Actually, for dynamics, we need forces. 
    # Let's define a potential U = sum_{i<j} U_pair(d_ij) + sum_i U_bound(||p_i||)
    
    # Pair potential: Repulsive when d < 2*r_target
    # U_pair(d) = 0.5 * k * (2*r_target - d)^2 if d < 2*r_target else 0
    # This is differentiable almost everywhere.
    
    dist = np.sqrt(dist_sq + 1e-12)
    
    # Mask for pairs where d < 2*r_target
    mask = dist < 2.0 * r_target
    # Set diagonal to 0 to avoid self-interaction issues in sum
    np.fill_diagonal(mask, False)
    
    # Calculate violation magnitude
    viol = 2.0 * r_target - dist
    viol = np.where(mask, viol, 0.0)
    
    pair_energy = 0.5 * np.sum(viol**2)
    
    # Boundary potential: Repulsive when ||p_i|| > 1 - r_target
    norms = np.linalg.norm(P, axis=1)
    bound_viol = norms - (1.0 - r_target)
    bound_viol = np.where(bound_viol > 0, bound_viol, 0.0)
    
    bound_energy = 0.5 * np.sum(bound_viol**2)
    
    return pair_energy + bound_energy

def _dynamics_optimize(params_init, r_target, time_budget=5.0, seed=42):
    """
    Use Velocity-Verlet integration to minimize the energy of the ring structure.
    params: 8-dimensional vector.
    """
    rng = np.random.default_rng(seed)
    
    # Initial parameters
    params = params_init.copy()
    velocity = np.zeros(8)
    
    # Learning rate / Step size for parameters
    # Parameters are radii (0..1) and angles (0..2pi).
    # We need to scale the forces appropriately.
    dt = 0.01
    
    start_time = time.time()
    
    # Pre-compute ring indices for force calculation? 
    # It's easier to just recompute P and forces numerically or analytically.
    # Given low dimension (8), numerical gradient is fast enough.
    
    def get_forces(p):
        eps = 1e-5
        forces = np.zeros(8)
        for i in range(8):
            p_plus = p.copy()
            p_plus[i] += eps
            e_plus = _energy(p_plus, r_target)
            
            p_minus = p.copy()
            p_minus[i] -= eps
            e_minus = _energy(p_minus, r_target)
            
            forces[i] = -(e_plus - e_minus) / (2 * eps)
        return forces
    
    # Main loop
    while time.time() - start_time < time_budget:
        force = get_forces(params)
        
        # Damping
        velocity = velocity * 0.9 + force * dt
        
        # Update params
        new_params = params + velocity * dt
        
        # Constrain angles to [0, 2pi] to avoid wrap-around issues in gradients?
        # Actually, angles are periodic. But for simplicity, let's allow them to go out of range
        # and let the cosine/sine handle it. However, the gradient might suffer if we cross boundaries.
        # Better to map angles back to [0, 2pi] for the next evaluation if they drift too far.
        
        # Simple constraint: keep radii positive and < 1
        new_params[0:4] = np.clip(new_params[0:4], 0.01, 0.99)
        
        # Check if energy decreased (optional, for stability)
        # If not, revert and dampen
        e_old = _energy(params, r_target)
        e_new = _energy(new_params, r_target)
        
        if e_new < e_old or np.random.random() < 0.1: # Always accept if better, sometimes if worse (annealing-like)
            params = new_params
        else:
            velocity *= 0.5 # Dampen if bad step
            # Revert
            params = params - velocity * dt # This logic is getting messy, let's just use a simple gradient descent with momentum
            
    # Switch to simple gradient descent for the last bit of time if dynamics is unstable
    # Actually, let's just run a standard L-BFGS-B from the result of dynamics.
    return params

def _local_optimize(params, r_target, deadline):
    """
    Local optimization using L-BFGS-B to minimize energy for a fixed r_target.
    """
    def objective(p):
        return _energy(p, r_target)
    
    # Bounds: radii (0.01, 0.99), angles (0, 2*pi)
    bounds = [
        (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99),
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)
    ]
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        return result.x
    except Exception:
        return params

def _binary_search_radius(params, r_lo, r_hi, deadline, max_iter=40):
    """
    Binary search for the maximum r such that the structure is valid.
    """
    best_params = params.copy()
    best_r = r_lo
    
    for _ in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Optimize structure for this r
        # Use a few steps of gradient descent to settle
        p_opt = _local_optimize(params, r_mid, deadline)
        
        # Check if valid
        P = _ring_params_to_points(p_opt)
        
        # Check constraints
        norms = np.linalg.norm(P, axis=1)
        boundary_ok = np.all(norms <= 1.0 - r_mid + 1e-9)
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pairwise_ok = np.all(dist >= 2.0 * r_mid - 1e-9)
        
        if boundary_ok and pairwise_ok:
            r_lo = r_mid
            best_r = r_mid
            best_params = p_opt
        else:
            r_hi = r_mid
            # Keep previous best params for next iteration? Or use p_opt as start?
            # Using p_opt is better as it's closer to the constraint surface.
            params = p_opt
            
    return best_params, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial guess for ring parameters
    # Radii roughly: 0.3, 0.55, 0.75, 0.92
    # Offsets: 0, 0, 0, 0
    initial_params = np.array([0.3, 0.55, 0.75, 0.92, 0.0, 0.0, 0.0, 0.0])
    
    best_params = initial_params
    best_score = -1.0
    
    # Phase 1: Find a good initial radius and structure using dynamics/annealing
    # We don't know the exact r, so we estimate it.
    # Area argument: 50 * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(50) ~ 0.141
    # Best known is 0.1258.
    
    # Try a few initial radii with dynamics
    candidate_rs = [0.120, 0.123, 0.125, 0.126, 0.127]
    
    for r_est in candidate_rs:
        if time.time() > deadline - 30.0:
            break
            
        # Run dynamics to settle structure for this r
        # We use a short budget for dynamics
        p_dyn = _dynamics_optimize(initial_params, r_est, time_budget=5.0, seed=int(r_est*1000))
        
        # Local polish
        p_loc = _local_optimize(p_dyn, r_est, deadline)
        
        # Check score
        P = _ring_params_to_points(p_loc)
        score = _compute_score(P)
        
        if score > best_score:
            best_score = score
            best_params = p_loc
            
    # Phase 2: Binary search for max radius starting from best found
    if best_score > 0:
        r_lo = best_score
        r_hi = best_score + 0.005 # Small range
        
        # Ensure r_hi is feasible? No, binary search handles it.
        # But we need a valid start.
        
        best_params, best_r = _binary_search_radius(best_params, r_lo, r_hi, deadline)
        
        # Final verification and score
        P_final = _ring_params_to_points(best_params)
        final_score = _compute_score(P_final)
        
        if final_score > best_score:
            best_score = final_score
            best_params = best_params
        else:
            # If binary search failed or got stuck, stick with previous best
            P_final = _ring_params_to_points(best_params)
            # Re-verify
            if _compute_score(P_final) < best_score:
                 # Fallback to the best_params from phase 1
                 pass 

    # Return the best points found
    # Ensure we return the points corresponding to the best score
    # We need to track which params gave the best score
    # In the loop above, we updated best_params when best_score improved.
    # In binary search, we returned best_params.
    
    # One final check: if the returned points are invalid, try a safe fallback
    P_out = _ring_params_to_points(best_params)
    if not np.all(np.isfinite(P_out)) or P_out.shape != (N, 2):
         # Fallback to a simple grid
         P_out = np.zeros((N, 2))
         # ... generate valid points ...
    
    return P_out

# EVOLVE-BLOCK-END
