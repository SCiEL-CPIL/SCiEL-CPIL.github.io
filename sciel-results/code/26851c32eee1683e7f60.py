"""
Packing 50 circles in a unit square using Simulated Annealing with a Log-Sum-Exp Surrogate.

Strategy:
1.  Use a high-quality initial configuration based on the known optimal packing structure
    (Boll et al. 2000) or a dense hexagonal grid perturbed to fit the 50-circle constraint.
2.  Define a smooth surrogate objective function: 
    E(T) = -T * log( sum_{constraints} exp(-d_k / T) )
    where d_k are the pairwise distances and boundary distances.
    As T -> 0, this approaches -min(d_k), which we want to maximize.
    The gradient of this surrogate provides smooth forces that push all "tight" constraints apart,
    not just the single minimum, allowing the system to escape local minima more effectively than
    simple relaxation or raw force simulation.
3.  Use Simulated Annealing:
    - Start with a high temperature T.
    - Compute gradient of the surrogate energy.
    - Update positions: x_new = x_old + alpha * grad + noise.
    - Cool T down according to a schedule.
    - Accept moves based on Metropolis criterion (though with gradient descent, we usually just
      follow the gradient, but adding stochastic noise helps).
    - Actually, a more robust variant for packing is "Soft-Body" with a specific potential.
    Let's use the "Soft-Body" approach with a specific potential that is smooth everywhere:
    U = sum_{i<j} (1/d_ij)^p + sum_i (1/d_wall_i)^p
    But we want to maximize the MIN distance.
    
    Better Strategy: "Repulsion with Adaptive Temperature"
    1. Initialize with a good guess (Hex grid + jitter).
    2. Loop:
       a. Compute all pairwise and boundary distances.
       b. Identify the "critical" set of constraints (those close to the current min).
       c. Apply forces to increase these distances.
       d. Use a simulated annealing temperature to control the magnitude of random perturbations
          vs. deterministic gradient steps.
    
    Let's implement a robust Simulated Annealing on the objective function:
    f(x) = min( min_{i<j} ||x_i - x_j||, min_i min_k (0.5 - |x_ik|) )
    
    Since f(x) is non-smooth, we use a smooth approximation:
    F_T(x) = -T * log( sum_{c} exp(-d_c / T) )
    where c iterates over all pairwise and boundary distances.
    
    Gradient of F_T:
    dF/dx_i = (1/T) * sum_c (d_c / T * exp(-d_c/T)) / (sum_c' exp(-d_c'/T)) * (d d_c / d x_i)
             = sum_c w_c * (d d_c / d x_i)
    where w_c = exp(-d_c/T) / sum exp(-d_c'/T) are softmax weights.
    
    This effectively pushes apart the "most constrained" pairs, but with a smooth blend of all constraints.
    As T decreases, the weights concentrate on the minimum distance constraints.
"""

import numpy as np
import time

N = 50

def _compute_radius(centers):
    """
    Compute the maximum common radius for a given set of centers.
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    centers = np.clip(centers, -0.5, 0.5)
    
    # Boundary distance
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    
    # Pairwise distance
    min_pairwise = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_pairwise:
                min_pairwise = dist
    
    min_pairwise_radius = min_pairwise / 2.0
    
    return min(min_boundary, min_pairwise_radius)

def _hex_init_perturbed(seed=42):
    """
    Generate a hexagonal grid initialization that fits 50 circles well.
    The optimal packing for 50 circles is not a perfect hex grid, but a perturbed one
    starting from a hex-like arrangement is a much better starting point than random.
    """
    rng = np.random.default_rng(seed)
    
    # Approximate layout: 5 rows of 10? Or 7x8?
    # 50 = 5 * 10.
    # Let's try a 5x10 grid with hexagonal offset.
    rows = 5
    cols = 10
    
    points = []
    # Scale to fit in [-0.5, 0.5]
    # Width span: (cols - 1) * dx + dx/2 (offset) ?
    # Let's just generate a hex grid and scale it.
    
    dx = 1.0 / (cols - 0.5) # Approx spacing
    dy = dx * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j - (cols - 1) / 2.0) * dx
            y = (i - (rows - 1) / 2.0) * dy
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    points = np.array(points[:N])
    
    # Scale to fit within [-0.5, 0.5] with some margin
    # Find extent
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # We want the final extent to be slightly less than 1.0
    # Let's scale to fit in [-0.45, 0.45] initially to allow room for optimization
    target_extent = 0.9
    current_extent = max(max_x - min_x, max_y - min_y)
    scale = target_extent / current_extent if current_extent > 0 else 1.0
    
    points *= scale
    
    # Center
    points -= points.mean(axis=0)
    
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.01, size=points.shape)
    
    # Clip to valid range
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _random_init(rng):
    """Generate a random initial configuration."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _simulated_annealing(centers_init, time_limit=20.0, seed=42):
    """
    Run Simulated Annealing to maximize the minimum distance.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    pos = centers_init.copy()
    
    # Parameters
    T = 0.01 # Initial temperature (scale of distances)
    T_min = 1e-6
    cooling = 0.995 # Cooling rate per step
    alpha = 0.05 # Step size for gradient
    noise_scale = 0.01 # Scale of random noise
    
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Pre-compute indices for pairwise distances for efficiency?
    # For N=50, 1225 pairs. It's fine to loop.
    
    while time.time() - start_time < time_limit:
        # 1. Compute distances and gradients
        # We want to maximize min(d).
        # Surrogate: F = -T * log( sum exp(-d/T) )
        # Grad F = sum w_i * grad(d_i)
        # w_i = exp(-d_i/T) / sum exp(-d_j/T)
        
        # Collect all distances and their gradients (as vectors in R^N x 2)
        # This is memory intensive if we store all gradients.
        # Instead, we accumulate the weighted gradient directly.
        
        # To avoid numerical overflow in exp(-d/T), we can subtract the min d.
        # But we need the weights for the sum.
        # Let's use a stable log-sum-exp trick.
        
        d_vals = []
        # We can't easily vectorize the gradient accumulation without storing intermediates
        # or doing it in a loop. Given N=50, a loop is acceptable.
        
        # First pass: compute all d and find min_d for stability
        min_d = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
            for k in range(2):
                d_wall = 0.5 - np.abs(pos[i, k])
                if d_wall < min_d:
                    min_d = d_wall
        
        # If min_d is very small, T should be small relative to it?
        # Actually, T is a hyperparameter. Let's keep T fixed or adaptive.
        # If T is too large, the gradient is smooth but weak.
        # If T is too small, it's like subgradient of min.
        
        # Second pass: accumulate gradient
        grad = np.zeros_like(pos)
        sum_exp = 0.0
        exp_terms = [] # Store (d, exp_val, grad_contrib) ? No, too much memory.
        
        # We need sum_exp to normalize weights.
        # Let's do two passes.
        # Pass 1: Compute d and exp(-d/T) scaled by exp(min_d/T) to avoid underflow.
        # exp(-(d - min_d)/T)
        
        scaled_exp_terms = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1e-9:
                    d = 1e-9
                scaled_exp = np.exp(-(d - min_d) / T)
                scaled_exp_terms.append((d, scaled_exp, i, j, 'pair'))
            for k in range(2):
                d_wall = 0.5 - np.abs(pos[i, k])
                if d_wall < 1e-9:
                    d_wall = 1e-9
                scaled_exp = np.exp(-(d_wall - min_d) / T)
                scaled_exp_terms.append((d_wall, scaled_exp, i, k, 'wall'))
                
        sum_scaled_exp = sum(term[1] for term in scaled_exp_terms)
        
        if sum_scaled_exp == 0:
            break
            
        # Pass 2: Accumulate weighted gradient
        # w_c = exp_c / sum_exp
        # grad_d_c is the gradient of distance c w.r.t positions.
        
        for d_val, exp_val, idx1, idx2, type in scaled_exp_terms:
            w = exp_val / sum_scaled_exp
            
            if type == 'pair':
                i, j = idx1, idx2
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-9:
                    continue
                # grad of d w.r.t x_i is (x_i - x_j) / d
                # grad of d w.r.t x_j is (x_j - x_i) / d
                gx = dx / d
                gy = dy / d
                grad[i, 0] += w * gx
                grad[i, 1] += w * gy
                grad[j, 0] -= w * gx
                grad[j, 1] -= w * gy
            else: # wall
                i = idx1
                axis = idx2
                # d = 0.5 - |x_i, axis|
                # grad w.r.t x_i, axis:
                # if x > 0, d = 0.5 - x, grad = -1
                # if x < 0, d = 0.5 + x, grad = 1
                if pos[i, axis] > 0:
                    g = -1.0
                else:
                    g = 1.0
                grad[i, axis] += w * g
        
        # Update position
        # We want to maximize d, so we move in direction of grad (which points to increasing d).
        # Wait, grad of F = sum w * grad(d).
        # If we move x += alpha * grad, we increase F?
        # F is a surrogate for -min(d). We want to minimize F.
        # So we should move x -= alpha * grad.
        # Let's check signs.
        # d = ||x_i - x_j||. grad d / dx_i = (x_i - x_j)/d.
        # If we move x_i in direction (x_i - x_j)/d, we increase d.
        # So moving in direction of grad increases d.
        # We want to increase d.
        # So x += alpha * grad is correct for maximizing d.
        
        pos += alpha * grad
        
        # Add noise
        pos += rng.normal(0, noise_scale * T, size=pos.shape)
        
        # Clip to bounds (hard constraint)
        pos = np.clip(pos, -0.5, 0.5)
        
        # Check score
        if np.random.rand() < 0.05: # Check every ~20 steps
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_pos, best_r

def _construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Strategy:
    # 1. Start with a good hex-like init.
    # 2. Run SA.
    # 3. If time remains, try random inits with SA.
    
    inits = []
    inits.append(("hex", _hex_init_perturbed(seed=42)))
    inits.append(("hex2", _hex_init_perturbed(seed=123)))
    
    for _ in range(2):
        inits.append(("rand", _random_init(rng)))
        
    for i, (name, init) in enumerate(inits):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        sim_time = min(remaining * 0.8, 30.0) # Cap individual sim time
        
        if sim_time < 2.0:
            continue
            
        try:
            centers, r = _simulated_annealing(init, seed=42 + i, time_limit=sim_time)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # Refine best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        try:
            # Use a lower temperature start for refinement
            centers, r = _simulated_annealing(best_centers, seed=999, time_limit=remaining)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # Fallback
    if best_centers is None:
        best_centers = _hex_init_perturbed(seed=42)
        
    # Final validity check and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

def construct():
    return _construct()
