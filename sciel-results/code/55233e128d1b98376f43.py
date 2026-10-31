"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing the sum of radii.
    Uses a coverage-based formulation with differentiable soft-min operators
    to enforce that a grid of probe points is covered by the union of disks.
    """
    n = N
    np.random.seed(42)
    
    # Define probe points: 10x10 grid for better coverage enforcement
    grid_size = 10
    probe_x = np.linspace(0.05, 0.95, grid_size)
    probe_y = np.linspace(0.05, 0.95, grid_size)
    probes = np.array([(x, y) for x in probe_x for y in probe_y])
    num_probes = len(probes)
    
    # Initial placement: Hexagonal-like grid
    # Approximate radius for 26 circles in unit square: area ~ 1, 26*pi*r^2 ~ 1 -> r ~ 0.1
    # Let's start with a 5x5 grid plus one center, or a tighter hexagonal pattern.
    # A 5x5 grid has 25 spots. Add one in the center of one of the cells?
    # Better: Use a random restart or a structured grid.
    # Let's use a 5x5 grid of centers, scaled to fit, and add a 26th.
    # Actually, a hexagonal packing is denser.
    # Let's just use a grid initialization that is feasible.
    
    # Create a 5x5 grid of centers
    centers = []
    r_init = 0.15
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            centers.append([x, y])
    # Add 26th circle in the middle of the square, might overlap, optimizer will fix
    centers.append([0.5, 0.5])
    
    init_centers = np.array(centers)
    init_radii = np.full(n, r_init)
    
    # Parameters: x, y, r
    x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], init_radii])
    
    def unpack(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        return x, y, r
    
    def coverage_penalty(params, t=20.0):
        x, y, r = unpack(params)
        penalty = 0.0
        
        # Vectorize over probes for efficiency? 
        # probes shape (P, 2), x,y,r shape (n,)
        # dists shape (P, n)
        dx = probes[:, 0:1] - x[None, :]  # (P, n)
        dy = probes[:, 1:2] - y[None, :]  # (P, n)
        dists = np.sqrt(dx**2 + dy**2 + 1e-12) # (P, n)
        
        # z_k = dist - r_k. If z_k > 0, point is outside circle k.
        # We want min_k z_k <= 0 for all points.
        # Soft min of z over k.
        z = dists - r[None, :] # (P, n)
        
        # Soft min: -1/t * log(sum(exp(-t*z)))
        # To avoid overflow, shift by max
        z_max = np.max(z, axis=1, keepdims=True) # (P, 1)
        exp_z = np.exp(-t * (z - z_max)) # (P, n)
        sum_exp = np.sum(exp_z, axis=1) # (P,)
        soft_min = z_max[:, 0] - (1.0/t) * np.log(sum_exp) # (P,)
        
        # Penalty for positive soft_min (uncovered)
        # Use ReLU squared or just square
        # If soft_min > 0, it means the point is likely outside all circles.
        # We penalize the square of the positive part.
        pos_part = np.maximum(soft_min, 0.0)
        penalty = np.sum(pos_part**2)
        
        return penalty

    def penalized_objective(params, mu=1.0):
        x, y, r = unpack(params)
        
        # Base objective: maximize sum of radii -> minimize -sum(r)
        obj = -np.sum(r)
        
        # Boundary penalty: ensure circles stay inside [0,1]^2
        # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # Penalty: sum of squared violations
        b_pen = 0.0
        for i in range(n):
            if x[i] < r[i]: b_pen += (r[i] - x[i])**2
            if 1.0 - x[i] < r[i]: b_pen += (r[i] - (1.0 - x[i]))**2
            if y[i] < r[i]: b_pen += (r[i] - y[i])**2
            if 1.0 - y[i] < r[i]: b_pen += (r[i] - (1.0 - y[i]))**2
            
        # Coverage penalty
        c_pen = coverage_penalty(params)
        
        return obj + mu * (b_pen + c_pen)
        
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-4, 0.5))
        
    best_params = x0.copy()
    # Initial evaluation
    best_val = penalized_objective(best_params, mu=1.0)
    
    # Multi-stage optimization: increase mu to enforce constraints
    # Start with low mu to find a feasible region, then increase to maximize radii
    # Note: As mu increases, the weight of the objective -sum(r) becomes relatively smaller
    # compared to the penalties, so we might need to balance this.
    # Actually, we want to maximize sum(r) subject to constraints.
    # Standard approach: minimize -sum(r) + mu * penalty.
    # If mu is too high, we get small circles that satisfy constraints perfectly.
    # If mu is too low, we get large circles that violate constraints.
    # We want the largest circles that *satisfy* constraints.
    # So we should start with high mu to find feasible positions, then lower mu?
    # No, usually we start with low mu to explore, then increase mu to tighten constraints.
    # But here the objective is to MAXIMIZE radii.
    # If we increase mu, the optimizer will shrink radii to satisfy coverage/bounds.
    # So we want to find the "ceiling" of radii.
    # Strategy: 
    # 1. Find any feasible configuration with moderate radii.
    # 2. Try to increase radii.
    # 
    # Let's try a sequence of mu values.
    # Start with mu=10 to get a valid packing.
    # Then mu=100 to tighten it? No, that will shrink radii.
    # We want to push radii UP.
    # The term -sum(r) pushes radii up. The penalty pushes them down (by violating constraints).
    # If we increase mu, the penalty becomes more dominant, so radii shrink.
    # If we decrease mu, radii grow but constraints are violated.
    # So we want to find the mu where the gradient balances.
    # Alternatively, use a barrier method or just iterate.
    
    # Let's try optimizing with a fixed high mu first to get a valid packing,
    # then relax the constraints slightly to allow radii to grow?
    # Actually, the standard way to maximize radii subject to constraints is to
    # minimize -sum(r) subject to constraints.
    # If we use a penalty method, we minimize -sum(r) + mu * penalty.
    # As mu -> infinity, we get the constrained optimum.
    # So we should INCREASE mu.
    
    mu_schedule = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for mu in mu_schedule:
        res = minimize(
            penalized_objective,
            best_params,
            args=(mu,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
        else:
            # If it didn't improve, keep previous best
            pass
            
    x, y, r = unpack(best_params)
    
    # Post-processing: Fix any remaining overlaps and boundary violations
    # This is a local repair step.
    for _ in range(50):
        adjusted = False
        # Boundary check
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                required = r[i] + r[j]
                if dist < required:
                    # Scale down both radii slightly to resolve overlap
                    # Or move centers. Scaling radii is easier but reduces objective.
                    # Moving centers is better for objective.
                    # Let's move centers apart.
                    if dist > 1e-6:
                        dir_x = dx / dist
                        dir_y = dy / dist
                        shift = (required - dist) / 2.0 + 1e-6
                        x[i] += dir_x * shift
                        y[i] += dir_y * shift
                        x[j] -= dir_x * shift
                        y[j] -= dir_y * shift
                    else:
                        # If centers are identical, push apart randomly
                        x[i] += 0.01
                        y[i] += 0.01
                    adjusted = True
                    
        if not adjusted:
            break
            
    # Final boundary clip
    for i in range(n):
        x[i] = np.clip(x[i], r[i], 1.0 - r[i])
        y[i] = np.clip(y[i], r[i], 1.0 - r[i])
        
    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
