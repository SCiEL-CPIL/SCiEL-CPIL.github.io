"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # We formulate the packing as maximizing the minimum distance between circles
    # and to the boundary. This is equivalent to maximizing the radius r.
    # The objective function is:
    #   f(x) = min( min_{i<j} ||x_i - x_j||, min_i min_k (1/2 - |x_{i,k}|) )
    # We want to maximize f(x).
    #
    # However, standard optimizers minimize. So we minimize -f(x).
    # But f(x) is non-smooth (min of distances). We can use a smooth approximation
    # or use a local search from a good initial guess.
    #
    # A common approach is to use a "soft" objective that sums up penalties for
    # overlaps, but here we want the *minimum* distance to be large.
    #
    # Let's use a different strategy:
    # 1. Start with a grid or random initialization.
    # 2. Use a local optimization method (like L-BFGS-B) to maximize the minimum distance.
    #    To handle the non-smoothness, we can use a "smooth min" approximation:
    #    soft_min(a, b) = -logsumexp(-a, -b) / (something) ... actually that's for sum.
    #
    # Actually, a better approach for packing problems is to use a "repulsive force" model
    # or to directly optimize the minimum distance using a local search.
    #
    # Let's try a simple local search:
    # 1. Initialize points in a grid.
    # 2. Compute the current minimum distance d_min.
    # 3. For each point, try small random perturbations. If the perturbation increases
    #    the minimum distance (or doesn't decrease it too much), accept it.
    #
    # But this is slow. Let's use scipy.optimize.minimize with a smooth objective.
    #
    # Objective: minimize -min_dist
    # This is non-differentiable. We can use Nelder-Mead or Powell which don't require gradients.
    #
    # However, N=200 means 400 variables. Nelder-Mead might be too slow or get stuck.
    #
    # Alternative: Use a "soft" objective that encourages separation.
    # For example, minimize the sum of exp(-k * dist_ij) for all pairs.
    # As k increases, this approximates maximizing the minimum distance.
    #
    # Let's try:
    #   obj(x) = sum_{i<j} exp(-k * ||x_i - x_j||) + sum_i exp(-k * (1/2 - |x_i|))
    #
    # This is smooth and differentiable. We can use L-BFGS-B.
    #
    # Steps:
    # 1. Initialize with a grid.
    # 2. Optimize the soft objective with increasing k.
    #
    # But the problem asks for a specific algorithm: "soft-contact model with high-order polynomial potential".
    # Let's stick closer to the directive.
    #
    # Directive: "rigid-body dynamics problem using a 'soft-contact' model where circles interact
    # via a high-order polynomial potential that penalizes overlap, solved via a symplectic Euler integrator
    # with adaptive time-stepping"
    #
    # So, let's implement a simple physics simulation:
    # 1. Initialize positions in a grid.
    # 2. Define a potential V(r) that is 0 for r >= 1 (non-overlapping) and large for r < 1.
    #    V(r) = (1 - r)^4 for r < 1, else 0. (High-order polynomial)
    # 3. Forces are -grad V.
    # 4. Use symplectic Euler: v_{n+1} = v_n + dt * a_n; x_{n+1} = x_n + dt * v_{n+1}
    # 5. Add damping to settle the system.
    # 6. Adaptive time-stepping: reduce dt if energy increases or if forces are large.
    #
    # We want to maximize the radius, which means we want to maximize the minimum distance.
    # The physics simulation will naturally push circles apart until they are as far as possible.
    #
    # Let's implement this.
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x = grid[:N].copy()
    
    # Randomize slightly to break symmetry
    x += 0.01 * np.random.randn(N, 2)
    
    # Velocity
    v = np.zeros_like(x)
    
    # Parameters
    dt = 0.01
    damping = 0.99
    steps = 10000
    
    for step in range(steps):
        # Compute forces
        forces = np.zeros_like(x)
        
        # Pairwise interactions
        for i in range(N):
            for j in range(i + 1, N):
                diff = x[i] - x[j]
                dist = np.linalg.norm(diff)
                if dist < 1.0:
                    # Potential V(r) = (1 - r)^4
                    # dV/dr = -4(1-r)^3
                    # Force = -dV/dr * (diff/dist) = 4(1-r)^3 * (diff/dist)
                    force_mag = 4.0 * (1.0 - dist) ** 3
                    if dist > 1e-10:
                        force_vec = force_mag * diff / dist
                    else:
                        force_vec = np.zeros(2)
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        # Boundary forces
        for i in range(N):
            for k in range(2):
                # Distance to boundary
                dist_to_boundary = 0.5 - abs(x[i, k])
                if dist_to_boundary < 1.0:
                    # Same potential
                    force_mag = 4.0 * (1.0 - dist_to_boundary) ** 3
                    # Force direction: towards center
                    if x[i, k] > 0:
                        force_vec_k = -force_mag
                    else:
                        force_vec_k = force_mag
                    forces[i, k] += force_vec_k
        
        # Symplectic Euler
        v = v * damping + dt * forces
        x = x + dt * v
        
        # Adaptive time-stepping: if the system is too energetic, reduce dt
        # Simple heuristic: if the max force is large, reduce dt
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 100:
            dt *= 0.9
        elif max_force < 1:
            dt = min(dt * 1.1, 0.01)
        
        # Constrain to square (soft constraint is handled by forces, but hard constrain for safety)
        x = np.clip(x, -0.5, 0.5)
    
    return x
# EVOLVE-BLOCK-END
