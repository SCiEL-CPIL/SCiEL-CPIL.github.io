"""Pack 50 equal circles in a unit square, maximize common radius.

Strategy: Physics-based Molecular Dynamics (MD) simulation.
We model the circles as particles in a 2D box with hard-wall boundaries.
Inter-particle interactions are modeled by a stiff repulsive potential that activates
when the distance between centers is less than a dynamic 'cutoff' radius.
We use a Velocity Verlet integrator with adaptive damping (friction) to simulate
the system's evolution. The process involves an 'annealing' phase where the
cutoff radius is gradually increased, forcing the circles to push against each other
and find a stable, dense packing. This avoids the non-differentiability issues of
the min-function and the topological constraints of contact-graph methods by letting
the physics naturally resolve overlaps and maximize clearance.
"""
import numpy as np
import time

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _md_simulate(centers_init, dt=0.001, steps=5000, stiffness=1000.0, damping=0.99):
    """
    Run a simple MD simulation with Velocity Verlet integration.
    Forces:
    1. Pairwise repulsion: F = k * (r_cut - dist)^2 * dir, if dist < r_cut
    2. Wall repulsion: F = k * (0.5 - |coord| - r_target)^2 * dir, if close to wall
    
    Note: We simulate forces relative to a 'virtual' radius that we try to enforce.
    Actually, a simpler approach for packing is to just repel when overlapping a fixed
    target radius, or use a soft potential that pushes them apart until they touch.
    
    Let's use a potential that penalizes overlap beyond a current 'target radius' r_t.
    However, r_t is what we want to maximize. 
    Alternative: Use a potential that is always repulsive but stiff, and let them
    settle. Then the radius is determined by the minimum distance.
    
    Better approach: 
    We don't know the final radius. We can simulate a system where particles repel
    with a force that is very stiff when they are close. 
    To maximize the minimum distance, we can use an 'inverse distance' potential 
    or a 'soft sphere' potential.
    
    Let's use: V(r) = 0.5 * k * (1/r - 1/r_max)^2 for r < r_max, else 0.
    This is complicated.
    
    Simpler: 
    Use a force that pushes particles apart if they are closer than a certain distance.
    But we want them to be as far apart as possible.
    
    Standard approach for circle packing via MD:
    1. Start with small radius.
    2. Increase radius slowly.
    3. At each step, relax the system using MD where particles repel if dist < 2*r.
    
    Let's implement this 'gradual expansion' with MD relaxation.
    """
    centers = centers_init.copy()
    velocities = np.zeros_like(centers)
    
    # We will try to expand the radius from a small value to a larger one.
    # Initial radius estimate
    r_start = 0.01
    r_end = 0.08 # Slightly larger than expected best (0.0714)
    
    # Number of expansion steps
    n_expansion_steps = 50
    
    radii = np.linspace(r_start, r_end, n_expansion_steps)
    
    # Pre-allocate for speed? No, just loop.
    # To save time, we reduce steps if time is tight.
    # For now, let's assume this is fast enough.
    
    for r_target in radii:
        # Relax at this radius
        # We use a stiff spring force for overlaps: dist < 2*r_target
        # Force magnitude: k * (2*r_target - dist)
        
        for step in range(steps):
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise forces
            # Vectorized calculation of distances and forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Overlap condition
            # We want force to be proportional to penetration depth
            # penetration = 2*r_target - dist
            # Force is along the vector connecting centers, pointing away.
            # F_ij = k * penetration * (c_i - c_j) / dist
            # Only if dist < 2*r_target
            
            k = stiffness
            # Create a mask for overlaps
            mask = dist < 2 * r_target
            
            # To avoid division by zero and handle non-overlaps, we compute force matrix
            # Force on i due to j is k * (2r - dist_ij) * (c_i - c_j) / dist_ij
            # We can compute this for all pairs and sum.
            
            # dist is NxN
            # diff is NxNx2
            # We need (2r - dist) / dist * diff
            # Be careful with dist=0 (diagonal) and dist>=2r.
            
            # Initialize force contribution matrix
            force_mat = np.zeros_like(diff)
            
            # Only compute where mask is True
            # dist_masked = np.where(mask, dist, 1.0) # avoid div by zero
            # factor = np.where(mask, (2*r_target - dist_masked) / dist_masked * k, 0.0)
            
            # Vectorized:
            dist_safe = np.where(mask, dist, 1.0)
            factor = np.where(mask, (2 * r_target - dist_safe) / dist_safe * k, 0.0)
            
            # factor is NxN
            # Force on i is sum over j of factor[i,j] * (c_i - c_j)
            # diff[i,j] = c_i - c_j
            # So force[i] = sum_j factor[i,j] * diff[i,j]
            
            forces += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Wall forces
            # Left wall: x < -0.5 + r_target -> penetration = (-0.5 + r_target) - x
            # Force pushes right (positive x)
            # Right wall: x > 0.5 - r_target -> penetration = x - (0.5 - r_target)
            # Force pushes left (negative x)
            
            # Left
            pen_left = np.maximum(0, (-0.5 + r_target) - centers[:, 0])
            forces[:, 0] += k * pen_left
            
            # Right
            pen_right = np.maximum(0, centers[:, 0] - (0.5 - r_target))
            forces[:, 0] -= k * pen_right
            
            # Bottom
            pen_bottom = np.maximum(0, (-0.5 + r_target) - centers[:, 1])
            forces[:, 1] += k * pen_bottom
            
            # Top
            pen_top = np.maximum(0, centers[:, 1] - (0.5 - r_target))
            forces[:, 1] -= k * pen_top
            
            # Velocity Verlet Update
            # v_half = v + 0.5 * dt * F
            # x_new = x + dt * v_half
            # F_new (recompute)
            # v_new = v_half + 0.5 * dt * F_new
            
            # For simplicity and stability with damping, let's use a simpler Euler-Cromer or similar,
            # or just update velocity and position.
            # Given stiff forces, small dt is needed.
            
            # Apply damping
            velocities *= damping
            
            # Update velocity
            velocities += forces * dt
            
            # Update position
            centers += velocities * dt
            
            # Hard clamp to square (prevents escaping to infinity if force is too strong)
            # If a particle goes outside, it gets pushed back hard in next step,
            # but clamping is safer.
            centers[:, 0] = np.clip(centers[:, 0], -0.5 + r_target, 0.5 - r_target)
            centers[:, 1] = np.clip(centers[:, 1], -0.5 + r_target, 0.5 - r_target)
            
            # Check time?
            # If this inner loop is too slow, we need to reduce steps.
            
    return centers

def _refine_gradient(centers, time_limit=5.0):
    """
    Quick gradient-based refinement using soft-min to polish the MD result.
    MD might leave small gaps or non-optimal local arrangements.
    """
    from scipy.optimize import minimize
    
    def _soft_min_radius(centers, beta=100.0):
        d_left = centers[:, 0] + 0.5
        d_right = 0.5 - centers[:, 0]
        d_bottom = centers[:, 1] + 0.5
        d_top = 0.5 - centers[:, 1]
        wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        idx = np.triu_indices(N, 1)
        pair_dists = dist[idx] / 2.0
        
        all_dists = np.concatenate([wall_dists, pair_dists])
        u = -beta * all_dists
        max_u = np.max(u)
        sum_exp = np.sum(np.exp(u - max_u))
        log_sum = max_u + np.log(sum_exp)
        return -log_sum / beta

    def _objective(x):
        c = x.reshape(N, 2)
        return -_soft_min_radius(c)
    
    x0 = centers.ravel()
    bounds = [(-0.5, 0.5)] * (2 * N)
    start_time = time.time()
    
    try:
        res = minimize(
            _objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return centers

def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    inits = []
    
    # 1. 7x7 Grid + 1 center
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    grid_7x7 = np.array([(x, y) for y in ticks for x in ticks])
    c1 = np.vstack([grid_7x7, np.array([0.0, 0.0])])
    c1 = c1 + np.random.normal(0, 0.01, size=c1.shape)
    inits.append(c1)
    
    # 2. Hexagonal-like grid
    # Generate a hex grid
    pts = []
    dy = 1.0 / 8
    dx = dy * np.sqrt(3) / 2
    for row in range(9):
        y = -0.5 + (row + 0.5) * dy
        offset = 0 if row % 2 == 0 else dx / 2
        for col in range(8):
            x = -0.5 + (col + 0.5) * dx + offset
            if -0.45 < x < 0.45 and -0.45 < y < 0.45:
                pts.append((x, y))
    # Take first 50
    if len(pts) >= 50:
        c2 = np.array(pts[:50])
        c2 = c2 + np.random.normal(0, 0.01, size=c2.shape)
        inits.append(c2)
        
    # 3. Random
    np.random.seed(42)
    c3 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c3)
    
    # 4. Another random
    np.random.seed(123)
    c4 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c4)
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Run MD simulation
        # We need to be careful with the number of steps to stay within time.
        # Let's estimate time for one MD run.
        # If it's too slow, we reduce steps.
        
        # Try a shorter MD run first to see timing
        t_start_md = time.time()
        # Use a subset of steps for the first try to gauge speed
        md_centers = _md_simulate(init, dt=0.0005, steps=1000, stiffness=500.0, damping=0.98)
        t_md = time.time() - t_start_md
        
        r_md = _radius(md_centers)
        
        if r_md > best_r:
            best_r = r_md
            best_centers = md_centers.copy()
        
        # If we have time, do more MD steps or refine
        time_left = DEADLINE - 10 - (time.time() - start_time)
        
        if time_left > 5:
            # Polish with gradient
            polished = _refine_gradient(md_centers, time_limit=min(5.0, time_left * 0.5))
            r_polished = _radius(polished)
            
            if r_polished > best_r:
                best_r = r_polished
                best_centers = polished.copy()
                
    # If time remains, try more random restarts with shorter MD
    while time.time() - start_time < DEADLINE - 5:
        np.random.seed(int(time.time() * 1000) % 10000)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Short MD
        md_centers = _md_simulate(c_rand, dt=0.0005, steps=500, stiffness=500.0, damping=0.98)
        r_md = _radius(md_centers)
        
        if r_md > best_r:
            best_r = r_md
            best_centers = md_centers.copy()
            
            # Quick polish
            time_left = DEADLINE - 5 - (time.time() - start_time)
            if time_left > 2:
                polished = _refine_gradient(md_centers, time_limit=min(2.0, time_left * 0.5))
                r_polished = _radius(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished.copy()

    return best_centers
