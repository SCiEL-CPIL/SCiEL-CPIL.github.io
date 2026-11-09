import numpy as np
import time

N = 50

def _radius_score(centers):
    """Compute the radius score for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    
    # Pairwise distances
    # Vectorized pairwise distance calculation
    # diff[i, j, k] = centers[i, k] - centers[j, k]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal (self-distance)
    np.fill_diagonal(dist, np.inf)
    
    min_pairwise = np.min(dist)
    min_pairwise_radius = min_pairwise / 2.0
    
    return min(min_boundary, min_pairwise_radius)

def _physics_simulate(init_centers, max_iters=5000, time_limit=10.0):
    """
    Simulate repulsive forces to pack circles.
    Forces:
    1. Pairwise repulsion if distance < 2*r_current (or a target distance).
       Actually, to maximize radius, we want to push them apart until they touch.
       A standard approach is to simulate particles with a repulsive potential
       that is very strong when overlapping, and let them settle.
       
       Better approach: 
       We don't know the final radius R. 
       Instead, we can simulate a system where particles repel each other with a 
       force proportional to overlap, and are attracted to the center or confined.
       
       A robust method for packing is the "soft disk" model:
       Energy E = sum_{i<j} k * (overlap_ij)^2 + boundary_penalty
       Where overlap_ij = max(0, 2*R_target - dist_ij).
       
       But R_target is unknown. 
       
       Alternative: 
       Use a "pressure" approach. 
       Start with a small radius. 
       The simulation should naturally expand the "effective size" of particles 
       as they find space.
       
       Let's use a simpler heuristic:
       Iterate:
       1. Calculate forces based on current "virtual radius" r.
       2. Move particles.
       3. Increase r slightly.
       
       Or, even simpler and very effective for small N:
       Random Restack / Jitter + Local Push.
       
       Let's implement a specific algorithm:
       "Repulsive Force Relaxation"
       We assume a target radius R. 
       If we pick R close to the optimal, we can check if a valid config exists.
       But we want to maximize R.
       
       Strategy: 
       1. Start with a random config.
       2. Run a simulation where particles repel each other with a strong force 
          if they are closer than a certain distance d_min.
          The force pushes them apart.
          The boundaries act as hard walls.
          We don't fix R. We just let them find the "maximum entropy" or "maximum spread" state.
          In a square, the "maximum spread" state for equal disks corresponds to the densest packing 
          (locally).
          
          Force F_ij = k * (d_target - dist_ij) if dist_ij < d_target, else 0.
          What is d_target? 
          If we set d_target to a large value, they will spread out.
          If we set d_target to 0, they collapse.
          
          Actually, the standard "Molecular Dynamics" for packing uses a potential:
          V(r) = 0 if r > sigma
          V(r) = (sigma - r)^2 if r < sigma
          
          Here sigma is the diameter. 
          If we don't know sigma, we can't define the potential.
          
          Let's use the "Expanding Radius" method:
          1. Initialize centers.
          2. Set r = current_min_dist/2.
          3. While time remains:
             a. Try to increase r by a small amount dr.
             b. Run a few steps of force-directed relaxation with radius r.
             c. If relaxation fails (particles can't separate), decrease r or stop.
             
          This is essentially a local search on r.
          
          Let's implement a simpler, robust "Jitter and Push":
          1. Start with init.
          2. For many iterations:
             - Compute min_dist and min_boundary.
             - Identify the "bottleneck" constraint (smallest gap).
             - Apply a small random perturbation to all particles.
             - Apply a "push" force to particles that are violating the "target" gap.
             
          Let's try a direct gradient ascent on the radius score using a custom step.
          Score S = min(min_pairwise/2, min_boundary).
          This is non-differentiable.
          
          Let's use the "Soft-Body" simulation with a fixed, slightly too-large diameter, 
          and see if it converges to a valid state. If it does, the radius is at least that.
          We can binary search on the diameter? No, simulation is stochastic.
          
          Let's use the algorithm from "Packing circles in a square" literature:
          1. Start with a grid.
          2. Apply random forces.
          3. Apply repulsive forces between circles.
          4. Apply attractive forces to walls? No, repulsive from walls.
          
          Let's implement a standard MD simulation for hard disks with a "temperature" that decays.
          We don't know the diameter. 
          
          Insight: The optimal packing is the one where the circles are "stuck".
          We can simulate particles that repel each other with a force F = 1/d^2 (or similar) 
          and are confined in the box. 
          As they spread out, the minimum distance increases. 
          The process stops when no more energy can be extracted (local minimum of potential energy).
          For repulsive forces, the potential energy is minimized when particles are as far apart as possible.
          This corresponds to the maximum minimum distance.
          
          Algorithm:
          1. Initialize positions.
          2. Define Potential U = sum_{i<j} 1/d_ij^4 (or similar steep repulsion) + Boundary Penalty.
          3. Perform Gradient Descent on U.
          4. The resulting configuration maximizes the minimum distance locally.
          
          Let's use U = sum_{i<j} (1/d_ij)^k. Minimizing this pushes particles apart.
          Boundary penalty: If x < -0.5, add large penalty.
          
          We can use scipy.optimize.minimize on this smooth objective!
          This avoids the constraint non-differentiability.
          
          Objective: Minimize Sum(1/d_ij^4) + Boundary_Violation_Penalty
          
          This is a smooth, convex-like (locally) function that encourages spreading.
          The minimum of this function corresponds to the densest packing (locally).
          
          Let's try this.
          """
    
    # We will use scipy.optimize.minimize on a smooth surrogate objective.
    # Objective: Minimize sum(1/d_ij^4) + 1000 * sum(boundary_violations^2)
    # This effectively maximizes the minimum distance.
    
    from scipy.optimize import minimize
    
    def smooth_objective(x):
        centers = x.reshape(N, 2)
        
        # Boundary penalty
        # We want centers in [-0.5, 0.5]
        # Penalty for going outside
        d_left = centers[:, 0] + 0.5
        d_right = 0.5 - centers[:, 0]
        d_bottom = centers[:, 1] + 0.5
        d_top = 0.5 - centers[:, 1]
        
        # Only penalize if outside
        # Use max(0, -dist)^2
        pen = 0.0
        if np.any(d_left < 0): pen += np.sum(d_left[d_left < 0]**2)
        if np.any(d_right < 0): pen += np.sum(d_right[d_right < 0]**2)
        if np.any(d_bottom < 0): pen += np.sum(d_bottom[d_bottom < 0]**2)
        if np.any(d_top < 0): pen += np.sum(d_top[d_top < 0]**2)
        
        # Pairwise repulsion
        # Sum 1/d^4
        # Avoid division by zero
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid 0
        
        # If dist is very small, 1/d^4 is huge.
        # We can clip dist_sq to a small value to prevent explosion
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        repulsion = np.sum(1.0 / (dist_sq**2))
        
        return 1000.0 * pen + repulsion
    
    # Gradient is not provided, so we use BFGS or L-BFGS-B
    # L-BFGS-B is good for large N and bounds.
    
    # We need to handle the case where the optimizer gets stuck in a bad local minimum.
    # We will run multiple restarts.
    
    best_centers = init_centers.copy()
    best_score = _radius_score(best_centers)
    
    # Try to optimize
    try:
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        # Options
        options = {'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
        
        res = minimize(smooth_objective, init_centers.flatten(), method='L-BFGS-B', 
                       bounds=bounds, options=options)
        
        if res.success or res.fun < smooth_objective(init_centers.flatten()):
            new_centers = res.x.reshape(N, 2)
            new_score = _radius_score(new_centers)
            
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
                
    except Exception:
        pass
        
    return best_centers, best_score

def _construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    # 1. Grid
    # 2. Random
    # 3. Hex-like
    
    inits = []
    
    # Grid 7x8 (56 points, take 50)
    pts = []
    for i in range(8):
        for j in range(7):
            if len(pts) >= N:
                break
            x = (j + 0.5) / 7.0 - 0.5
            y = (i + 0.5) / 8.0 - 0.5
            pts.append([x, y])
    inits.append(np.array(pts[:N]))
    
    # Grid 8x7
    pts = []
    for i in range(7):
        for j in range(8):
            if len(pts) >= N:
                break
            x = (j + 0.5) / 8.0 - 0.5
            y = (i + 0.5) / 7.0 - 0.5
            pts.append([x, y])
    inits.append(np.array(pts[:N]))
    
    # Random
    for _ in range(10):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 2)))
        
    # Hexagonal-ish
    # Spacing 0.12
    pts = []
    y = -0.45
    row = 0
    while len(pts) < N and y < 0.45:
        offset = 0.0 if row % 2 == 0 else 0.06
        x = -0.45 + offset
        while x < 0.45 and len(pts) < N:
            pts.append([x, y])
            x += 0.12
        y += 0.10
        row += 1
    if len(pts) >= N:
        inits.append(np.array(pts[:N]))
        
    # Run physics/optimization on each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Jitter to break symmetry
        jittered = init + rng.normal(0, 0.01, size=(N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        
        try:
            centers, r = _physics_simulate(jittered)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # Polish the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            # Run again on the best result with tighter tolerance?
            # The smooth objective might have converged. 
            # Let's just try one more time to ensure stability.
            centers, r = _physics_simulate(best_centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        
    # Final validity check and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

def construct():
    return _construct()
