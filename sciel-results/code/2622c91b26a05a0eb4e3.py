"""Pack 50 equal circles in a unit square using Physics-based Simulated Annealing."""
import numpy as np
import time
import random

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _generate_initial_grid(n_rows, n_cols):
    """Generate a grid initial configuration."""
    # Center the grid in the square
    # We want N points. If n_rows * n_cols < N, we might need to adjust or just take first N.
    # Better: generate a grid that covers the square.
    # Let's try to fit N points roughly.
    # For 50, 7x7=49 (close), 5x10=50, 10x5=50.
    
    if n_rows * n_cols < N:
        # Add extra points randomly or adjust
        pass
        
    points = []
    # Scale to fit inside [-0.5, 0.5] with some margin
    # Actually, let's just place them evenly.
    # x in [-0.4, 0.4], y in [-0.4, 0.4]
    xs = np.linspace(-0.4, 0.4, n_cols) if n_cols > 1 else [0.0]
    ys = np.linspace(-0.4, 0.4, n_rows) if n_rows > 1 else [0.0]
    
    for y in ys:
        for x in xs:
            points.append((x, y))
            
    points = np.array(points)
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # Fill with random points near center
        extra = np.random.uniform(-0.4, 0.4, size=(N - len(points), 2))
        points = np.vstack([points, extra])
        
    return points

def _physics_simulated_annealing(centers, time_limit=10.0):
    """
    Perform physics-based simulated annealing to maximize the minimum radius.
    We simulate hard disks with repulsive forces.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Initial temperature
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.99
    
    best_r = _radius(centers)
    best_centers = centers.copy()
    
    # Precompute indices for pairwise interactions
    # We will use a simple force model:
    # 1. Repulsion between circles if dist < 2*r_target (or just always repel with strength based on dist)
    # 2. Attraction to center? No, we want to maximize min dist to walls and each other.
    # Standard approach for packing: minimize potential energy where E = sum(1/d^2) for pairs and walls.
    # Or maximize min dist.
    # Let's use a "soft" repulsion that pushes apart overlapping circles.
    # We want to find a configuration where min_dist is maximized.
    
    # Strategy:
    # 1. Pick a target radius r_target (start with current best estimate + epsilon).
    # 2. Apply forces to push circles apart if they are closer than 2*r_target.
    # 3. Apply forces to push circles away from walls if they are closer than r_target.
    # 4. Add random noise (temperature) to escape local minima.
    # 5. If all constraints satisfied, try to increase r_target.
    
    # Let's refine this:
    # Instead of a fixed r_target, let's minimize a potential energy function:
    # E = sum_{i<j} max(0, 2*r - d_ij)^2 + sum_i max(0, r - dist_to_wall_i)^2
    # This is similar to the violation function in the previous code, but we use it with SA.
    
    # However, to "maximize" r, we can use the following loop:
    # 1. Estimate current max r.
    # 2. Set r_target = current_r * (1 + small_increment).
    # 3. Run SA to minimize violations for r_target.
    # 4. If violations ~ 0, accept new r. Else, reduce increment.
    
    # Given time constraints, a simpler and often effective approach for packing is:
    # Directly optimize the "minimum distance" using a force-directed layout where
    # the force is proportional to the gradient of the soft-min, but with large random kicks.
    
    # Let's try a different physics model:
    # Treat circles as repelling each other with a long-range force (1/r^2) and short-range hard core.
    # The equilibrium of this system tends to maximize the minimum distance.
    
    def _compute_forces_and_energy(centers, r_ref):
        """
        Compute forces based on repulsion.
        We use a potential U(r) = k / d^2 for d < d_cut, and 0 otherwise?
        No, to maximize min dist, we want to push the closest pairs apart.
        
        Let's use the gradient of the soft-min objective as the "force".
        Force_i = -grad_i ( -soft_min ) = grad_i (soft_min)
        This pushes points to increase the minimum distance.
        """
        # This is computationally expensive to compute gradients analytically.
        # Let's use a simpler heuristic force:
        # For each pair (i, j), if d_ij < 2*r_ref, push apart.
        # For each point i, if dist_to_wall < r_ref, push in.
        
        forces = np.zeros_like(centers)
        energy = 0.0
        
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    # Random push
                    forces[i] += np.random.normal(0, 0.01, 2)
                    forces[j] -= np.random.normal(0, 0.01, 2)
                    continue
                
                # We want to maximize the minimum distance.
                # A good heuristic: force proportional to 1/d^2 if d < 2*r_ref, else 0.
                # But this only works if we have a reference r_ref.
                # Let's use a "pressure" model.
                
                # Actually, let's just use the violation gradient directly.
                # If dist < 2*r_ref:
                #   violation = (2*r_ref - dist)
                #   grad_violation w.r.t c_i = - (c_i - c_j) / dist
                #   We want to minimize violation, so force = -grad = (c_i - c_j) / dist
                #   Scaled by some factor.
                
                if dist < 2 * r_ref:
                    # Push apart
                    direction = diff / dist
                    strength = (2 * r_ref - dist) / (2 * r_ref) # 0 to 1
                    f = strength * direction
                    forces[i] += f
                    forces[j] -= f
                    energy += strength ** 2
                else:
                    # Weak attraction to prevent floating? No, we want them spread out.
                    # Maybe a weak repulsion to fill space?
                    # Let's add a small constant repulsion to encourage spreading.
                    pass
        
        # Wall forces
        for i in range(N):
            # x < -0.5 + r_ref
            if centers[i, 0] < -0.5 + r_ref:
                forces[i, 0] += ( -0.5 + r_ref - centers[i, 0] )
                energy += ( -0.5 + r_ref - centers[i, 0] ) ** 2
            # x > 0.5 - r_ref
            if centers[i, 0] > 0.5 - r_ref:
                forces[i, 0] += ( 0.5 - r_ref - centers[i, 0] )
                energy += ( 0.5 - r_ref - centers[i, 0] ) ** 2
            # y < -0.5 + r_ref
            if centers[i, 1] < -0.5 + r_ref:
                forces[i, 1] += ( -0.5 + r_ref - centers[i, 1] )
                energy += ( -0.5 + r_ref - centers[i, 1] ) ** 2
            # y > 0.5 - r_ref
            if centers[i, 1] > 0.5 - r_ref:
                forces[i, 1] += ( 0.5 - r_ref - centers[i, 1] )
                energy += ( 0.5 - r_ref - centers[i, 1] ) ** 2
                
        return forces, energy

    # Main SA Loop
    # We will try to increase r_ref gradually.
    r_ref = _radius(centers) * 1.01
    if r_ref < 0.05:
        r_ref = 0.05
    
    step_size = 0.01
    
    while time.time() - start_time < time_limit:
        # Compute forces for current r_ref
        forces, energy = _compute_forces_and_energy(centers, r_ref)
        
        # Update positions
        # Newton-Cauchy: c_new = c + step_size * force + noise
        noise = np.random.normal(0, T, size=(N, 2))
        new_centers = centers + step_size * forces + noise
        
        # Keep inside square (hard constraint)
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
        
        # Evaluate new radius
        new_r = _radius(new_centers)
        
        # Acceptance criterion
        # If new_r > best_r, always accept.
        # Else, accept with probability exp((new_r - best_r) / T)
        # Note: r is the "energy" we want to MAXIMIZE.
        # So standard SA is for minimizing E. Here we maximize R.
        # Let E = -R. Minimize E.
        # Delta E = -Delta R.
        # Prob = exp( -Delta E / T ) = exp( Delta R / T ).
        
        delta_r = new_r - best_r
        if delta_r > 0:
            centers = new_centers
            best_r = new_r
            best_centers = new_centers.copy()
            # Try to increase r_ref slightly if we are making progress
            if energy < 1e-6:
                r_ref *= 1.001
        else:
            # delta_r is negative
            prob = np.exp(delta_r / T)
            if random.random() < prob:
                centers = new_centers
            # If we reject, r_ref stays same or decreases?
            # If we are stuck, maybe decrease r_ref to find a feasible config first?
            # No, we want to maximize. If we can't improve, we are in a local min.
            # The noise helps escape.
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Reheat?
            # If we are stuck for a while, reheat to escape local minima.
            # Simple heuristic: if best_r hasn't changed in last 100 iterations, reheat.
            # We don't track iterations easily here, so let's just reheat periodically.
            if random.random() < 0.01:
                T = 0.05
                step_size = 0.02 # Larger steps when hot
        
        # Adjust step size based on energy?
        # If energy is high, take smaller steps?
        # If energy is low, take larger steps?
        
    return best_centers, best_r

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Grids
    for nr, nc in [(7, 7), (5, 10), (10, 5), (8, 7), (7, 8)]:
        inits.append(_generate_initial_grid(nr, nc))
        
    # 2. Hexagonal
    # Create a hex grid
    spacing = 0.15
    dy = spacing * np.sqrt(3) / 2
    points = []
    for j in range(10):
        for i in range(10):
            x = i * spacing - 0.5
            y = j * dy - 0.5
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    hex_grid = np.array(points)
    # Center and scale
    hex_grid = hex_grid - np.mean(hex_grid, axis=0)
    max_ext = np.max(np.abs(hex_grid))
    if max_ext > 0.45:
        hex_grid = hex_grid * (0.45 / max_ext)
    inits.append(hex_grid[:N])
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Strategy:
    # 1. Run SA on each init for a short time to find a good local optimum.
    # 2. Keep the best one.
    # 3. Run SA on the best one for a longer time to refine.
    
    time_per_init = 10.0
    time_left = DEADLINE - 10 - (time.time() - start_time)
    
    # Phase 1: Quick search on all inits
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_left = DEADLINE - 20 - (time.time() - start_time)
        if time_left < 2:
            break
            
        # Allocate time proportional to number of inits remaining?
        # Just use fixed time per init for simplicity, but cap it.
        t_alloc = min(time_per_init, time_left / (len(inits) - i))
        
        refined_centers, r = _physics_simulated_annealing(init, time_limit=t_alloc)
        r = _radius(refined_centers)
        
        if r > best_r:
            best_r = r
            best_centers = refined_centers.copy()
            
    # Phase 2: Deep refinement on the best result
    if best_centers is not None:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 5:
            # Use a fresh start from the best center, but maybe with a small perturbation
            # to allow SA to explore around it.
            # Or just continue from where it left off?
            # _physics_simulated_annealing modifies centers in place (copy).
            # Let's run it again on the best_centers.
            refined_centers, r = _physics_simulated_annealing(best_centers, time_limit=time_left)
            r = _radius(refined_centers)
            
            if r > best_r:
                best_r = r
                best_centers = refined_centers.copy()
                
    # Phase 3: Final polish with a few random restarts if time allows?
    # Or just return.
    
    if best_centers is None:
        # Fallback
        best_centers = _generate_initial_grid(7, 7)
        
    return best_centers

if __name__ == "__main__":
    # Quick test
    start = time.time()
    centers = construct()
    end = time.time()
    r = _radius(centers)
    print(f"Radius: {r}, Time: {end - start:.2f}s")
