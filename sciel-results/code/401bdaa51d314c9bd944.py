"""Circle packing in unit square using electrostatic equilibrium approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_max_radii(centers, margin=0.0):
        """
        Given circle centers, compute maximum non-overlapping radii.
        Uses iterative relaxation to find consistent radii.
        """
        centers = np.asarray(centers)
        n = len(centers)
        
        # Initialize radii based on boundary distance
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Iteratively enforce pairwise constraints
        for _ in range(100):
            updated = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Constraint: r_i + r_j <= dist
                    # If violated, reduce the larger radius
                    if radii[i] + radii[j] > dist:
                        excess = radii[i] + radii[j] - dist
                        # Reduce both proportionally
                        total_r = radii[i] + radii[j]
                        if total_r > 1e-12:
                            scale_i = radii[i] / total_r
                            scale_j = radii[j] / total_r
                            radii[i] -= excess * scale_i
                            radii[j] -= excess * scale_j
                        else:
                            radii[i] = dist / 2
                            radii[j] = dist / 2
                        updated = True
            if not updated:
                break
        
        # Apply margin
        if margin > 0:
            radii = np.maximum(0.0, radii - margin)
        
        return radii

    def electrostatic_forces(positions, radii):
        """
        Compute electrostatic forces on each circle.
        Each circle has charge proportional to its radius.
        Repulsive Coulomb force between pairs.
        Attractive harmonic potential toward center.
        Boundary repulsion when approaching edges.
        """
        n = len(positions)
        forces = np.zeros((n, 2))
        
        # Pairwise repulsive forces (Coulomb-like)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-10:
                    # Random small perturbation
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                # Charge proportional to radius
                q_i = radii[i] + 0.01  # Small offset to avoid zero charge
                q_j = radii[j] + 0.01
                
                # Coulomb force: F = k * q_i * q_j / dist^2, direction along separation
                # We want repulsion, so force on i is away from j
                k = 1.0
                F = k * q_i * q_j / (dist * dist)
                
                # Unit vector from j to i
                ux = dx / dist
                uy = dy / dist
                
                forces[i, 0] += F * ux
                forces[i, 1] += F * uy
                forces[j, 0] -= F * ux
                forces[j, 1] -= F * uy
        
        # Attractive harmonic potential toward center (0.5, 0.5)
        k_center = 0.5
        for i in range(n):
            forces[i, 0] += k_center * (0.5 - positions[i, 0])
            forces[i, 1] += k_center * (0.5 - positions[i, 1])
        
        # Boundary repulsion
        k_boundary = 10.0
        for i in range(n):
            r = radii[i]
            # Left boundary
            if positions[i, 0] - r < 0.1:
                forces[i, 0] += k_boundary * (0.1 - (positions[i, 0] - r))
            # Right boundary
            if 1.0 - positions[i, 0] - r < 0.1:
                forces[i, 0] -= k_boundary * (0.1 - (1.0 - positions[i, 0] - r))
            # Bottom boundary
            if positions[i, 1] - r < 0.1:
                forces[i, 1] += k_boundary * (0.1 - (positions[i, 1] - r))
            # Top boundary
            if 1.0 - positions[i, 1] - r < 0.1:
                forces[i, 1] -= k_boundary * (0.1 - (1.0 - positions[i, 1] - r))
        
        return forces

    def symplectic_integrate(positions, radii, velocities, dt=0.005, damping=0.95):
        """
        Symplectic integrator with adaptive velocity damping.
        Returns updated positions and velocities.
        """
        n = len(positions)
        new_positions = positions.copy()
        new_velocities = velocities.copy()
        
        for i in range(n):
            # Compute forces at current position
            forces = electrostatic_forces(positions, radii)
            
            # Update velocity with force (acceleration = F/m, assume m=1)
            new_velocities[i] = velocities[i] + dt * forces[i]
            
            # Apply damping
            new_velocities[i] *= damping
            
            # Update position with new velocity
            new_positions[i] = positions[i] + dt * new_velocities[i]
            
            # Clip to valid range
            r = radii[i]
            new_positions[i, 0] = np.clip(new_positions[i, 0], r, 1.0 - r)
            new_positions[i, 1] = np.clip(new_positions[i, 1], r, 1.0 - r)
        
        return new_positions, new_velocities

    def electrostatic_equilibrium(centers_init, max_iters=5000, dt=0.005, damping=0.95):
        """
        Run electrostatic simulation until equilibrium.
        Returns final positions.
        """
        positions = centers_init.copy()
        velocities = np.zeros_like(positions)
        
        # Initial radii
        radii = compute_max_radii(positions, margin=0.01)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update positions using symplectic integrator
            positions, velocities = symplectic_integrate(positions, radii, velocities, dt, damping)
            
            # Recompute radii based on new positions
            radii = compute_max_radii(positions, margin=0.005)
            
            # Check for convergence: max force magnitude
            forces = electrostatic_forces(positions, radii)
            max_force = np.max(np.linalg.norm(forces, axis=1))
            
            if max_force < 1e-4:
                break
        
        return positions

    def gradient_descent_refine(positions, radii, max_iters=2000, lr=0.01):
        """
        Gradient descent refinement to maximize sum of radii.
        """
        best_positions = positions.copy()
        best_radii = radii.copy()
        best_score = np.sum(radii)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute forces (gradient of negative objective)
            forces = electrostatic_forces(positions, radii)
            
            # Update positions
            positions = positions - lr * forces
            
            # Clip to valid range
            for i in range(N):
                r = radii[i]
                positions[i, 0] = np.clip(positions[i, 0], r, 1.0 - r)
                positions[i, 1] = np.clip(positions[i, 1], r, 1.0 - r)
            
            # Recompute radii
            radii = compute_max_radii(positions, margin=0.001)
            
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_radii = radii.copy()
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.95
        
        return best_positions, best_radii

    def local_search_refine(positions, radii, max_iters=3000, step_size=0.02, seed=0):
        """
        Best-fit local search refinement.
        """
        rng = np.random.RandomState(seed)
        best_positions = positions.copy()
        best_radii = radii.copy()
        best_score = np.sum(radii)
        
        current_positions = positions.copy()
        current_radii = radii.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random circle
            i = rng.randint(N)
            
            # Try to move center
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            
            new_positions = current_positions.copy()
            new_positions[i, 0] = np.clip(new_positions[i, 0] + dx, 0.0, 1.0)
            new_positions[i, 1] = np.clip(new_positions[i, 1] + dy, 0.0, 1.0)
            
            # Recompute radii
            new_radii = compute_max_radii(new_positions, margin=0.001)
            
            # Check validity
            valid = True
            for j in range(N):
                r = new_radii[j]
                if new_positions[j, 0] - r < -1e-8 or new_positions[j, 0] + r > 1 + 1e-8:
                    valid = False
                    break
                if new_positions[j, 1] - r < -1e-8 or new_positions[j, 1] + r > 1 + 1e-8:
                    valid = False
                    break
            if valid:
                for j in range(N):
                    for k in range(j+1, N):
                        dx2 = new_positions[j, 0] - new_positions[k, 0]
                        dy2 = new_positions[j, 1] - new_positions[k, 1]
                        dist = np.sqrt(dx2*dx2 + dy2*dy2)
                        if dist < new_radii[j] + new_radii[k] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                new_score = np.sum(new_radii)
                if new_score > current_score + 1e-10:
                    current_positions = new_positions
                    current_radii = new_radii
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_positions = new_positions.copy()
                        best_radii = new_radii.copy()
        
        return best_positions, best_radii

    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_init(seed=0):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    def hexagonal_init():
        """Hexagonal packing initialization."""
        centers = []
        # Hexagonal lattice
        dx = 0.2
        dy = dx * np.sqrt(3) / 2
        for row in range(5):
            for col in range(6):
                if len(centers) >= N:
                    break
                x = 0.1 + col * dx + (row % 2) * dx/2
                y = 0.1 + row * dy
                if x < 0.9 and y < 0.9:
                    centers.append([x, y])
        # Fill remaining with random
        rng = np.random.RandomState(42)
        while len(centers) < N:
            centers.append(rng.uniform(0.1, 0.9, size=2))
        return np.array(centers[:N])

    # Main strategy: Multiple initializations + Electrostatic equilibrium + Refinement
    best_result = None
    best_score = -np.inf
    
    inits = [
        grid_init(),
        hexagonal_init(),
        random_init(1),
        random_init(7),
        random_init(13),
        random_init(23),
        random_init(37),
        random_init(53),
        random_init(71),
        random_init(89),
    ]
    
    for idx, centers in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Step 1: Electrostatic equilibrium
        eq_positions = electrostatic_equilibrium(centers, max_iters=3000, dt=0.005, damping=0.95)
        
        # Compute radii
        radii = compute_max_radii(eq_positions, margin=0.001)
        
        # Check validity and score
        if is_valid(np.column_stack([eq_positions, radii]), tol=1e-6):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([eq_positions, radii]).copy()
        
        # Step 2: Gradient descent refinement
        gd_positions, gd_radii = gradient_descent_refine(eq_positions, radii, max_iters=1000, lr=0.01)
        
        if is_valid(np.column_stack([gd_positions, gd_radii]), tol=1e-6):
            score = np.sum(gd_radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([gd_positions, gd_radii]).copy()
        
        # Step 3: Local search refinement
        ls_positions, ls_radii = local_search_refine(gd_positions, gd_radii, max_iters=2000, step_size=0.015, seed=idx)
        
        if is_valid(np.column_stack([ls_positions, ls_radii]), tol=1e-6):
            score = np.sum(ls_radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([ls_positions, ls_radii]).copy()
    
    # Final refinement if time allows
    if best_result is not None and time.time() < deadline - 10:
        final_positions = best_result[:, :2]
        final_radii = best_result[:, 2]
        ls_positions, ls_radii = local_search_refine(final_positions, final_radii, max_iters=3000, step_size=0.01, seed=999)
        
        if is_valid(np.column_stack([ls_positions, ls_radii]), tol=1e-6):
            score = np.sum(ls_radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([ls_positions, ls_radii]).copy()
    
    if best_result is None:
        # Fallback
        centers = grid_init()
        radii = compute_max_radii(centers, margin=0.001)
        best_result = np.column_stack([centers, radii])
    
    return best_result

# EVOLVE-BLOCK-END
