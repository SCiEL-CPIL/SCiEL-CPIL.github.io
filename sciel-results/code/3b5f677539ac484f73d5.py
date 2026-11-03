"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute maximum possible radii for each circle given positions.
    Uses iterative relaxation to ensure strict non-overlap and boundary containment.
    """
    radii = np.minimum(
        np.minimum(pos[:, 0], pos[:, 1]),
        np.minimum(1.0 - pos[:, 0], 1.0 - pos[:, 1])
    )
    radii = np.maximum(radii, 0.0)
    
    for _ in range(80):
        changed = False
        for i in range(n):
            r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            if radii[i] > r_bound:
                radii[i] = r_bound
                changed = True
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    radii[i] = max(0.0, radii[i] - excess / 2.0)
                    radii[j] = max(0.0, radii[j] - excess / 2.0)
                    changed = True
        if not changed:
            break
    
    return radii

def _physics_relax(pos_init, n, deadline, pressure_coeff=1.0, dt=0.001, max_steps=5000):
    """
    Physics-based bubble relaxation: circles modeled as pressurized bubbles.
    Internal pressure proportional to r^2 pushes circles apart.
    Repulsive contact forces prevent overlap.
    Boundary constraints keep circles inside [0,1]^2.
    """
    pos = pos_init.copy().astype(float)
    radii = _compute_radii(pos, n)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Physics parameters
    k_contact = 50.0  # Contact stiffness
    k_boundary = 100.0  # Boundary stiffness
    damping = 0.95  # Velocity damping
    
    velocities = np.zeros_like(pos)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Contact forces (repulsive when overlapping)
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                min_dist = radii[i] + radii[j]
                
                if d > 1e-10:
                    if d < min_dist:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - d
                        force_mag = k_contact * overlap
                        direction = d_vec / d
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
                    else:
                        # Weak attractive force to encourage close packing
                        # This helps circles find dense configurations
                        gap = d - min_dist
                        if gap < 0.1:
                            force_mag = -0.1 * gap  # Small attraction
                            direction = d_vec / d
                            forces[i] += force_mag * direction
                            forces[j] -= force_mag * direction
        
        # Boundary forces
        for i in range(n):
            # Left boundary
            if pos[i, 0] < radii[i]:
                forces[i, 0] += k_boundary * (radii[i] - pos[i, 0])
            # Right boundary
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= k_boundary * (pos[i, 0] - (1.0 - radii[i]))
            # Bottom boundary
            if pos[i, 1] < radii[i]:
                forces[i, 1] += k_boundary * (radii[i] - pos[i, 1])
            # Top boundary
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= k_boundary * (pos[i, 1] - (1.0 - radii[i]))
        
        # Pressure forces: each circle wants to expand
        # Pressure gradient pushes against neighbors and boundaries
        for i in range(n):
            # Compute net pressure force direction (toward center of mass of neighbors)
            # Simplified: pressure tries to increase radius, which means pushing outward
            # We model this as a force that resists compression
            for j in range(n):
                if i == j:
                    continue
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d > 1e-10:
                    # Pressure contribution: proportional to r_i^2
                    pressure_force = pressure_coeff * radii[i] ** 2 / (d + 0.01)
                    direction = d_vec / d
                    forces[i] += pressure_force * direction
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        pos = pos + velocities * dt
        
        # Clamp positions to valid range (with some margin)
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
        
        # Recompute radii after position update
        radii = _compute_radii(pos, n)
        
        # Track best configuration
        current_score = np.sum(radii)
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
            best_radii = radii.copy()
        
        # Early stopping if converged
        if step > 100 and np.max(np.abs(velocities)) < 1e-6:
            break
    
    return best_pos, best_radii, best_score

def _local_refine(pos_init, radii_init, n, deadline):
    """
    Local refinement using discrete best-fit search.
    Tries to increase each circle's radius by finding a better position.
    """
    pos = pos_init.copy()
    radii = radii_init.copy()
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Try to improve each circle's radius
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try to increase radius of circle i
            current_r = radii[i]
            max_possible = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            
            if current_r >= max_possible - 1e-6:
                continue
            
            # Binary search for maximum radius that fits
            lo = current_r
            hi = max_possible
            best_r = current_r
            best_x = pos[i, 0]
            best_y = pos[i, 1]
            
            for _ in range(10):
                mid = (lo + hi) / 2.0
                
                # Check if circle i with radius mid can be placed
                # without overlapping others
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < mid + radii[j]:
                        valid = False
                        break
                
                if valid:
                    best_r = mid
                    lo = mid
                else:
                    hi = mid
            
            if best_r > current_r + 1e-6:
                radii[i] = best_r
                improved = True
        
        if not improved:
            break
        
        # Recompute all radii to ensure consistency
        radii = _compute_radii(pos, n)
        
        current_score = np.sum(radii)
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    return best_pos, best_radii, best_score

def _initial_hexagonal(n):
    """Create hexagonal packing initial configuration."""
    pos = np.zeros((n, 2))
    
    # Hexagonal arrangement
    row_height = 0.18
    col_width = 0.18
    idx = 0
    
    for row in range(6):
        y = (row + 0.5) * row_height
        offset = (row % 2) * col_width / 2.0
        for col in range(6):
            if idx >= n:
                break
            x = (col + 0.5) * col_width + offset
            pos[idx] = [x, y]
            idx += 1
    
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    
    return pos

def _initial_grid(n):
    """Create initial grid configuration."""
    pos = np.zeros((n, 2))
    rows = 5
    cols = 6
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            pos[idx] = [x, y]
            idx += 1
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    return pos

def _initial_random(n, seed=42):
    """Create random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    min_sep = 0.08
    for i in range(n):
        attempts = 0
        while attempts < 100:
            p = rng.uniform(0.05, 0.95, size=2)
            ok = True
            for j in range(i):
                if np.linalg.norm(p - pos[j]) < min_sep:
                    ok = False
                    break
            if ok:
                pos[i] = p
                break
            attempts += 1
        if attempts >= 100:
            pos[i] = rng.uniform(0.05, 0.95, size=2)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initial configurations
    configs = []
    
    # Hexagonal (best known structure for dense packing)
    pos_hex = _initial_hexagonal(N)
    configs.append(pos_hex)
    
    # Grid
    pos_grid = _initial_grid(N)
    configs.append(pos_grid)
    
    # Random starts with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        pos_rand = _initial_random(N, seed=seed)
        configs.append(pos_rand)
    
    # Process each config with physics relaxation + local refinement
    for idx, pos_init in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        try:
            # Phase 1: Physics-based relaxation
            pos_phys, radii_phys, score_phys = _physics_relax(
                pos_init, N, deadline - 5, 
                pressure_coeff=1.0, dt=0.001, max_steps=3000
            )
            
            # Phase 2: Local refinement
            pos_final, radii_final, score_final = _local_refine(
                pos_phys, radii_phys, N, deadline - 2
            )
            
            if score_final > best_score:
                best_score = score_final
                result = np.zeros((N, 3))
                result[:, :2] = pos_final
                result[:, 2] = radii_final
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
    
    return best_result

# EVOLVE-BLOCK-END
