"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _hex_layout(n, jitter=0.0):
    """Hexagonal close-packed layout in unit square."""
    # Use a hex grid pattern
    positions = []
    # Try different spacings
    for cols in range(3, 8):
        rows = int(np.ceil(n / cols))
        if cols * rows < n:
            continue
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        # Hex offset
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= n:
                    break
                x = (col + 0.5) * spacing_x
                y = (row + 0.5) * spacing_y
                if row % 2 == 1:
                    x += spacing_x * 0.5
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                positions.append([x, y])
        if len(positions) >= n:
            break
    positions = positions[:n]
    if jitter > 0:
        positions = np.array(positions) + np.random.uniform(-jitter, jitter, size=np.array(positions).shape)
        positions = np.clip(positions, 0.01, 0.99)
    else:
        positions = np.array(positions)
    return positions


def _random_layout(n, seed):
    """Random layout with minimum separation."""
    rng = np.random.RandomState(seed)
    positions = []
    min_dist = 0.08
    attempts = 0
    while len(positions) < n and attempts < 5000:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = True
        for px, py in positions:
            if (x - px)**2 + (y - py)**2 < min_dist**2:
                ok = False
                break
        if ok:
            positions.append([x, y])
        attempts += 1
    # If not enough, fill remaining randomly
    while len(positions) < n:
        positions.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
    return np.array(positions[:n])


def _grid_layout(n, seed=0):
    """Grid layout with slight perturbation."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    positions = np.array(positions[:n])
    if seed > 0:
        rng = np.random.RandomState(seed)
        positions += rng.uniform(-0.02, 0.02, size=positions.shape)
        positions = np.clip(positions, 0.01, 0.99)
    return positions


def _vectorized_penalty(params, n, penalty_weight=1000.0):
    """
    Vectorized objective: minimize -sum(radii) + penalties.
    params: (3n,) array [x0,y0,r0, x1,y1,r1, ...]
    """
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Boundary violations (vectorized)
    # x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    viol_x_min = np.maximum(0, radii - positions[:, 0])
    viol_y_min = np.maximum(0, radii - positions[:, 1])
    viol_x_max = np.maximum(0, positions[:, 0] + radii - 1.0)
    viol_y_max = np.maximum(0, positions[:, 1] + radii - 1.0)
    boundary_viol = np.sum(viol_x_min**2 + viol_y_min**2 + viol_x_max**2 + viol_y_max**2)
    
    # Overlap violations (vectorized using broadcasting)
    # dist[i,j] for all pairs
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 2)
    dists = np.sqrt(np.sum(diff**2, axis=2))  # (n, n)
    min_dists = radii[:, np.newaxis] + radii[np.newaxis, :]  # (n, n)
    
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    overlap_viol = np.sum(np.maximum(0, min_dists[iu] - dists[iu])**2)
    
    return obj + penalty_weight * (boundary_viol + overlap_viol)


def _vectorized_gradient(params, n, penalty_weight=1000.0, eps=1e-5):
    """Numerical gradient for the penalty objective."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (_vectorized_penalty(params_plus, n, penalty_weight) - 
                   _vectorized_penalty(params_minus, n, penalty_weight)) / (2 * eps)
    return grad


def _physics_optimize(positions, n, iterations=2000, dt=0.001, growth_rate=0.0001):
    """
    Physics-based optimization:
    - Circles repel each other (spring force when overlapping)
    - Circles repel from walls
    - Radii grow slowly
    - Damping to prevent oscillation
    """
    positions = positions.copy().astype(float)
    radii = np.full(n, 0.01)  # Start small
    
    best_sum = 0.0
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    for it in range(iterations):
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force_mag = overlap * 50.0
                    fx = (dx / dist) * force_mag
                    fy = (dy / dist) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist <= 1e-10:
                    # Random kick
                    forces[i, 0] += np.random.uniform(-0.01, 0.01)
                    forces[i, 1] += np.random.uniform(-0.01, 0.01)
                    forces[j, 0] += np.random.uniform(-0.01, 0.01)
                    forces[j, 1] += np.random.uniform(-0.01, 0.01)
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if positions[i, 0] - radii[i] < 0:
                forces[i, 0] += (radii[i] - positions[i, 0]) * 50.0
            # Right wall
            if positions[i, 0] + radii[i] > 1:
                forces[i, 0] -= (positions[i, 0] + radii[i] - 1.0) * 50.0
            # Bottom wall
            if positions[i, 1] - radii[i] < 0:
                forces[i, 1] += (radii[i] - positions[i, 1]) * 50.0
            # Top wall
            if positions[i, 1] + radii[i] > 1:
                forces[i, 1] -= (positions[i, 1] + radii[i] - 1.0) * 50.0
        
        # Update positions with damping
        positions += dt * forces
        positions *= 0.95  # Damping
        
        # Keep within bounds
        positions = np.clip(positions, 0.0, 1.0)
        
        # Grow radii
        radii += growth_rate
        # Cap radii by boundary
        for i in range(n):
            max_r = min(positions[i, 0], positions[i, 1], 1 - positions[i, 0], 1 - positions[i, 1])
            radii[i] = min(radii[i], max(0.001, max_r))
        
        # Track best
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_positions = positions.copy()
            best_radii = radii.copy()
    
    return best_positions, best_radii, best_sum


def _refine_with_scipy(positions, radii, n, maxiter=500, penalty_weight=1000.0):
    """Use scipy L-BFGS-B to refine the configuration."""
    params = np.zeros(3 * n)
    params[:2*n] = positions.flatten()
    params[2*n:] = radii
    
    bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)] * n
    
    result = minimize(
        _vectorized_penalty,
        params,
        args=(n, penalty_weight),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    
    return result.x


def _fix_and_validate(params, n, tol=1e-7):
    """Fix minor violations and check validity."""
    params = np.array(params, dtype=float)
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Fix boundary violations
    for i in range(n):
        max_r = min(positions[i, 0], positions[i, 1], 1 - positions[i, 0], 1 - positions[i, 1])
        radii[i] = min(radii[i], max(0.0, max_r))
    
    # Fix overlaps iteratively
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - tol:
                    excess = min_dist - dist
                    # Reduce both radii
                    radii[i] -= excess / 2
                    radii[j] -= excess / 2
                    # Also nudge positions apart
                    if dist > 1e-10:
                        push = excess * 0.5
                        positions[i, 0] += (dx / dist) * push * 0.1
                        positions[i, 1] += (dy / dist) * push * 0.1
                        positions[j, 0] -= (dx / dist) * push * 0.1
                        positions[j, 1] -= (dy / dist) * push * 0.1
                    changed = True
        if not changed:
            break
    
    # Re-fix boundaries after position nudges
    for i in range(n):
        positions[i, 0] = np.clip(positions[i, 0], 0, 1)
        positions[i, 1] = np.clip(positions[i, 1], 0, 1)
        max_r = min(positions[i, 0], positions[i, 1], 1 - positions[i, 0], 1 - positions[i, 1])
        radii[i] = min(radii[i], max(0.0, max_r))
    
    radii = np.maximum(radii, 0.0)
    
    # Validate
    valid = True
    for i in range(n):
        if positions[i, 0] - radii[i] < -tol or positions[i, 1] - radii[i] < -tol:
            valid = False
        if positions[i, 0] + radii[i] > 1 + tol or positions[i, 1] + radii[i] > 1 + tol:
            valid = False
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                valid = False
    
    result = np.zeros(3 * n)
    result[:2*n] = positions.flatten()
    result[2*n:] = radii
    
    return result, valid


def _local_search(params, n, iterations=1000, step_size=0.005):
    """Simple local search: try small random perturbations."""
    best_params = params.copy()
    best_score = _vectorized_penalty(best_params, n, 1.0)  # Just sum of radii
    
    for it in range(iterations):
        trial = best_params.copy()
        # Perturb a few random elements
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(len(trial), num_perturb, replace=False)
        for idx in indices:
            trial[idx] += np.random.uniform(-step_size, step_size)
        # Clip to bounds
        trial[:2*n] = np.clip(trial[:2*n], 0, 1)
        trial[2*n:] = np.clip(trial[2*n:], 0.001, 0.5)
        
        score = _vectorized_penalty(trial, n, 1.0)
        if score < best_score:
            best_params = trial
            best_score = score
    
    return best_params


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_result = None
    best_score = -np.inf
    
    # Generate diverse initial layouts
    layouts = []
    
    # Hexagonal layouts with different jitter
    for jitter in [0.0, 0.01, 0.02, 0.03, 0.05]:
        layouts.append(_hex_layout(N, jitter=jitter))
    
    # Grid layouts
    for seed in [0, 1, 2, 3]:
        layouts.append(_grid_layout(N, seed=seed))
    
    # Random layouts
    for seed in range(10):
        layouts.append(_random_layout(N, seed=100 + seed))
    
    # Staggered grid
    side = 5
    for offset in [0.0, 0.25, 0.5]:
        positions = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5 + offset) / side
                y = (j + 0.5) / side
                if x > 1.0:
                    x = 1.0 - x
                if x < 0:
                    x = -x
                positions.append([x, y])
                count += 1
        if len(positions) >= N:
            layouts.append(np.array(positions[:N]))
    
    # Shuffle layouts for diversity
    np.random.seed(42)
    layout_indices = np.random.permutation(len(layouts))
    
    n_processed = 0
    for layout_idx in layout_indices:
        if time.time() > deadline - 10.0:
            break
        
        positions = layouts[layout_idx].copy()
        n_processed += 1
        
        # Phase 1: Physics-based optimization
        try:
            phys_positions, phys_radii, phys_sum = _physics_optimize(
                positions, N, 
                iterations=1500, 
                dt=0.002, 
                growth_rate=0.00008
            )
            
            # Phase 2: Scipy refinement
            if time.time() < deadline - 15.0:
                params = _refine_with_scipy(phys_positions, phys_radii, N, maxiter=300)
            else:
                params = np.zeros(3 * N)
                params[:2*N] = phys_positions.flatten()
                params[2*N:] = phys_radii
            
            # Phase 3: Local search
            if time.time() < deadline - 10.0:
                params = _local_search(params, N, iterations=500, step_size=0.003)
            
            # Fix and validate
            fixed_params, valid = _fix_and_validate(params, N)
            
            if valid:
                score = np.sum(fixed_params[2*N:])
                if score > best_score:
                    best_score = score
                    best_result = fixed_params.reshape(N, 3)
                    
        except Exception:
            continue
    
    # If we have time, do more random restarts with physics
    while time.time() < deadline - 15.0:
        seed = int(time.time() * 1000) % 100000
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.05, 0.95, size=(N, 2))
        
        try:
            phys_positions, phys_radii, phys_sum = _physics_optimize(
                positions, N,
                iterations=1000,
                dt=0.002,
                growth_rate=0.0001
            )
            
            params = np.zeros(3 * N)
            params[:2*N] = phys_positions.flatten()
            params[2*N:] = phys_radii
            
            if time.time() < deadline - 10.0:
                params = _refine_with_scipy(phys_positions, phys_radii, N, maxiter=200)
            
            fixed_params, valid = _fix_and_validate(params, N)
            
            if valid:
                score = np.sum(fixed_params[2*N:])
                if score > best_score:
                    best_score = score
                    best_result = fixed_params.reshape(N, 3)
                    
        except Exception:
            continue
    
    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = min(x, y, 1-x, 1-y) * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
