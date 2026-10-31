"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _boundary_violation(pos, radii):
    """Compute squared boundary violation for each circle."""
    bv = np.zeros(N)
    bv += np.maximum(0, radii - pos[:, 0])**2
    bv += np.maximum(0, pos[:, 0] + radii - 1.0)**2
    bv += np.maximum(0, radii - pos[:, 1])**2
    bv += np.maximum(0, pos[:, 1] + radii - 1.0)**2
    return bv

def _pairwise_overlap(pos, radii):
    """Compute overlap violations for all pairs. Returns (violation array, mask)."""
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    dist = np.sqrt(dist_sq)
    viol = np.maximum(0, min_dist - dist)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return viol, mask

def _compute_energy(pos, radii, penalty_weight=200.0):
    """
    Energy = -sum(radii) + penalty_weight * (boundary_viol + overlap_viol)
    """
    bv = _boundary_violation(pos, radii)
    boundary_viol = np.sum(bv)
    
    viol, mask = _pairwise_overlap(pos, radii)
    overlap_viol = np.sum(viol[mask]**2)
    
    total_viol = boundary_viol + overlap_viol
    score = -np.sum(radii) + penalty_weight * total_viol
    return score

def _is_valid(pos, radii, tol=1e-9):
    """Check if configuration is strictly valid."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist = np.sqrt(dist_sq)
    
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
        
    return True

def _enforce_validity(pos, radii):
    """
    Greedily shrink radii to ensure validity.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # 1. Boundary constraints
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    # 2. Overlap constraints - iterate until stable
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-12:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    changed = True
        if not changed:
            break
            
    # Final safety clip
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    return pos, radii

def _force_relaxation(pos, radii, n_steps=500, step_size=0.01, rng=None):
    """
    Force-based relaxation: push overlapping circles apart, walls inward.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    pos = pos.copy()
    radii = radii.copy()
    
    for step in range(n_steps):
        forces = np.zeros((N, 2))
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    dx = rng.uniform(-0.01, 0.01)
                    dy = rng.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = (radii[i] + radii[j]) - dist
                if overlap > 0:
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall forces
        wall_k = 10.0
        for i in range(N):
            r = radii[i]
            if pos[i, 0] < r:
                forces[i, 0] += wall_k * (r - pos[i, 0])
            if pos[i, 0] > 1.0 - r:
                forces[i, 0] -= wall_k * (pos[i, 0] - (1.0 - r))
            if pos[i, 1] < r:
                forces[i, 1] += wall_k * (r - pos[i, 1])
            if pos[i, 1] > 1.0 - r:
                forces[i, 1] -= wall_k * (pos[i, 1] - (1.0 - r))
        
        # Update positions
        pos += step_size * forces
        pos = np.clip(pos, 0.0, 1.0)
        
        # Decaying step size
        step_size *= 0.995
        
    return pos, radii

def _grow_and_relax(pos, radii, n_grow_steps=100, relax_steps=100, rng=None):
    """
    Iteratively grow radii and relax positions to find a better packing.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    pos = pos.copy()
    radii = radii.copy()
    
    # Enforce validity first
    pos, radii = _enforce_validity(pos, radii)
    
    for g in range(n_grow_steps):
        # Try to grow a random circle
        i = rng.randint(N)
        old_r = radii[i]
        grow_amount = 0.002 * (1.0 + old_r)
        new_r = old_r + grow_amount
        
        # Check boundary
        max_r_wall = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if new_r > max_r_wall:
            new_r = max_r_wall
            if new_r <= old_r:
                continue
        
        # Try to grow
        radii[i] = new_r
        
        # Relax to remove overlaps
        pos, radii = _force_relaxation(pos, radii, n_steps=relax_steps, 
                                         step_size=0.005, rng=rng)
        
        # Enforce validity (may shrink some radii)
        pos, radii = _enforce_validity(pos, radii)
        
    return pos, radii

def _simulated_annealing(pos, radii, time_limit, rng):
    """
    Simulated Annealing to maximize sum of radii.
    """
    start_time = time.time()
    T = 0.05
    cooling_rate = 0.9997
    min_T = 1e-7
    
    current_pos = pos.copy()
    current_radii = radii.copy()
    current_energy = _compute_energy(current_pos, current_radii)
    
    best_pos = current_pos.copy()
    best_radii = current_radii.copy()
    best_energy = current_energy
    
    step_size_pos = 0.03
    step_size_rad = 0.03
    
    while time.time() - start_time < time_limit and T > min_T:
        new_pos = current_pos.copy()
        new_radii = current_radii.copy()
        
        n_perturb = 1 if rng.random() < 0.6 else 2
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        for idx in indices:
            new_pos[idx, 0] += rng.normal(0, step_size_pos)
            new_pos[idx, 1] += rng.normal(0, step_size_pos)
            new_radii[idx] += rng.normal(0, step_size_rad)
            
            new_pos[idx, 0] = np.clip(new_pos[idx, 0], 0, 1)
            new_pos[idx, 1] = np.clip(new_pos[idx, 1], 0, 1)
            new_radii[idx] = np.clip(new_radii[idx], 0, 0.4)
            
        new_energy = _compute_energy(new_pos, new_radii)
        
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_pos = new_pos
            current_radii = new_radii
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                best_radii = current_radii.copy()
                
        T *= cooling_rate
        step_size_pos *= 0.9998
        step_size_rad *= 0.9998
        
    return best_pos, best_radii

def _polish_configuration(pos, radii, time_limit=5.0):
    """
    Local search to maximize sum of radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    start_time = time.time()
    
    pos, radii = _enforce_validity(pos, radii)
    
    rng = np.random.RandomState(42)
    
    for _ in range(8000):
        if time.time() - start_time > time_limit:
            break
            
        i = rng.randint(N)
        old_r = radii[i]
        step = 0.003 * (1.0 + old_r)
        new_r = old_r + step
        
        max_r_wall = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if new_r > max_r_wall:
            new_r = max_r_wall
            if new_r <= old_r:
                continue
        
        valid = True
        for j in range(N):
            if i == j:
                continue
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < new_r + radii[j]:
                valid = False
                break
        
        if valid:
            radii[i] = new_r
        else:
            # Try to move circle i
            fx, fy = 0.0, 0.0
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    dx = rng.uniform(-0.01, 0.01)
                    dy = rng.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = (new_r + radii[j]) - dist
                if overlap > 0:
                    fx += (dx / dist) * overlap
                    fy += (dy / dist) * overlap
            
            if pos[i,0] - new_r < 0:
                fx += (new_r - pos[i,0])
            if pos[i,0] + new_r > 1:
                fx -= (pos[i,0] + new_r - 1)
            if pos[i,1] - new_r < 0:
                fy += (new_r - pos[i,1])
            if pos[i,1] + new_r > 1:
                fy -= (pos[i,1] + new_r - 1)
            
            move_step = 0.015
            pos[i,0] += fx * move_step
            pos[i,1] += fy * move_step
            pos[i,0] = np.clip(pos[i,0], 0.0, 1.0)
            pos[i,1] = np.clip(pos[i,1], 0.0, 1.0)
            
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            
            radii[i] = max(0.0, max_r)
            
    pos, radii = _enforce_validity(pos, radii)
    
    return pos, radii

def _make_grid_init():
    """Create a grid-based initial configuration."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.09
                k += 1
    if k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Multi-start strategy
    n_starts = 12
    
    for s in range(n_starts):
        if time.time() > deadline - 8:
            break
            
        # Initialize
        if s == 0:
            pos, radii = _make_grid_init()
        elif s == 1:
            # Hexagonal-like packing
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            rows = 5
            cols = 5
            for i in range(rows):
                for j in range(cols):
                    if k >= N:
                        break
                    x = 0.1 + j * 0.2 + (0.1 if i % 2 == 1 else 0)
                    y = 0.1 + i * 0.2
                    if x > 0.9 or y > 0.9:
                        continue
                    pos[k, 0] = x
                    pos[k, 1] = y
                    radii[k] = 0.09
                    k += 1
            while k < N:
                pos[k] = rng.uniform(0.2, 0.8, 2)
                radii[k] = 0.05
                k += 1
        elif s == 2:
            # Concentric rings
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            pos[0] = [0.5, 0.5]
            radii[0] = 0.15
            k = 1
            # Ring 1
            for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                if k >= N:
                    break
                pos[k, 0] = 0.5 + 0.25 * np.cos(angle)
                pos[k, 1] = 0.5 + 0.25 * np.sin(angle)
                radii[k] = 0.07
                k += 1
            # Ring 2
            while k < N:
                angle = (k - 7) * 2 * np.pi / 12
                pos[k, 0] = 0.5 + 0.38 * np.cos(angle)
                pos[k, 1] = 0.5 + 0.38 * np.sin(angle)
                radii[k] = 0.05
                k += 1
        else:
            # Random init
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = rng.uniform(0.04, 0.12, N)
            pos, radii = _enforce_validity(pos, radii)
        
        # Run different optimization strategies based on start
        if s < 3:
            # Structured init: grow and relax, then SA
            pos_gr, rad_gr = _grow_and_relax(pos, radii, n_grow_steps=80, 
                                               relax_steps=80, rng=rng)
            
            sa_time = min(4.0, deadline - time.time() - 6)
            if sa_time > 1:
                pos_sa, rad_sa = _simulated_annealing(pos_gr, rad_gr, sa_time, rng)
            else:
                pos_sa, rad_sa = pos_gr, rad_gr
            
            polish_time = min(3.0, deadline - time.time() - 3)
            if polish_time > 0.5:
                pos_pol, rad_pol = _polish_configuration(pos_sa, rad_sa, polish_time)
            else:
                pos_pol, rad_pol = pos_sa, rad_sa
        else:
            # Random init: SA with polish
            sa_time = min(4.5, deadline - time.time() - 5)
            if sa_time > 1:
                pos_sa, rad_sa = _simulated_annealing(pos, radii, sa_time, rng)
            else:
                pos_sa, rad_sa = pos, radii
            
            polish_time = min(3.0, deadline - time.time() - 2)
            if polish_time > 0.5:
                pos_pol, rad_pol = _polish_configuration(pos_sa, rad_sa, polish_time)
            else:
                pos_pol, rad_pol = pos_sa, rad_sa
        
        # Validate
        if _is_valid(pos_pol, rad_pol):
            score = np.sum(rad_pol)
            if score > best_score:
                best_score = score
                best_pos = pos_pol
                best_radii = rad_pol
                
        # Early stop if very good
        if best_score > 2.6:
            break

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid_init()
        pos, radii = _enforce_validity(pos, radii)
        pos, radii = _polish_configuration(pos, radii, time_limit=5.0)
        best_pos, best_radii = pos, radii

    # Final strict validation
    best_pos, best_radii = _enforce_validity(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
