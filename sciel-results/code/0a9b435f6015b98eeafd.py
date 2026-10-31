import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 26
DEADLINE = 115.0  # Safety margin before 120s limit


def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Vectorized penalty objective for L-BFGS-B
    def penalty_objective(x, penalty_weight=100.0):
        pos = x[:2 * N].reshape(N, 2)
        radii = np.maximum(x[2 * N:], 1e-6)
        
        # Boundary violations
        bv_left = np.maximum(0, radii - pos[:, 0])
        bv_right = np.maximum(0, pos[:, 0] + radii - 1.0)
        bv_bottom = np.maximum(0, radii - pos[:, 1])
        bv_top = np.maximum(0, pos[:, 1] + radii - 1.0)
        boundary_pen = np.sum(bv_left**2 + bv_right**2 + bv_bottom**2 + bv_top**2)
        
        # Overlap violations
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol = np.maximum(0, min_dist_sq[mask] - dist_sq[mask])
        overlap_pen = np.sum(viol**2)
        
        return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)
    
    def _is_valid(pos, radii, tol=1e-9):
        """Strict validity check."""
        if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
            return False
        if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
            return False
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
            return False
        return True
    
    def _post_process(pos, radii):
        """Ensure strict validity by minimally adjusting radii/positions."""
        pos = pos.copy()
        radii = radii.copy()
        
        for _ in range(200):
            changed = False
            for i in range(N):
                max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                if radii[i] > max_r_bound:
                    radii[i] = max_r_bound * (1.0 - 1e-10)
                    changed = True
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[i] + radii[j]:
                        total_r = radii[i] + radii[j]
                        if total_r > 0:
                            scale = dist / total_r * (1.0 - 1e-10)
                            radii[i] *= scale
                            radii[j] *= scale
                            changed = True
            if not changed:
                break
        
        if not _is_valid(pos, radii, tol=1e-9):
            for i in range(N):
                radii[i] *= (1.0 - 1e-8)
        
        return pos, radii
    
    def _get_contact_graph(pos, radii, tol=0.01):
        """Identify pairs of circles in contact or near-contact."""
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        contacts = np.argwhere(mask & (dist <= min_dist + tol))
        return contacts
    
    def _best_fit_init(seed=0):
        """Greedy Best-Fit initialization."""
        rng = np.random.RandomState(seed)
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        
        pos[0] = [0.5, 0.5]
        radii[0] = 0.25

        for i in range(1, N):
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            found = False
            
            for _ in range(10000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
                max_r_circ = max_r_wall
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
                    found = True
            
            if found and best_r > 0.01:
                pos[i] = [best_x, best_y]
                radii[i] = best_r * 0.98 
            else:
                for _ in range(1000):
                    x = rng.uniform(0.05, 0.95)
                    y = rng.uniform(0.05, 0.95)
                    min_dist = min(x, 1-x, y, 1-y)
                    for j in range(i):
                        d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                        min_dist = min(min_dist, d - radii[j])
                    if min_dist > 0.005:
                        pos[i] = [x, y]
                        radii[i] = min_dist * 0.95
                        break
                else:
                    pos[i] = [0.5, 0.5]
                    radii[i] = 0.001

        return pos, radii
    
    def _hex_init():
        """Hexagonal lattice initialization."""
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        r = 0.11
        dx = r * np.sqrt(3)
        dy = r * 1.5
        
        k = 0
        for row in range(6):
            offset = dx / 2 if row % 2 == 1 else 0
            start_x = r + offset
            y = row * dy + r
            for col in range(5):
                if k >= N:
                    break
                x = start_x + col * dx
                if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                    pos[k] = [x, y]
                    radii[k] = r
                    k += 1
            if k >= N:
                break

        rng = np.random.RandomState(123)
        while k < N:
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            for _ in range(5000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
                max_r_circ = max_r_wall
                for j in range(k):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
            if best_r > 0.01:
                pos[k] = [best_x, best_y]
                radii[k] = best_r * 0.98
            else:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
            k += 1
        
        return pos, radii
    
    def _grid_init():
        """Grid-based initial configuration."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        pos = np.array([c[:2] for c in cells])
        radii = np.array([c[2] for c in cells])
        return pos, radii

    def _optimize_with_contact_refinement(pos, radii, deadline, seed=0):
        """
        Optimize using L-BFGS-B with penalty, refining contact graph iteratively.
        """
        rng = np.random.RandomState(seed)
        best_pos = pos.copy()
        best_radii = radii.copy()
        best_value = np.sum(radii)
        
        # Phase 1: Initial optimization with moderate penalty
        for phase in range(3):
            if time_left() < 10:
                break
            
            penalty_weight = 100.0 * (10 ** phase)
            
            def obj(x):
                return penalty_objective(x, penalty_weight)
            
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
            
            try:
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
                )
                
                if result.fun < penalty_objective(x0, penalty_weight):
                    new_pos = result.x[:2*N].reshape(N, 2)
                    new_radii = np.maximum(result.x[2*N:], 1e-6)
                    
                    # Check validity
                    if _is_valid(new_pos, new_radii, tol=1e-7):
                        new_value = np.sum(new_radii)
                        if new_value > best_value:
                            best_pos = new_pos
                            best_radii = new_radii
                            best_value = new_value
                    
                    # Update contact graph and use it for perturbation
                    contacts = _get_contact_graph(best_pos, best_radii, tol=0.02)
                    
                    # Perturb non-contact circles slightly to escape local optima
                    if phase < 2 and time_left() > 20:
                        perturbed_pos = best_pos.copy()
                        perturbed_radii = best_radii.copy()
                        
                        # Identify circles not in contact
                        in_contact = np.zeros(N, dtype=bool)
                        for i, j in contacts:
                            in_contact[i] = True
                            in_contact[j] = True
                        
                        # Perturb circles not in contact
                        for i in range(N):
                            if not in_contact[i]:
                                perturbed_pos[i, 0] += rng.uniform(-0.01, 0.01)
                                perturbed_pos[i, 1] += rng.uniform(-0.01, 0.01)
                                perturbed_pos[i] = np.clip(perturbed_pos[i], 0.0, 1.0)
                        
                        # Try optimization from perturbed state
                        x0_pert = np.concatenate([perturbed_pos.flatten(), perturbed_radii])
                        result_pert = minimize(
                            obj,
                            x0_pert,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
                        )
                        
                        if result_pert.fun < result.fun:
                            new_pos_pert = result_pert.x[:2*N].reshape(N, 2)
                            new_radii_pert = np.maximum(result_pert.x[2*N:], 1e-6)
                            if _is_valid(new_pos_pert, new_radii_pert, tol=1e-7):
                                new_value_pert = np.sum(new_radii_pert)
                                if new_value_pert > best_value:
                                    best_pos = new_pos_pert
                                    best_radii = new_radii_pert
                                    best_value = new_value_pert
            
            except Exception:
                pass
        
        return best_pos, best_radii, best_value
    
    # Generate diverse initializations
    initializations = []
    
    # Best-fit
    initializations.append(_best_fit_init(seed=42))
    initializations.append(_best_fit_init(seed=123))
    
    # Hex
    initializations.append(_hex_init())
    
    # Grid
    initializations.append(_grid_init())
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(2):
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.02, 0.08, size=N)
        initializations.append((pos, radii))
    
    best_pos = None
    best_radii = None
    best_value = -np.inf
    
    for init_idx, (pos, radii) in enumerate(initializations):
        if time_left() < 15:
            break
        
        pos_opt, radii_opt, value_opt = _optimize_with_contact_refinement(pos, radii, DEADLINE, seed=init_idx)
        
        if value_opt > best_value:
            best_value = value_opt
            best_pos = pos_opt
            best_radii = radii_opt
    
    if best_pos is None:
        best_pos, best_radii = _grid_init()
        best_value = np.sum(best_radii)
    
    # Final polish
    if time_left() > 10:
        best_pos, best_radii, best_value = _optimize_with_contact_refinement(
            best_pos, best_radii, DEADLINE, seed=999
        )
    
    # Post-process to ensure strict validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result
