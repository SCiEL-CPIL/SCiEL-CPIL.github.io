"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    best_sum = 0.0
    best_result = None

    def check_validity(positions, radii, w, h):
        """Check if packing is valid and return sum of radii if so."""
        if w <= 0 or h <= 0:
            return -1.0
        if np.any(positions[:, 0] < radii - 1e-9):
            return -1.0
        if np.any(positions[:, 0] > w - radii + 1e-9):
            return -1.0
        if np.any(positions[:, 1] < radii - 1e-9):
            return -1.0
        if np.any(positions[:, 1] > h - radii + 1e-9):
            return -1.0

        dx = positions[:, 0][:, None] - positions[:, 0][None, :]
        dy = positions[:, 1][:, None] - positions[:, 1][None, :]
        d = np.sqrt(dx * dx + dy * dy)
        r_sum = radii[:, None] + radii[None, :]
        if np.any(d < r_sum - 1e-9):
            return -1.0

        return np.sum(radii)

    def _hexagonal_init(w, h, n_rows, n_cols):
        """Create a hexagonal/staggered grid initialization."""
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        
        idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if idx >= N:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                # Stagger odd rows
                if r % 2 == 1:
                    cx += cell_w * 0.25
                cx = np.clip(cx, cell_w * 0.1, w - cell_w * 0.1)
                
                max_r = min(cell_w, cell_h) / 2.0 * 0.85
                # Vary radii slightly for diversity
                max_r *= (0.9 + 0.1 * np.sin(r * 0.7 + c * 1.3))
                
                positions[idx, 0] = cx
                positions[idx, 1] = cy
                radii[idx] = max(max_r, 0.02)
                idx += 1
        
        while idx < N:
            positions[idx, 0] = w / 2.0 + (idx - N/2) * 0.1
            positions[idx, 1] = h / 2.0 + (idx % 3 - 1) * 0.1
            radii[idx] = 0.02
            idx += 1

        return positions, radii

    def _square_grid_init(w, h, n_rows, n_cols):
        """Create a square grid initialization."""
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        
        idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if idx >= N:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                max_r = min(cell_w, cell_h) / 2.0 * 0.85
                positions[idx, 0] = cx
                positions[idx, 1] = cy
                radii[idx] = max(max_r, 0.02)
                idx += 1
        
        while idx < N:
            positions[idx, 0] = w / 2.0 + (idx - N/2) * 0.1
            positions[idx, 1] = h / 2.0 + (idx % 3 - 1) * 0.1
            radii[idx] = 0.02
            idx += 1

        return positions, radii

    def _concentric_ring_init(w, h):
        """Create a concentric ring initialization."""
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        cx, cy = w / 2.0, h / 2.0
        max_r = min(w, h) / 2.0 * 0.9
        
        # Center circle
        positions[0, 0] = cx
        positions[0, 1] = cy
        radii[0] = max_r * 0.3
        
        # Ring 1: 6 circles
        ring1_r = max_r * 0.7
        ring1_cr = max_r * 0.25
        idx = 1
        for i in range(6):
            if idx >= N:
                break
            angle = i * np.pi / 3.0
            positions[idx, 0] = cx + ring1_r * np.cos(angle)
            positions[idx, 1] = cy + ring1_r * np.sin(angle)
            radii[idx] = ring1_cr
            idx += 1
        
        # Ring 2: 12 circles
        ring2_r = max_r * 1.0
        ring2_cr = max_r * 0.18
        for i in range(12):
            if idx >= N:
                break
            angle = i * np.pi / 6.0 + np.pi / 12.0
            positions[idx, 0] = cx + ring2_r * np.cos(angle)
            positions[idx, 1] = cy + ring2_r * np.sin(angle)
            radii[idx] = ring2_cr
            idx += 1
        
        while idx < N:
            angle = idx * 0.5
            positions[idx, 0] = cx + max_r * 0.5 * np.cos(angle)
            positions[idx, 1] = cy + max_r * 0.5 * np.sin(angle)
            radii[idx] = 0.02
            idx += 1

        return positions, radii

    def _direct_constraint_objective(z, w, h):
        """
        Direct constraint-based objective.
        z: [x0, y0, r0, x1, y1, r1, ..., x_{N-1}, y_{N-1}, r_{N-1}]
        Minimize negative sum of radii with hard constraints.
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        
        return -np.sum(r)

    def _build_constraints(w, h):
        """Build constraint functions for SLSQP."""
        # Wall constraints: r_i <= x_i, x_i + r_i <= w, r_i <= y_i, y_i + r_i <= h
        # Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        def wall_constraints(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            return np.concatenate([
                x - r,           # x >= r
                w - x - r,       # x + r <= w
                y - r,           # y >= r
                h - y - r        # y + r <= h
            ])
        
        def non_overlap_constraints(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            constraints = []
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    r_sum = r[i] + r[j]
                    # dist_sq >= r_sum^2  =>  dist_sq - r_sum^2 >= 0
                    constraints.append(dist_sq - r_sum * r_sum)
            return np.array(constraints)
        
        return wall_constraints, non_overlap_constraints

    def _polish_slsqp(w, h, positions, radii, max_iter=200):
        """Polish using SLSQP with direct constraints."""
        z0 = np.concatenate([positions.flatten(), radii.flatten()])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01, w - 0.01))  # x
            bounds.append((0.01, h - 0.01))  # y
            bounds.append((0.01, min(w, h) / 2.0 - 0.01))  # r
        
        try:
            result = minimize(
                lambda z: _direct_constraint_objective(z, w, h),
                z0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda z: _build_constraints(w, h)[0](z)},
                    {'type': 'ineq', 'fun': lambda z: _build_constraints(w, h)[1](z)}
                ],
                options={'maxiter': max_iter, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < 0:
                z_opt = result.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                positions_opt = np.column_stack([x_opt, y_opt])
                
                score = check_validity(positions_opt, r_opt, w, h)
                if score > 0:
                    return positions_opt, r_opt, w, h, score
        except Exception:
            pass
        
        return None

    def _simulated_annealing_polish(w, h, positions, radii, max_iter=5000, t0=0.05, t1=0.001):
        """Simulated annealing to escape local minima."""
        positions = positions.copy().astype(float)
        radii = radii.copy().astype(float)
        
        best_positions = positions.copy()
        best_radii = radii.copy()
        best_score = check_validity(best_positions, best_radii, w, h)
        
        current_score = best_score
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            temperature = t0 * (t1 / t0) ** (it / max_iter)
            
            # Pick a random circle and perturb it
            i = np.random.randint(N)
            old_x, old_y, old_r = positions[i, 0], positions[i, 1], radii[i]
            
            # Random perturbation
            new_x = old_x + np.random.uniform(-0.05, 0.05)
            new_y = old_y + np.random.uniform(-0.05, 0.05)
            new_r = old_r + np.random.uniform(-0.02, 0.02)
            
            # Clamp to valid range
            new_r = np.clip(new_r, 0.01, min(w, h) / 2.0 - 0.01)
            new_x = np.clip(new_x, new_r, w - new_r)
            new_y = np.clip(new_y, new_r, h - new_r)
            
            positions[i, 0] = new_x
            positions[i, 1] = new_y
            radii[i] = new_r
            
            new_score = check_validity(positions, radii, w, h)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_positions = positions.copy()
                    best_radii = radii.copy()
            else:
                # Accept with probability exp((new_score - current_score) / temperature)
                delta = new_score - current_score
                if delta < 0 and np.random.rand() < np.exp(delta / temperature):
                    current_score = new_score
                else:
                    # Revert
                    positions[i, 0] = old_x
                    positions[i, 1] = old_y
                    radii[i] = old_r
        
        if best_score > 0:
            return best_positions, best_radii, w, h, best_score
        return None

    def _expand_radii(w, h, positions, radii, max_rounds=50):
        """Try to expand each circle's radius while maintaining validity."""
        positions = positions.copy().astype(float)
        radii = radii.copy().astype(float)
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old_r = radii[i]
                max_possible = min(w, h) / 2.0 - 0.01
                for dr in [0.003, 0.0015, 0.0008, 0.0004]:
                    new_r = old_r + dr
                    if new_r > max_possible:
                        continue
                    
                    # Try to find a valid position for the larger circle
                    found = False
                    for _ in range(20):
                        cand_x = positions[i, 0] + np.random.uniform(-0.03, 0.03)
                        cand_y = positions[i, 1] + np.random.uniform(-0.03, 0.03)
                        
                        # Wall check
                        if cand_x < new_r or cand_x > w - new_r or \
                           cand_y < new_r or cand_y > h - new_r:
                            continue
                        
                        # Check against all other circles
                        dx_o = cand_x - positions[:, 0]
                        dy_o = cand_y - positions[:, 1]
                        dists = np.sqrt(dx_o * dx_o + dy_o * dy_o)
                        dists[i] = np.inf
                        if np.all(dists >= new_r + radii - 1e-9):
                            positions[i, 0] = cand_x
                            positions[i, 1] = cand_y
                            radii[i] = new_r
                            improved = True
                            found = True
                            break
                    if found:
                        break
            
            if not improved:
                break
        
        score = check_validity(positions, radii, w, h)
        if score > 0:
            return positions, radii, w, h, score
        return None

    def _optimize_aspect_ratio(positions, radii, w, h):
        """Try different aspect ratios to improve the packing."""
        best = None
        best_score = 0
        
        for w_frac in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            w_new = w_frac
            h_new = 2.0 - w_new
            if w_new < 0.3 or h_new < 0.3:
                continue
            
            # Scale positions to new rectangle
            scale_x = w_new / w
            scale_y = h_new / h
            pos_scaled = positions.copy()
            pos_scaled[:, 0] *= scale_x
            pos_scaled[:, 1] *= scale_y
            radii_scaled = radii * min(scale_x, scale_y)
            
            # Clamp to new bounds
            for i in range(N):
                pos_scaled[i, 0] = np.clip(pos_scaled[i, 0], radii_scaled[i], w_new - radii_scaled[i])
                pos_scaled[i, 1] = np.clip(pos_scaled[i, 1], radii_scaled[i], h_new - radii_scaled[i])
            
            score = check_validity(pos_scaled, radii_scaled, w_new, h_new)
            if score > best_score:
                best_score = score
                best = (pos_scaled, radii_scaled, w_new, h_new)
        
        if best is not None and best_score > 0:
            return best
        return None

    # Main search
    initializations = []
    
    # Hexagonal grids
    for rows in [3, 4, 5]:
        for cols in [4, 5, 6, 7]:
            if rows * cols >= N:
                for w_frac in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    w = w_frac
                    h = 2.0 - w
                    if w < 0.3 or h < 0.3:
                        continue
                    pos, rad = _hexagonal_init(w, h, rows, cols)
                    initializations.append((w, h, pos, rad))
    
    # Square grids
    for rows in [3, 4, 5]:
        for cols in [4, 5, 6, 7]:
            if rows * cols >= N:
                for w_frac in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    w = w_frac
                    h = 2.0 - w
                    if w < 0.3 or h < 0.3:
                        continue
                    pos, rad = _square_grid_init(w, h, rows, cols)
                    initializations.append((w, h, pos, rad))
    
    # Concentric rings
    for w_frac in [0.9, 1.0, 1.1]:
        w = w_frac
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            continue
        pos, rad = _concentric_ring_init(w, h)
        initializations.append((w, h, pos, rad))

    # Sort by initial sum of radii
    initializations.sort(key=lambda x: -np.sum(x[3]))

    for w, h, positions, radii in initializations:
        if time.time() > deadline - 15.0:
            break
        
        # First, try to fix any overlaps with a simple relaxation
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-12:
                        dist = 1e-12
                        dx, dy = 1e-6, 0.0
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        fx = 0.3 * overlap * dx / dist
                        fy = 0.3 * overlap * dy / dist
                        positions[i, 0] += fx * 0.5
                        positions[i, 1] += fy * 0.5
                        positions[j, 0] -= fx * 0.5
                        positions[j, 1] -= fy * 0.5
                        improved = True
            
            # Wall clamping
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
            
            if not improved:
                break
        
        score = check_validity(positions, radii, w, h)
        if score <= 0:
            continue
        
        # SLSQP polish
        result = _polish_slsqp(w, h, positions, radii)
        if result is not None:
            pos_opt, r_opt, w_opt, h_opt, score_opt = result
            if score_opt > best_sum:
                best_sum = score_opt
                best_result = np.column_stack([pos_opt, r_opt])
                positions, radii, w, h = pos_opt, r_opt, w_opt, h_opt
        
        # Simulated annealing
        if time.time() < deadline - 10.0:
            sa_result = _simulated_annealing_polish(w, h, positions, radii, max_iter=3000)
            if sa_result is not None:
                pos_sa, r_sa, w_sa, h_sa, score_sa = sa_result
                if score_sa > best_sum:
                    best_sum = score_sa
                    best_result = np.column_stack([pos_sa, r_sa])
                    positions, radii, w, h = pos_sa, r_sa, w_sa, h_sa
        
        # Expand radii
        if time.time() < deadline - 5.0:
            expand_result = _expand_radii(w, h, positions, radii)
            if expand_result is not None:
                pos_exp, r_exp, w_exp, h_exp, score_exp = expand_result
                if score_exp > best_sum:
                    best_sum = score_exp
                    best_result = np.column_stack([pos_exp, r_exp])
        
        # Try aspect ratio optimization
        if time.time() < deadline - 5.0:
            ar_result = _optimize_aspect_ratio(positions, radii, w, h)
            if ar_result is not None:
                pos_ar, r_ar, w_ar, h_ar = ar_result
                score_ar = check_validity(pos_ar, r_ar, w_ar, h_ar)
                if score_ar > best_sum:
                    best_sum = score_ar
                    best_result = np.column_stack([pos_ar, r_ar])

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = _square_grid_init(w, h, 5, 5)
        best_result = np.column_stack([positions, radii])

    return best_result
# EVOLVE-BLOCK-END
