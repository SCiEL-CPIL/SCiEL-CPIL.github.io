import numpy as np
import time
from scipy.optimize import linprog
from scipy.spatial import distance

N = 26
DEADLINE = 115.0  # Safety margin before 120s limit


def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
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
    
    def _lp_maximize_radii(pos, tol=1e-6):
        """
        Given fixed positions, solve LP to maximize sum of radii subject to:
        - r_i <= pos[i,0], r_i <= 1-pos[i,0], r_i <= pos[i,1], r_i <= 1-pos[i,1]
        - r_i + r_j <= dist(i,j) for all pairs
        - r_i >= 0
        """
        n = N
        # Variables: r_0, ..., r_{n-1}
        # Maximize sum(r) -> minimize -sum(r)
        c = -np.ones(n)
        
        # Inequality constraints: A_ub @ r <= b_ub
        # Boundary: r_i <= pos[i,0], etc.
        A_rows = []
        b_rows = []
        
        for i in range(n):
            # r_i <= pos[i,0]
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(pos[i, 0])
            
            # r_i <= 1 - pos[i,0]
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0 - pos[i, 0])
            
            # r_i <= pos[i,1]
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(pos[i, 1])
            
            # r_i <= 1 - pos[i,1]
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0 - pos[i, 1])
        
        # Pairwise: r_i + r_j <= dist(i,j)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_rows.append(d)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * n
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                radii = np.maximum(result.x, 0.0)
                return radii
        except Exception:
            pass
        
        # Fallback: greedy radius assignment
        radii = np.zeros(n)
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if j != i:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(0.0, max_r)
        return radii
    
    def _force_directed_relaxation(pos, radii, n_iter=2000, lr=0.01):
        """
        Force-directed relaxation: push overlapping circles apart,
        pull circles toward center if near boundary.
        """
        pos = pos.copy()
        radii = radii.copy()
        
        for it in range(n_iter):
            if time_left() < 5:
                break
            
            forces = np.zeros_like(pos)
            
            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = radii[i] + radii[j]
                    if d < min_d and d > 1e-10:
                        overlap = min_d - d
                        fx = (dx / d) * overlap
                        fy = (dy / d) * overlap
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    elif d <= 1e-10:
                        # Random small push
                        angle = np.random.uniform(0, 2 * np.pi)
                        forces[i, 0] += 0.01 * np.cos(angle)
                        forces[i, 1] += 0.01 * np.sin(angle)
                        forces[j, 0] -= 0.01 * np.cos(angle)
                        forces[j, 1] -= 0.01 * np.sin(angle)
            
            # Boundary repulsion
            for i in range(N):
                if pos[i, 0] < radii[i]:
                    forces[i, 0] += (radii[i] - pos[i, 0])
                if pos[i, 0] > 1.0 - radii[i]:
                    forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i]))
                if pos[i, 1] < radii[i]:
                    forces[i, 1] += (radii[i] - pos[i, 1])
                if pos[i, 1] > 1.0 - radii[i]:
                    forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i]))
            
            # Apply forces with adaptive learning rate
            decay = 1.0 - 0.5 * (it / n_iter)
            step = lr * decay
            pos += step * forces
            
            # Clamp to unit square
            pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
            pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        return pos, radii
    
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
    
    def _optimize(pos_init, radii_init, seed=0):
        """
        Structural optimization:
        1. Force-directed relaxation to resolve overlaps
        2. LP to maximize radii given positions
        3. Iterative refinement: adjust positions based on LP results, re-solve LP
        """
        rng = np.random.RandomState(seed)
        pos = pos_init.copy()
        radii = radii_init.copy()
        
        best_pos = pos.copy()
        best_radii = radii.copy()
        best_value = np.sum(radii)
        
        # Phase 1: Force-directed relaxation
        if time_left() > 30:
            pos, radii = _force_directed_relaxation(pos, radii, n_iter=1500, lr=0.005)
            radii = _lp_maximize_radii(pos)
            if np.sum(radii) > best_value and _is_valid(pos, radii, tol=1e-7):
                best_pos = pos.copy()
                best_radii = radii.copy()
                best_value = np.sum(radii)
        
        # Phase 2: Iterative position refinement with LP radius optimization
        for iteration in range(50):
            if time_left() < 15:
                break
            
            pos = best_pos.copy()
            
            # Small random perturbation to escape local optima
            if iteration > 5:
                pos += rng.uniform(-0.005, 0.005, size=pos.shape)
                pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
                pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
            
            # Force-directed relaxation
            pos, _ = _force_directed_relaxation(pos, best_radii, n_iter=500, lr=0.003)
            
            # LP maximize radii
            radii_new = _lp_maximize_radii(pos)
            
            if _is_valid(pos, radii_new, tol=1e-7):
                value_new = np.sum(radii_new)
                if value_new > best_value:
                    best_pos = pos.copy()
                    best_radii = radii_new.copy()
                    best_value = value_new
        
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
    
    # Random with larger radii
    rng = np.random.RandomState(42)
    for _ in range(3):
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.10, size=N)
        initializations.append((pos, radii))
    
    best_pos = None
    best_radii = None
    best_value = -np.inf
    
    for init_idx, (pos, radii) in enumerate(initializations):
        if time_left() < 20:
            break
        
        pos_opt, radii_opt, value_opt = _optimize(pos, radii, seed=init_idx)
        
        if value_opt > best_value:
            best_value = value_opt
            best_pos = pos_opt
            best_radii = radii_opt
    
    if best_pos is None:
        best_pos, best_radii = _grid_init()
        best_value = np.sum(best_radii)
    
    # Final polish: more iterations
    if time_left() > 20:
        best_pos, best_radii, best_value = _optimize(
            best_pos, best_radii, seed=999
        )
    
    # Post-process to ensure strict validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result
