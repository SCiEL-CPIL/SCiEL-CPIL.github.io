"""Physics-based bubble relaxation for circle packing in unit square."""
N = 26
import numpy as np
import time


def _clamp_to_square(pos, rad):
    """Clamp circle centers so circles stay inside [0,1]^2."""
    pos[:, 0] = np.clip(pos[:, 0], rad, 1.0 - rad)
    pos[:, 1] = np.clip(pos[:, 1], rad, 1.0 - rad)
    return pos


def _check_validity(arr):
    """Check if all circles are inside the square and disjoint."""
    if arr is None or arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise disjoint
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _score(arr):
    """Sum of radii."""
    if arr is None:
        return 0.0
    return float(np.sum(arr[:, 2]))


def _bubble_relax(pos, rad, dt=0.002, max_iters=20000, pressure_k=1.0, repulse_k=50.0):
    """
    Physics-based relaxation:
    - Circles have internal pressure pushing them to expand (increase radius)
    - Repulsive force between overlapping circles pushes them apart
    - Boundary forces keep circles inside the square
    - We iteratively adjust positions and radii to maximize sum of radii
    """
    n = len(rad)
    pos = pos.copy().astype(np.float64)
    rad = rad.copy().astype(np.float64)
    
    # We'll use a simple approach:
    # 1. Try to grow all radii by a small amount
    # 2. Resolve overlaps by pushing circles apart
    # 3. Clamp to square
    # 4. Repeat
    
    best_rad_sum = float(np.sum(rad))
    best_pos = pos.copy()
    best_rad = rad.copy()
    
    for it in range(max_iters):
        # Try to grow radii
        grow = 0.0005
        rad_new = rad + grow
        rad_new = np.clip(rad_new, 0.001, 0.45)
        
        # Check if growing creates overlaps; if so, resolve
        # Use iterative separation
        for sep_iter in range(50):
            _clamp_to_square(pos, rad_new)
            
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.hypot(dx, dy)
                    min_dist = rad_new[i] + rad_new[j]
                    if dist < min_dist - 1e-10:
                        if dist < 1e-12:
                            # Random push
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1e-8
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart proportionally to inverse radius
                        wi = rad_new[j] / (rad_new[i] + rad_new[j])
                        wj = rad_new[i] / (rad_new[i] + rad_new[j])
                        pos[i, 0] -= nx * overlap * wi
                        pos[i, 1] -= ny * overlap * wi
                        pos[j, 0] += nx * overlap * wj
                        pos[j, 1] += ny * overlap * wj
                        moved = True
            
            if not moved:
                break
        
        _clamp_to_square(pos, rad_new)
        
        # Final validity check
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad_new[i] + rad_new[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            rad = rad_new
            current_sum = float(np.sum(rad))
            if current_sum > best_rad_sum:
                best_rad_sum = current_sum
                best_pos = pos.copy()
                best_rad = rad.copy()
        else:
            # Revert
            break
    
    return best_pos, best_rad, best_rad_sum


def _gradient_ascent(pos, rad, max_iters=5000, step=0.001):
    """
    Simple gradient ascent on sum of radii with constraint projection.
    At each step, try to increase each radius slightly, then resolve overlaps.
    """
    n = len(rad)
    pos = pos.copy().astype(np.float64)
    rad = rad.copy().astype(np.float64)
    
    best_sum = float(np.sum(rad))
    best_pos = pos.copy()
    best_rad = rad.copy()
    
    for it in range(max_iters):
        # Grow all radii uniformly
        grow = 0.0003
        rad += grow
        rad = np.clip(rad, 0.001, 0.45)
        
        # Resolve overlaps with multiple iterations
        for _ in range(100):
            _clamp_to_square(pos, rad)
            max_overlap = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.hypot(dx, dy)
                    min_dist = rad[i] + rad[j]
                    if dist < min_dist - 1e-10:
                        if dist < 1e-12:
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx, dy = np.cos(angle), np.sin(angle)
                            dist = 1e-8
                        overlap = min_dist - dist
                        max_overlap = max(max_overlap, overlap)
                        nx, ny = dx / dist, dy / dist
                        # Push apart
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
            
            if max_overlap < 1e-10:
                break
        
        _clamp_to_square(pos, rad)
        
        # Check validity
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad[i] + rad[j] - 1e-8:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s = float(np.sum(rad))
            if s > best_sum:
                best_sum = s
                best_pos = pos.copy()
                best_rad = rad.copy()
        else:
            break
    
    return best_pos, best_rad, best_sum


def _seed_hex():
    """Hexagonal packing seed."""
    pos = np.zeros((N, 2))
    rad = np.zeros(N)
    
    # Hexagonal lattice
    # Try to fit as many as possible in a hex pattern
    # Start with a reasonable radius estimate
    # For 26 circles, approximate hex packing
    # Row-based hex packing
    r_est = 0.08
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    count = 0
    row = 0
    while count < N:
        offset = (row % 2) * r_est
        col = 0
        while count < N:
            x = r_est + offset + col * dx
            y = r_est + row * dy
            if x + r_est <= 1.0 and y + r_est <= 1.0:
                pos[count, 0] = x
                pos[count, 1] = y
                rad[count] = r_est
                count += 1
            col += 1
        row += 1
    
    # If we didn't fill N, add remaining at random positions
    while count < N:
        pos[count, 0] = np.random.uniform(0.1, 0.9)
        pos[count, 1] = np.random.uniform(0.1, 0.9)
        rad[count] = r_est
        count += 1
    
    return pos, rad


def _seed_random():
    """Random seed."""
    pos = np.random.uniform(0.1, 0.9, size=(N, 2))
    rad = np.full(N, 0.06)
    return pos, rad


def _seed_grid():
    """Grid seed."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    pos = np.array(cells[:N])
    r = 0.5 / side - 0.005
    rad = np.full(N, r)
    return pos, rad


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_score = 0.0
    best_arr = None
    
    # Generate multiple seeds
    seeds = []
    seeds.append(('hex', _seed_hex()))
    seeds.append(('grid', _seed_grid()))
    for k in range(10):
        seeds.append((f'rand{k}', _seed_random()))
    
    for name, (pos, rad) in seeds:
        if time.time() > deadline - 10:
            break
        
        # Run bubble relaxation
        pos_r, rad_r, score_r = _bubble_relax(pos, rad, dt=0.002, max_iters=15000)
        if score_r > best_score:
            best_score = score_r
            best_arr = np.column_stack([pos_r, rad_r])
        
        if time.time() > deadline - 10:
            break
        
        # Run gradient ascent from the relaxed position
        pos_g, rad_g, score_g = _gradient_ascent(pos_r, rad_r, max_iters=8000, step=0.001)
        if score_g > best_score:
            best_score = score_g
            best_arr = np.column_stack([pos_g, rad_g])
    
    # Final validation
    if best_arr is not None and _check_validity(best_arr):
        return best_arr
    
    # Fallback: return a valid grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
