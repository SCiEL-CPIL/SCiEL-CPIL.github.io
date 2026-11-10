import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 20.0

# Grid parameters for structured initialization
_GRID_COLS = 6
_GRID_ROWS = 6
_GRID_SPACING_X = 0.9 / (_GRID_COLS - 1)
_GRID_SPACING_Y = 0.9 / (_GRID_ROWS - 1)
_GRID_OFFSET = 0.05

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute turn matrix
        # turn_mat[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # Only valid for a < b < c
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # Precompute turn from pivot p
        # turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # DP
        # dp[l][s][p_idx][c_idx]
        # l is length of chain
        # s is start index
        # p_idx is previous index
        # c_idx is current end index
        
        # Initialize l=2
        # dp[2][s][s][c] = 1 if turn_p[s, c] > 1e-9
        # Note: turn_p[s, s] is 0.
        # We store dp as a 3D array: dp[s][p_idx][c_idx]
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        
        for s in range(m):
            valid_c = np.where(turn_p[s, s + 1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s + 1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    # dp_prev[s, s:p_idx, p_idx] is the vector of counts for chains ending at p_idx
                    # with start s and previous index in s..p_idx-1
                    vec = dp_prev[s, s:p_idx, p_idx]
                    if np.any(vec > 0):
                        # turn_mat[s:p_idx, p_idx, p_idx+1:]
                        # Shape: (p_idx-s, m-p_idx-1)
                        # We need to mask turns > 1e-9
                        mask = (turn_mat[s:p_idx, p_idx, p_idx + 1:] > 1e-9).astype(np.int64)
                        # contrib[c] = sum_{prev} vec[prev] * mask[prev, c]
                        contrib = vec @ mask
                        dp_curr[s, p_idx, p_idx + 1:] = contrib
            dp_prev = dp_curr

        # Sum up for l=6 with vectorized closing-turn checks
        # dp_prev[s][p_idx][c_idx] is count of chains of length 6
        # s < p_idx < c_idx
        # Need to check closing turns:
        # 1. (q[p_idx], q[c_idx], p)
        # 2. (q[c_idx], p, q[s])
        
        # Vectorized: for each (s, p_idx), compute valid c_idx range
        for s in range(m):
            for p_idx in range(s + 1, m):
                counts = dp_prev[s, p_idx, p_idx + 1:]
                if not np.any(counts > 0):
                    continue
                # c_idx ranges from p_idx+1 to m-1
                c_indices = np.arange(p_idx + 1, m)
                
                # Check closing 1: (q[p_idx], q[c_idx], p)
                # v1 = q[c_idx] - q[p_idx], v2 = p - q[c_idx]
                v1x = q[c_indices, 0] - q[p_idx, 0]
                v1y = q[c_indices, 1] - q[p_idx, 1]
                v2x = p[0] - q[c_indices, 0]
                v2y = p[1] - q[c_indices, 1]
                cross1 = v1x * v2y - v1y * v2x
                
                # Check closing 2: (q[c_idx], p, q[s])
                # v1 = p - q[c_idx], v2 = q[s] - p
                v1x2 = p[0] - q[c_indices, 0]
                v1y2 = p[1] - q[c_indices, 1]
                v2x2 = q[s, 0] - p[0]
                v2y2 = q[s, 1] - p[1]
                cross2 = v1x2 * v2y2 - v1y2 * v2x2
                
                valid = (cross1 > 1e-9) & (cross2 > 1e-9)
                total += int(np.sum(counts[valid]))

    return total

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _get_grid_positions():
    """Generate a structured grid of candidate positions within the unit square."""
    positions = []
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            x = _GRID_OFFSET + col * _GRID_SPACING_X
            y = _GRID_OFFSET + row * _GRID_SPACING_Y
            positions.append((x, y))
    # Add some diagonal positions for extra structure
    for d in range(_GRID_COLS + _GRID_ROWS - 1):
        for col in range(_GRID_COLS):
            row = d - col
            if 0 <= row < _GRID_ROWS:
                x = _GRID_OFFSET + col * _GRID_SPACING_X
                y = _GRID_OFFSET + row * _GRID_SPACING_Y
                positions.append((x, y))
    # Remove duplicates
    seen = set()
    unique = []
    for p in positions:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return np.array(unique)

def _generate_grid_initial(rng):
    """
    Initialize 33 points on a structured grid to maximize collinearity.
    Use row/diagonal bias to create dense non-convex structures.
    """
    grid_pos = _get_grid_positions()
    n_grid = len(grid_pos)
    
    # Select 33 positions from the grid with row/diagonal bias
    # Assign higher probability to positions that form lines
    weights = np.ones(n_grid)
    for i, (x, y) in enumerate(grid_pos):
        # Boost positions on main diagonals
        if abs(x - y) < 0.01:
            weights[i] = 3.0
        if abs(x + y - 1.0) < 0.01:
            weights[i] = 3.0
        # Boost middle rows/columns
        if 0.3 < x < 0.7:
            weights[i] *= 1.5
        if 0.3 < y < 0.7:
            weights[i] *= 1.5
    
    weights /= weights.sum()
    selected = rng.choice(n_grid, size=33, replace=False, p=weights)
    pts = grid_pos[selected].copy()
    
    # Add tiny perturbation to avoid exact collinearity violations
    pts += rng.normal(0, 0.002, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 200:
        # Nudge problematic points
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _snap_to_grid(pts, grid_pos, snap_radius=0.03):
    """Snap points that are close to grid positions."""
    snapped = pts.copy()
    for i in range(len(pts)):
        dists = np.sum((grid_pos - pts[i])**2, axis=1)
        min_dist = np.min(dists)
        if min_dist < snap_radius**2:
            nearest = grid_pos[np.argmin(dists)]
            # Blend toward grid position
            blend = 0.5
            snapped[i] = (1 - blend) * pts[i] + blend * nearest
    return np.clip(snapped, 0.01, 0.99)

def _hill_climb_grid(pts, rng, time_limit):
    """Grid-aware hill climbing that snaps points near grid positions."""
    grid_pos = _get_grid_positions()
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 60
    step_size = 0.04
    snap_prob = 0.3  # probability of snapping to grid after a move
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        offset = rng.normal(0, step_size, 2)
        new_pts[i] = best_pts[i] + offset
        new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        # Optionally snap to nearby grid position
        if rng.random() < snap_prob:
            dists = np.sum((grid_pos - new_pts[i])**2, axis=1)
            min_dist = np.min(dists)
            if min_dist < 0.05**2:
                nearest = grid_pos[np.argmin(dists)]
                blend = 0.4
                new_pts[i] = (1 - blend) * new_pts[i] + blend * nearest
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size *= 0.95
            elif c == best_c and best_c > 1:
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        if not improved and no_improve % 8 == 0:
            # Kick: move multiple points, possibly to grid positions
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for j, i in enumerate(idx):
                if rng.random() < 0.4:
                    # Move to a random grid position
                    gp = grid_pos[rng.integers(0, len(grid_pos))]
                    new_pts[i] = gp + rng.normal(0, 0.005, 2)
                else:
                    offset = rng.normal(0, step_size * 2, 2)
                    new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size *= 0.95

        if no_improve % 40 == 0:
            step_size = min(0.15, step_size * 1.4)

    return best_pts, best_c


def _polish(pts, rng, time_limit):
    """Final polish pass with smaller step sizes and grid snapping."""
    grid_pos = _get_grid_positions()
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    step_size = 0.008
    
    while time.time() - start_time < time_limit and no_improve < 120:
        improved = False
        
        # Try moving 1 point with small step
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        offset = rng.normal(0, step_size, 2)
        new_pts[i] = best_pts[i] + offset
        new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        # Occasionally snap to grid
        if rng.random() < 0.15:
            dists = np.sum((grid_pos - new_pts[i])**2, axis=1)
            min_dist = np.min(dists)
            if min_dist < 0.02**2:
                nearest = grid_pos[np.argmin(dists)]
                blend = 0.3
                new_pts[i] = (1 - blend) * new_pts[i] + blend * nearest
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size *= 0.9
            elif c == best_c and best_c > 1:
                if rng.random() < 0.2:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if no_improve % 40 == 0:
            step_size = max(0.001, step_size * 1.2)

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
        
    # Start with grid-based initialization
    best_pts = _generate_grid_initial(rng)
    best_c = _count_convex7_fast(best_pts)
    
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c
    
    print(f"Initial C (grid): {best_c}")
    
    # Main hill climbing phase with grid-aware moves
    while time_left() > 120:
        chunk_time = min(40, time_left() - 120)
        if chunk_time < 10:
            break
            
        best_pts, best_c = _hill_climb_grid(best_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
            
        print(f"After hill climb: C={best_c}, best_ever={best_ever_c}, t={time.time() - start_time:.1f}")
        
        if best_ever_c <= 1:
            break
            
        # Restart with different grid selection
        if time_left() > 150:
            new_pts = _generate_grid_initial(rng)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                best_pts = new_pts
                best_c = new_c
                best_ever_pts = new_pts.copy()
                best_ever_c = new_c
                print(f"Restart improved to C: {best_c}")
            else:
                best_pts = best_ever_pts.copy()
                best_c = best_ever_c
    
    # Final polish phase
    if time_left() > 30:
        polish_time = min(60, time_left() - 30)
        if polish_time > 5:
            print(f"Starting polish phase with {polish_time:.1f}s budget")
            best_ever_pts, best_ever_c = _polish(best_ever_pts, rng, polish_time)
            print(f"After polish: C={best_ever_c}, t={time.time() - start_time:.1f}")

    return best_ever_pts
