"""Circle packing in perimeter-4 rectangle.
Approach: Simulated Annealing for position optimization, followed by Apollonius growth.
- We use Simulated Annealing to find good positions for the circles.
- The energy function penalizes overlaps and wall violations, and rewards large radii.
- After the annealing phase, we use the Apollonius growth routine to maximize radii for the fixed positions.
- We try multiple aspect ratios.
"""
import numpy as np
import time

N = 21
DEADLINE_PAD = 20.0


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_PAD

    best = None
    best_val = 0.0

    # Aspect ratios (w + h = 2)
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55,
                     1.25, 0.8, 1.4, 0.7, 1.6, 0.63, 1.1, 0.9]

    for w0 in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        # Initialize with a grid
        x, y = _grid_init(w0, h0)
        
        # Run Simulated Annealing
        x, y, r = _simulated_annealing(x, y, w0, h0, deadline)
        
        # Final Apollonius growth to tighten
        x, y, r = _apollonius_pack(x, y, r, w0, h0, deadline)

        val = _sum_if_valid(x, y, r, w0, h0)
        if val > best_val:
            best_val = val
            best = (x.copy(), y.copy(), r.copy(), w0, h0)

    if best is None:
        # Fallback
        w0, h0 = 1.0, 1.0
        x, y = _grid_init(w0, h0)
        r = np.full(N, 0.05)
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)
        return np.column_stack((x, y, r))

    x, y, r, w0, h0 = best
    # Final safety shrink
    for _ in range(20):
        if _is_valid(x, y, r * (1 - 1e-7), w0, h0):
            r = r * (1 - 1e-7)
        else:
            break

    return np.column_stack((x, y, r))


def _grid_init(w, h):
    side = int(np.ceil(np.sqrt(N)))
    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
    return x, y


def _max_radius(i, x, y, r, w, h):
    """Exact max radius of circle i at its current position (Apollonius)."""
    mr = min(x[i], w - x[i], y[i], h - y[i])
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.hypot(dx, dy)
        cand = d - r[j]
        if cand < mr:
            mr = cand
    return max(0.0, mr)


def _apollonius_pack(x, y, r, w, h, deadline):
    """Iteratively grow circles to their exact Apollonius radius and push
    overlapping circles toward their cell centers. Returns improved arrays."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    # Phase A: grow to exact radius (positions fixed).
    for _ in range(60):
        if time.time() > deadline:
            break
        changed = False
        for i in range(N):
            mr = _max_radius(i, x, y, r, w, h)
            if mr > r[i] + 1e-9:
                r[i] = mr
                changed = True
        if not changed:
            break

    # Phase B: push overlapping circles toward cell centers, then re-grow.
    for outer in range(400):
        if time.time() > deadline - 5:
            break

        # Find all overlapping pairs.
        overlaps = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j] - 1e-10:
                    overlaps.append((i, j, r[i] + r[j] - d))
        if not overlaps:
            break

        # Compute net push direction per circle: toward cell center, weighted.
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i, j, ov in overlaps:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < 1e-9:
                dx, dy, d = 1e-4, 0.0, 1e-4
            ux, uy = dx / d, dy / d
            fx[i] += ux * ov
            fy[i] += uy * ov
            fx[j] -= ux * ov
            fy[j] -= uy * ov

        # Add a gentle pull toward each circle's own cell center.
        for i in range(N):
            cx, cy = _cell_center(i, x, y, r, w, h)
            fx[i] += 0.15 * (cx - x[i])
            fy[i] += 0.15 * (cy - y[i])

        # Adaptive step.
        mag = np.max(np.hypot(fx, fy)) if N else 0.0
        step = 0.5 / (1.0 + mag)
        step = min(step, 0.02)

        for i in range(N):
            x[i] += step * fx[i]
            y[i] += step * fy[i]
            x[i] = min(max(x[i], r[i]), w - r[i])
            y[i] = min(max(y[i], r[i]), h - r[i])

        # Re-grow to exact radius.
        for i in range(N):
            mr = _max_radius(i, x, y, r, w, h)
            if mr > r[i] + 1e-9:
                r[i] = mr

    # Phase C: final local expansion sweep.
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            mr = _max_radius(i, x, y, r, w, h)
            if mr > r[i] + 1e-8:
                r[i] = mr
                improved = True
        if not improved:
            break

    return x, y, r


def _cell_center(i, x, y, r, w, h):
    """Centroid of the free (unoccupied) region of circle i's Voronoi cell."""
    lo_x = max(0.0, x[i] - (r[i] + 0.3))
    hi_x = min(w, x[i] + (r[i] + 0.3))
    lo_y = max(0.0, y[i] - (r[i] + 0.3))
    hi_y = min(h, y[i] + (r[i] + 0.3))
    if hi_x <= lo_x or hi_y <= lo_y:
        return x[i], y[i]

    sx, sy = np.linspace(lo_x, hi_x, 12)
    sy_grid, sx_grid = np.meshgrid(sy, sx)
    pts = np.stack([sx_grid.ravel(), sy_grid.ravel()], axis=1)

    di = np.sqrt((pts[:, 0] - x[i]) ** 2 + (pts[:, 1] - y[i]) ** 2) - r[i]
    mask = np.ones(len(pts), dtype=bool)
    for j in range(N):
        if i == j:
            continue
        dj = np.sqrt((pts[:, 0] - x[j]) ** 2 + (pts[:, 1] - y[j]) ** 2) - r[j]
        mask &= (di < dj)
    pts = pts[mask]
    if len(pts) == 0:
        return x[i], y[i]
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _is_valid(x, y, r, w, h, tol=1e-8):
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > w + tol) or np.any(y + r > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_if_valid(x, y, r, w, h):
    if _is_valid(x, y, r, w, h):
        return float(np.sum(r))
    return 0.0


def _simulated_annealing(x, y, w, h, deadline):
    """
    Simulated Annealing to optimize positions.
    We start with small radii and grow them as we go, or keep radii fixed and move positions?
    Better: Keep radii fixed at a reasonable estimate, move positions to minimize overlap penalty.
    Then, at the end, we will grow radii.
    """
    x = x.copy()
    y = y.copy()
    
    # Initial radii: assume uniform packing
    # Area of rect = w*h. Area of N circles = N * pi * r^2.
    # r_approx = sqrt(w*h / (N*pi)) * packing_factor
    r_approx = np.sqrt(w * h / (N * np.pi)) * 0.9
    r = np.full(N, r_approx)
    
    # Ensure initial validity
    for i in range(N):
        r[i] = min(r[i], x[i], w-x[i], y[i], h-y[i])
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    max_iter = 5000
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_energy = _energy(x, y, r, w, h)
    
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
        if T < T_min:
            break
            
        # Pick a random circle
        idx = np.random.randint(N)
        
        # Propose a move
        dx = np.random.normal(0, T * 0.5)
        dy = np.random.normal(0, T * 0.5)
        
        new_x = x.copy()
        new_y = y.copy()
        new_x[idx] += dx
        new_y[idx] += dy
        
        # Clamp to walls (penalty will handle if we don't, but clamping is safer for SA stability)
        new_x[idx] = np.clip(new_x[idx], 0, w)
        new_y[idx] = np.clip(new_y[idx], 0, h)
        
        # Calculate energy change
        # Energy = -sum(r) + lambda * overlap_penalty
        # We can approximate delta energy by only checking interactions with idx
        # But for simplicity and correctness, we compute full energy if N is small enough.
        # N=21, O(N^2) is 441 ops. 5000 iters * 441 = 2.2M ops. Very fast.
        
        old_energy = _energy(x, y, r, w, h)
        new_energy = _energy(new_x, new_y, r, w, h)
        
        delta = new_energy - old_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x = new_x
            y = new_y
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                
        T *= cooling
        
    # Return best positions found, and the radii used during SA
    # Note: The radii in SA were fixed. We return these radii to be grown by Apollonius.
    return best_x, best_y, best_r


def _energy(x, y, r, w, h):
    """
    Energy function for SA.
    We want to maximize sum(r), so minimize -sum(r).
    We want to minimize overlaps and wall violations.
    """
    penalty = 0.0
    # Wall violations
    for i in range(N):
        d_wall = min(x[i], w - x[i], y[i], h - y[i])
        if d_wall < r[i]:
            penalty += (r[i] - d_wall) ** 2
            
    # Overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            min_d = r[i] + r[j]
            if d < min_d:
                penalty += (min_d - d) ** 2
                
    # Objective: -sum(r) + 100 * penalty
    return -np.sum(r) + 100.0 * penalty
