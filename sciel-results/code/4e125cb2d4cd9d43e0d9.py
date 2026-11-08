"""Circle packing in perimeter-4 rectangle: iterative placement + radius growth."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _is_valid(pos, radii, w, h, tol=1e-9):
    """Check all constraints are satisfied."""
    n = len(radii)
    for i in range(n):
        cx, cy = pos[i]
        r = radii[i]
        if cx - r < -tol or cy - r < -tol or cx + r > w + tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _compute_value(pos, radii, w, h):
    """Return sum of radii if valid, else -inf."""
    if not _is_valid(pos, radii, w, h):
        return -np.inf
    return float(np.sum(radii))

def _place_circle_greedy(pos, radii, w, h, idx):
    """Place circle idx to maximize its radius given existing circles."""
    n = len(radii)
    # Try many random positions
    best_r = 0.0
    best_pos = None
    np.random.seed(42 + idx * 1000)
    for _ in range(500):
        cx = np.random.uniform(0, w)
        cy = np.random.uniform(0, h)
        # Max radius at this position
        r = min(cx, cy, w - cx, h - cy)
        for j in range(n):
            if j == idx:
                continue
            d = np.sqrt((cx - pos[j,0])**2 + (cy - pos[j,1])**2)
            r = min(r, d - radii[j])
        if r > best_r:
            best_r = r
            best_pos = (cx, cy)
    if best_pos is None:
        best_pos = (w/2, h/2)
        best_r = 0.01
    return best_pos, max(best_r, 0.005)

def _grow_radii(pos, radii, w, h, steps=2000, lr=0.002):
    """Iteratively grow radii with local position adjustments."""
    n = len(radii)
    pos = pos.copy()
    radii = radii.copy()
    
    for step in range(steps):
        # Try to grow each radius slightly
        for i in range(n):
            # Try growing radius
            old_r = radii[i]
            new_r = old_r + lr * 0.1
            # Check if we can fit
            cx, cy = pos[i]
            feasible = True
            # Boundary
            if cx - new_r < 0 or cy - new_r < 0 or cx + new_r > w or cy + new_r > h:
                feasible = False
            # Pairwise
            if feasible:
                for j in range(n):
                    if j == i:
                        continue
                    d = np.sqrt((cx - pos[j,0])**2 + (cy - pos[j,1])**2)
                    if d < new_r + radii[j]:
                        feasible = False
                        break
            if feasible:
                radii[i] = new_r
            else:
                # Try moving position slightly
                # Find direction of most pressure
                dx, dy = 0.0, 0.0
                max_press = 0
                # Boundary pressure
                if cx - radii[i] < 0.01: dx += 1
                if cx + radii[i] > w - 0.01: dx -= 1
                if cy - radii[i] < 0.01: dy += 1
                if cy + radii[i] > h - 0.01: dy -= 1
                for j in range(n):
                    if j == i: continue
                    ddx = cx - pos[j,0]
                    ddy = cy - pos[j,1]
                    d = np.sqrt(ddx*ddx + ddy*ddy) + 1e-10
                    min_d = radii[i] + radii[j]
                    if d < min_d:
                        press = (min_d - d)
                        dx += press * ddx / d
                        dy += press * ddy / d
                norm = np.sqrt(dx*dx + dy*dy) + 1e-10
                step_size = 0.005
                pos[i,0] = np.clip(pos[i,0] + step_size * dx / norm, radii[i], w - radii[i])
                pos[i,1] = np.clip(pos[i,1] + step_size * dy / norm, radii[i], h - radii[i])
    return pos, radii

def _local_search(pos, radii, w, h, iters=5000):
    """Simple local search: random perturbations accepting improvements."""
    n = len(radii)
    best_val = _compute_value(pos, radii, w, h)
    if best_val == -np.inf:
        return pos, radii, best_val
    
    cur_pos = pos.copy()
    cur_radii = radii.copy()
    cur_val = best_val
    
    rng = np.random.RandomState(123)
    
    for it in range(iters):
        i = rng.randint(n)
        old_pos = cur_pos[i].copy()
        old_r = cur_radii[i]
        
        # Random perturbation
        cur_pos[i,0] += rng.normal(0, 0.01)
        cur_pos[i,1] += rng.normal(0, 0.01)
        cur_radii[i] = max(0.005, cur_radii[i] + rng.normal(0, 0.005))
        
        # Project to boundary
        cur_pos[i,0] = np.clip(cur_pos[i,0], cur_radii[i], w - cur_radii[i])
        cur_pos[i,1] = np.clip(cur_pos[i,1], cur_radii[i], h - cur_radii[i])
        
        # Check validity
        valid = True
        cx, cy = cur_pos[i]
        r = cur_radii[i]
        if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
            valid = False
        else:
            for j in range(n):
                if j == i: continue
                d = np.sqrt((cx - cur_pos[j,0])**2 + (cy - cur_pos[j,1])**2)
                if d < r + cur_radii[j]:
                    valid = False
                    break
        
        if valid:
            new_val = float(np.sum(cur_radii))
            if new_val > cur_val:
                cur_val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best_pos = cur_pos.copy()
                    best_radii = cur_radii.copy()
            else:
                # Revert
                cur_pos[i] = old_pos
                cur_radii[i] = old_r
        else:
            cur_pos[i] = old_pos
            cur_radii[i] = old_r
    
    return best_pos, best_radii, best_val

def _solve_for_aspect(w, h, seed=0, time_budget=10):
    """Solve for given aspect ratio."""
    start = time.time()
    rng = np.random.RandomState(seed)
    
    best_val = -np.inf
    best_pos = None
    best_radii = None
    
    # Strategy 1: Greedy placement
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(N):
        p, r = _place_circle_greedy(pos, radii, w, h, i)
        pos[i] = p
        radii[i] = r
    
    # Grow radii
    pos, radii = _grow_radii(pos, radii, w, h, steps=1000)
    val = _compute_value(pos, radii, w, h)
    if val > best_val:
        best_val = val
        best_pos = pos.copy()
        best_radii = radii.copy()
    
    if time.time() - start > time_budget - 2:
        if best_pos is not None:
            return best_pos, best_radii, best_val
        return None, None, -np.inf
    
    # Local search
    pos, radii, val = _local_search(pos, radii, w, h, iters=3000)
    if val > best_val:
        best_val = val
        best_pos = pos.copy()
        best_radii = radii.copy()
    
    # Strategy 2: Random initialization + local search
    if time.time() - start < time_budget - 3:
        for attempt in range(5):
            if time.time() - start > time_budget - 2:
                break
            pos_r = np.column_stack([
                rng.uniform(0.1, w - 0.1, N),
                rng.uniform(0.1, h - 0.1, N)
            ])
            radii_r = rng.uniform(0.05, 0.15, N)
            
            # Project to valid
            for i in range(N):
                radii_r[i] = min(radii_r[i], pos_r[i,0], pos_r[i,1], w - pos_r[i,0], h - pos_r[i,1])
                pos_r[i,0] = np.clip(pos_r[i,0], radii_r[i], w - radii_r[i])
                pos_r[i,1] = np.clip(pos_r[i,1], radii_r[i], h - radii_r[i])
            
            # Check if valid, if not try to fix
            val_r = _compute_value(pos_r, radii_r, w, h)
            if val_r != -np.inf:
                pos_r, radii_r, val_r = _local_search(pos_r, radii_r, w, h, iters=2000)
                if val_r > best_val:
                    best_val = val_r
                    best_pos = pos_r.copy()
                    best_radii = radii_r.copy()
    
    if best_pos is None:
        return None, None, -np.inf
    return best_pos, best_radii, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    best_val = -np.inf
    best_result = None
    best_w, best_h = 1.0, 1.0
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.33, 2.0, 0.5, 1.1, 0.9]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5:
            break
        
        w = 2.0 / (1.0 + 1.0/aspect)
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue
        
        time_per_aspect = min(10, (deadline - 5 - (time.time() - start_time)) / max(1, len(aspect_ratios) - aspect_ratios.index(aspect)))
        
        pos, radii, val = _solve_for_aspect(w, h, seed=aspect*100, time_budget=time_per_aspect)
        
        if pos is not None and val > best_val:
            best_val = val
            best_result = np.column_stack([pos, radii])
            best_w, best_h = w, h
    
    # Final polish: more local search on best
    if best_result is not None and time.time() < deadline - 5:
        pos = best_result[:, :2].copy()
        radii = best_result[:, 2].copy()
        remaining = deadline - 5 - (time.time() - start_time)
        iters = int(remaining * 2000)
        pos, radii, val = _local_search(pos, radii, best_w, best_h, iters=max(1000, iters))
        if val > best_val:
            best_val = val
            best_result = np.column_stack([pos, radii])
    
    if best_result is None:
        # Fallback: uniform grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * 1.0 / side
                cy = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 0.01
                cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
