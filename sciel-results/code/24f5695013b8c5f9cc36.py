"""Circle packing in perimeter-4 rectangle: Kinematic Jiggling + Greedy Max-Radius Expansion."""
import numpy as np
import time
from scipy.spatial import Delaunay

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x - r < -tol) or np.any(x + r > w + tol):
            return False, 0.0
        if np.any(y - r < -tol) or np.any(y + r > h + tol):
            return False, 0.0
        # Pairwise distance check
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Aspect ratios to try
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                # Add some jitter to avoid perfect symmetry lock
                cx = (col + 0.5) / side * w0 + np.random.uniform(-0.01, 0.01)
                cy = (row + 0.5) / side * h0 + np.random.uniform(-0.01, 0.01)
                r0 = min(w0 / side, h0 / side) * 0.4
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.10)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        return guesses

    def relax_positions(x, y, r, w, h, max_iter=1000, dt=0.01):
        """Force-directed relaxation to resolve overlaps."""
        x = x.copy()
        y = y.copy()
        for _ in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall repulsion
            fx -= np.maximum(r - x, 0.0) * 50
            fx += np.maximum(x + r - w, 0.0) * 50
            fy -= np.maximum(r - y, 0.0) * 50
            fy += np.maximum(y + r - h, 0.0) * 50

            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        overlap = r_sum - d
                        max_overlap = max(max_overlap, overlap)
                        if d > 1e-6:
                            fx[i] += (dx / d) * overlap * 10
                            fy[i] += (dy / d) * overlap * 10
                            fx[j] -= (dx / d) * overlap * 10
                            fy[j] -= (dy / d) * overlap * 10
            
            if max_overlap < 1e-6:
                break

            # Update with clamping
            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
        
        return x, y

    def jiggling_search(x, y, r, w, h, time_budget):
        """
        Kinematic jiggling: randomly nudge circles, resolve collisions, then expand.
        This is a derivative-free local search that escapes local minima better than gradient methods.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        rng = np.random.default_rng(123)
        
        # Precompute Delaunay neighbors for faster collision checks if needed, 
        # but for N=21, O(N^2) is fast enough.
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            # 1. Random Jiggle: Pick a random circle and move it slightly
            i = rng.integers(0, N)
            angle = rng.uniform(0, 2 * np.pi)
            # Step size decays slightly over time
            step = 0.02 * (1.0 - it / max_iters) + 0.002
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            
            new_x = x[i] + dx
            new_y = y[i] + dy
            
            # Check if this move is valid (doesn't cause immediate overlap or wall breach)
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue
                
            valid_move = True
            for j in range(N):
                if i == j:
                    continue
                ddx = new_x - x[j]
                ddy = new_y - y[j]
                d2 = ddx * ddx + ddy * ddy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - 1e-9:
                    valid_move = False
                    break
            
            if valid_move:
                x[i] = new_x
                y[i] = new_y
                
            # 2. Relax any slight overlaps created by numerical drift or previous moves
            # We do a very light relaxation here to keep things stable
            for _ in range(10):
                fx = np.zeros(N)
                fy = np.zeros(N)
                max_overlap = 0.0
                for a in range(N):
                    for b in range(a + 1, N):
                        dx_ab = x[a] - x[b]
                        dy_ab = y[a] - y[b]
                        d_ab = np.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)
                        r_sum_ab = r[a] + r[b]
                        if d_ab < r_sum_ab:
                            overlap = r_sum_ab - d_ab
                            max_overlap = max(max_overlap, overlap)
                            if d_ab > 1e-6:
                                fx[a] += (dx_ab / d_ab) * overlap
                                fy[a] += (dy_ab / d_ab) * overlap
                                fx[b] -= (dx_ab / d_ab) * overlap
                                fy[b] -= (dy_ab / d_ab) * overlap
                
                if max_overlap < 1e-7:
                    break
                
                x += 0.05 * fx
                y += 0.05 * fy
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
                
            # 3. Greedy Max-Radius Expansion
            # Try to expand each circle to its maximum possible radius
            improved = False
            for i in range(N):
                # Calculate max possible radius for circle i
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                
                if r_max > r[i] + 1e-6:
                    # Binary search for the true max valid radius
                    lo = r[i]
                    hi = r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        r_trial = r.copy()
                        r_trial[i] = mid
                        valid, _ = check_validity(x, y, r_trial, w, h)
                        if valid:
                            lo = mid
                        else:
                            hi = mid
                    new_r = lo
                    if new_r > r[i] + 1e-7:
                        r[i] = new_r
                        improved = True
            
            # 4. If no improvement in expansion after many jiggles, try a larger jump
            # (Implemented via the decaying step size above, but we can add a "restart" logic if needed)
            
            # Early exit if we've reached a stable state (no expansion possible and minimal jiggle effect)
            # This is hard to detect perfectly, so we rely on the time budget.

        # Final polish: one more pass of expansion
        for i in range(N):
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i]:
                r[i] = r_max

        valid, val = check_validity(x, y, r, w, h)
        if valid:
            return x, y, r, val
        else:
            return None

    guesses = get_initial_guesses()

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 10:
            break

        # Relax initial overlaps
        x_rel, y_rel = relax_positions(x0, y0, r0, w0, h0, max_iter=500)
        
        # Check validity after relaxation
        valid, val = check_validity(x_rel, y_rel, r0, w0, h0)
        if valid:
            # Run jiggling search
            result = jiggling_search(x_rel, y_rel, r0, w0, h0, deadline - 5)
            if result is not None:
                x_res, y_res, r_res, val_res = result
                if val_res > best_value:
                    best_value = val_res
                    best_result = (x_res, y_res, r_res, w0, h0)
        else:
            # If relaxation failed to make it valid, try a few more iterations with larger steps
            x_rel2, y_rel2 = relax_positions(x0, y0, r0, w0, h0, max_iter=2000, dt=0.05)
            valid, val = check_validity(x_rel2, y_rel2, r0, w0, h0)
            if valid:
                result = jiggling_search(x_rel2, y_rel2, r0, w0, h0, deadline - 5)
                if result is not None:
                    x_res, y_res, r_res, val_res = result
                    if val_res > best_value:
                        best_value = val_res
                        best_result = (x_res, y_res, r_res, w0, h0)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        return np.column_stack((x, y, r))

    x, y, r, w, h = best_result
    
    # Final safety check and slight shrink if needed
    for _ in range(5):
        r_trial = r * (1 - 1e-8)
        if check_validity(x, y, r_trial, w, h)[0]:
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
