"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def is_valid(circles, w, h, eps=1e-7):
        """
        circles: (N, 3) array of [x, y, r]
        Returns True if all circles are inside [0,w]x[0,h] and disjoint.
        """
        if len(circles) != N:
            return False
        for i in range(N):
            x, y, r = circles[i]
            if r < 1e-6:
                return False
            if x - r < -eps or x + r > w + eps:
                return False
            if y - r < -eps or y + r > h + eps:
                return False
            for j in range(i + 1, N):
                x2, y2, r2 = circles[j]
                dx = x - x2
                dy = y - y2
                dist_sq = dx*dx + dy*dy
                min_dist = r + r2
                if dist_sq < (min_dist - eps)**2:
                    return False
        return True

    def greedy_pack(w, h, grid_res=50):
        """
        Greedy packing on a grid. Returns a valid (N, 3) array or None if failed.
        """
        # Generate grid points
        xs = np.linspace(0.05 * w, 0.95 * w, grid_res)
        ys = np.linspace(0.05 * h, 0.95 * h, grid_res)
        
        candidates = []
        for x in xs:
            for y in ys:
                # Max radius based on boundaries
                r_bound = min(x, w - x, y, h - y)
                if r_bound > 1e-4:
                    candidates.append((x, y, r_bound))
        
        # Sort by max possible radius (descending)
        candidates.sort(key=lambda c: c[2], reverse=True)
        
        selected = []
        for x, y, r_bound in candidates:
            if len(selected) == N:
                break
            
            # Check overlap with already selected
            max_r = r_bound
            valid = True
            for sx, sy, sr in selected:
                dx = x - sx
                dy = y - sy
                dist = np.sqrt(dx*dx + dy*dy)
                # The new circle can have radius at most dist - sr
                r_allowed = dist - sr
                if r_allowed < 1e-4:
                    valid = False
                    break
                if r_allowed < max_r:
                    max_r = r_allowed
            
            if valid and max_r > 1e-4:
                selected.append((x, y, max_r))
        
        if len(selected) < N:
            return None
        
        # Convert to numpy array
        arr = np.array(selected, dtype=float)
        return arr

    def local_search(circles, w, h, time_limit):
        """
        Local search to maximize sum of radii, maintaining validity.
        """
        best = circles.copy()
        best_sum = np.sum(best[:, 2])
        T = 0.01
        alpha = 0.995
        T_min = 1e-5
        
        while time.time() < time_limit:
            # Pick a random circle to perturb
            idx = random.randint(0, N - 1)
            new_circles = best.copy()
            
            # Perturb x, y, r
            # Step size scales with T
            step = T * 0.5 + 0.001
            new_circles[idx, 0] += random.gauss(0, step)
            new_circles[idx, 1] += random.gauss(0, step)
            new_circles[idx, 2] += random.gauss(0, step * 0.5)
            
            # Clamp to bounds
            new_circles[idx, 0] = np.clip(new_circles[idx, 0], new_circles[idx, 2], w - new_circles[idx, 2])
            new_circles[idx, 1] = np.clip(new_circles[idx, 1], new_circles[idx, 2], h - new_circles[idx, 2])
            new_circles[idx, 2] = max(1e-4, new_circles[idx, 2])
            
            # Check validity
            if is_valid(new_circles, w, h):
                new_sum = np.sum(new_circles[:, 2])
                delta = new_sum - best_sum
                if delta > 0 or random.random() < np.exp(delta / T):
                    best = new_circles
                    best_sum = new_sum
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        return best, best_sum

    best_result = None
    best_score = -np.inf

    # Try several aspect ratios
    aspect_ratios = [0.4, 0.45, 0.5, 0.55, 0.6]
    
    for t in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w, h = get_rect_dims(t)
        
        # Phase 1: Greedy Init
        init = greedy_pack(w, h, grid_res=40)
        if init is None:
            # Try finer grid
            init = greedy_pack(w, h, grid_res=60)
        if init is None:
            continue
            
        if not is_valid(init, w, h):
            continue
            
        current_sum = np.sum(init[:, 2])
        
        # Phase 2: Local Search
        # Allocate time for local search
        time_for_ls = min(20, (deadline - time.time()) * 0.5)
        if time_for_ls < 2:
            time_for_ls = 2
            
        optimized, opt_sum = local_search(init, w, h, time.time() + time_for_ls)
        
        if opt_sum > best_score:
            best_score = opt_sum
            best_result = optimized.copy()
            best_result_t = t

    if best_result is None:
        # Fallback: Return a trivial valid packing (small circles)
        # Use the greedy from t=0.5
        w, h = get_rect_dims(0.5)
        best_result = greedy_pack(w, h, grid_res=50)
        if best_result is None:
            # Last resort: 21 tiny circles in a row
            best_result = np.zeros((N, 3))
            w, h = get_rect_dims(0.5)
            r = 0.01
            for i in range(N):
                best_result[i, 0] = (i + 0.5) * (w / N)
                best_result[i, 1] = h / 2
                best_result[i, 2] = r
    else:
        # Ensure the result corresponds to a valid rectangle
        # The result array is just (x, y, r). The evaluator implies the bounding box of the circles
        # must have perimeter <= 4.
        # Wait, the task says: "variable rectangle of perimeter 4".
        # "bounding rectangle has perimeter at most 4"
        # My `construct` returns circles. The evaluator checks the bounding box of the returned circles.
        # If I pack them in a rectangle of perimeter 4, their bounding box will be <= 4.
        # However, if I use a specific `t`, the circles are inside `w x h`.
        # The bounding box of the circles might be smaller than `w x h`.
        # This is fine. Perimeter of bounding box <= Perimeter of w x h = 4.
        
        # One detail: The local search might have moved circles such that they don't fill the whole w x h.
        # This is valid.
        
        pass

    return best_result

# EVOLVE-BLOCK-END
