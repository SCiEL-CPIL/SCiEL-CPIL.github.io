"""
Circle packing in perimeter-4 rectangle: 21 circles.
Combined approach:
1. Generate diverse initial guesses (grid, hexagonal, random, corner-biased).
2. Optimize using a smooth penalty-based objective (vectorized for speed).
3. Refine with a local search that tries to expand circles while maintaining validity.
4. Use a Delaunay/Voronoi-inspired geometric check to identify "tight" pairs and
   potentially merge or adjust locally, though primarily relying on the smooth optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def objective(z, penalty_weight=1000.0):
        """
        Vectorized objective with smooth penalties for speed.
        z: [x_0, y_0, r_0, ..., x_N-1, y_N-1, r_N-1, w]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Vectorized pairwise overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap * overlap)

        obj += overlap_penalty

        # Vectorized boundary penalty
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        boundary_penalty = penalty_weight * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )

        obj += boundary_penalty

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def refine_solution(x, y, r, w, time_budget):
        """
        Local refinement: try to expand circles slightly if they have room.
        This is a greedy heuristic to polish the solution from the optimizer.
        """
        h = 2.0 - w
        improved = True
        while improved and (time.time() - start_time) < deadline - 5.0:
            improved = False
            for i in range(N):
                # Try to expand circle i
                # Calculate max possible radius based on current position and other circles
                max_r = min(x[i], w - x[i], y[i], h - y[i])
                
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Constraint: dist >= r_i + r_j  => r_i <= dist - r_j
                    max_r = min(max_r, dist - r[j])
                
                if max_r > r[i] * 1.001: # Significant improvement
                    new_r = r[i]
                    # Try to increase r[i]
                    # We can increase it up to max_r, but let's be conservative to allow other circles to move?
                    # Actually, if we fix positions, we can just set to max_r.
                    # But maybe moving helps? For now, just expand in place.
                    r[i] = max_r * 0.9999
                    improved = True
        
        # Final check
        valid, val = check_validity(x, y, r, w)
        if valid:
            return x, y, r, w, val
        else:
            return None

    def generate_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # 1. Grid-based guesses with various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # 2. Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # 3. Hexagonal guess
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # 4. Staggered grid guess
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        sx = w0 / (cols + 0.3)
        sy = h0 / (rows + 0.3)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
            cy = (row + 0.5) * sy
            r0 = min(sx, sy) * 0.49
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        return guesses

    best_value = 0.0
    best_result = None

    guesses = generate_initial_guesses()

    for z_init in guesses:
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Optimize
            res = minimize(
                objective,
                z_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.fun < 1e5: # Filter out bad solutions
                z_best = res.x
                x = z_best[0::3][:N]
                y = z_best[1::3][:N]
                r = z_best[2::3][:N]
                w = z_best[-1]
                
                # Refine
                refined = refine_solution(x, y, r, w, deadline - time.time())
                if refined is not None:
                    rx, ry, rr, rw, rval = refined
                    if rval > best_value:
                        best_value = rval
                        best_result = (rx, ry, rr, rw)
                        
        except Exception as e:
            # print(e)
            pass

    if best_result is not None:
        x, y, r, w = best_result
        # Shrink slightly to ensure strict inequality
        r_final = r * (1 - 1e-9)
        return np.column_stack((x, y, r_final))

    # Fallback: Simple grid in unit square
    cols, rows = 5, 5 # 25 cells, take 21
    w = 1.0
    h = 1.0
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) * (w / cols)
        cy = (row + 0.5) * (h / rows)
        x[i] = cx
        y[i] = cy
        cell_w = w / cols
        cell_h = h / rows
        r[i] = min(cell_w, cell_h) * 0.5 # Inradius of uniform cell
        
    return np.column_stack((x, y, r * (1 - 1e-9)))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
