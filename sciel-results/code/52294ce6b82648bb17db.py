"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    def validity_penalty(x, y, r):
        """Return 0 if valid, else large penalty."""
        n = len(r)
        if np.any(r < 0):
            return 1e10
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return 1e10
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return 1e10
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 1e10
        return 0.0

    def evaluate(pos, rad):
        """Compute sum of radii if valid, else None."""
        x, y = pos[:, 0], pos[:, 1]
        if validity_penalty(x, y, rad) == 0.0:
            return np.sum(rad)
        return None

    def repair_config(x, y, r):
        """Ensure no overlaps by shrinking radii as needed."""
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        r = np.maximum(r, 0.0)

        for _ in range(500):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)

            r_sum = r[:, None] + r[None, :]
            overlaps = d < r_sum - 1e-9
            np.fill_diagonal(overlaps, False)

            if not np.any(overlaps):
                return x, y, r

            max_overlap = 0
            max_i, max_j = 0, 0
            for i in range(N):
                for j in range(i + 1, N):
                    if overlaps[i, j]:
                        ov = r_sum[i, j] - d[i, j]
                        if ov > max_overlap:
                            max_overlap = ov
                            max_i, max_j = i, j

            shrink_amount = max_overlap + 1e-5
            if r[max_i] >= r[max_j]:
                r[max_i] = max(0, r[max_i] - shrink_amount)
            else:
                r[max_j] = max(0, r[max_j] - shrink_amount)

        if validity_penalty(x, y, r) != 0:
            r *= 0.9
        return x, y, r

    def vectorized_relax(x, y, r, lr=0.005, lam_ov=5.0, lam_bd=5.0,
                         max_steps=4000):
        """Fast vectorized force-directed relaxation."""
        for step in range(max_steps):
            if time.time() > deadline:
                break
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            d_safe = np.where(d < 1e-8, 1e-8, d)

            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d)
            np.fill_diagonal(overlap, 0)

            grad_x_ov = -2 * lam_ov * np.sum(overlap * (dx / d_safe), axis=1)
            grad_y_ov = -2 * lam_ov * np.sum(overlap * (dy / d_safe), axis=1)
            grad_r_ov = 2 * lam_ov * np.sum(overlap, axis=1)

            bx_left = np.maximum(0, r - x)
            bx_right = np.maximum(0, r - (1 - x))
            by_bottom = np.maximum(0, r - y)
            by_top = np.maximum(0, r - (1 - y))

            grad_x_bd = -2 * lam_bd * bx_left + 2 * lam_bd * bx_right
            grad_y_bd = -2 * lam_bd * by_bottom + 2 * lam_bd * by_top
            grad_r_bd = 2 * lam_bd * (bx_left + bx_right + by_bottom + by_top)

            grad_x = grad_x_ov + grad_x_bd
            grad_y = grad_y_ov + grad_y_bd
            grad_r = grad_r_ov + grad_r_bd - np.ones(N)

            x = x - lr * grad_x
            y = y - lr * grad_y
            r = r - lr * grad_r

            r = np.maximum(r, 0.001)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            lr *= 0.9995

        return x, y, r

    def max_radius_for_position(x, y, r):
        """For each circle, compute the maximum radius given current positions."""
        new_r = np.zeros(N)
        for i in range(N):
            r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
            r_neighbor = 0.5
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                r_neighbor = min(r_neighbor, (dist - r[j]) * 0.999)
            new_r[i] = min(r_bound, r_neighbor)
            new_r[i] = max(new_r[i], 0.001)
        return new_r

    def local_search(x, y, r, max_iter=300):
        """Constraint-aware local search with radius maximization."""
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = -np.sum(best_r)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            idx = np.random.randint(N)
            cand_x = best_x.copy()
            cand_y = best_y.copy()
            cand_r = best_r.copy()

            progress = it / max_iter
            scale = 0.03 * (1 - progress) + 0.002
            dx = np.random.normal(0, scale)
            dy = np.random.normal(0, scale)
            dr = np.random.normal(0, scale * 0.4)

            cand_x[idx] += dx
            cand_y[idx] += dy
            cand_r[idx] += dr

            cand_x[idx] = np.clip(cand_x[idx], 0, 1)
            cand_y[idx] = np.clip(cand_y[idx], 0, 1)
            cand_r[idx] = np.clip(cand_r[idx], 0.001, 0.5)

            # Repair
            cand_x, cand_y, cand_r = repair_config(cand_x, cand_y, cand_r)

            # Maximize radii
            cand_r = max_radius_for_position(cand_x, cand_y, cand_r)

            if validity_penalty(cand_x, cand_y, cand_r) == 0:
                val = -np.sum(cand_r)
                if val < best_val:
                    best_x, best_y, best_r = cand_x, cand_y, cand_r
                    best_val = val

        return best_x, best_y, best_r

    def hex_init():
        """Generate hexagonal initial layout."""
        dx = np.sqrt(3) / 14.0
        dy = 1.0 / 7.0
        r_init = dx / 2.0

        cols = [7, 6, 7, 6]
        xs, ys, rs = [], [], []
        for c, n_j in enumerate(cols):
            x = (c + 0.5) * dx
            for j in range(n_j):
                y = (j + 0.5) * dy
                if c % 2 == 1:
                    y += dy / 2.0
                    y = min(y, 1.0 - r_init)
                    y = max(y, r_init)
                xs.append(x)
                ys.append(y)
                rs.append(r_init)
        return np.array(xs), np.array(ys), np.array(rs)

    def random_init():
        """Generate random initial layout."""
        xs = np.random.uniform(0.05, 0.95, N)
        ys = np.random.uniform(0.05, 0.95, N)
        rs = np.full(N, 0.03)
        return xs, ys, rs

    def scipy_optimize(x, y, r):
        """Use scipy minimize to optimize."""
        params0 = np.concatenate([x, y, r])

        def objective(params):
            x_ = params[:N]
            y_ = params[N:2 * N]
            r_ = params[2 * N:3 * N]

            pen = 0.0
            for i in range(N):
                if r_[i] > x_[i]:
                    pen += (r_[i] - x_[i]) ** 2
                if r_[i] > 1 - x_[i]:
                    pen += (r_[i] - (1 - x_[i])) ** 2
                if r_[i] > y_[i]:
                    pen += (r_[i] - y_[i]) ** 2
                if r_[i] > 1 - y_[i]:
                    pen += (r_[i] - (1 - y_[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_[i] - x_[j]
                    dy = y_[i] - y_[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < r_[i] + r_[j]:
                        pen += (r_[i] + r_[j] - d) ** 2

            return pen - np.sum(r_)

        result = minimize(objective, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10})
        if result.success:
            x_ = result.x[:N]
            y_ = result.x[N:2 * N]
            r_ = result.x[2 * N:3 * N]
            x_, y_, r_ = repair_config(x_, y_, r_)
            r_ = max_radius_for_position(x_, y_, r_)
            if validity_penalty(x_, y_, r_) == 0:
                return x_, y_, r_
        return x, y, r

    # Main optimization loop with multiple restarts
    best_score = 0.0
    best_pos = None
    best_rad = None

    np.random.seed(42)

    # First, try hexagonal init
    x0, y0, r0 = hex_init()
    x1, y1, r1 = vectorized_relax(x0, y0, r0)
    x2, y2, r2 = local_search(x1, y1, r1, max_iter=200)
    score = evaluate(np.column_stack([x2, y2]), r2)
    if score is not None and score > best_score:
        best_score = score
        best_pos = np.column_stack([x2, y2])
        best_rad = r2.copy()

    # Try scipy on the relaxed result
    if time.time() < deadline:
        x3, y3, r3 = scipy_optimize(x1, y1, r1)
        x4, y4, r4 = local_search(x3, y3, r3, max_iter=100)
        score = evaluate(np.column_stack([x4, y4]), r4)
        if score is not None and score > best_score:
            best_score = score
            best_pos = np.column_stack([x4, y4])
            best_rad = r4.copy()

    # Multiple random restarts
    for restart in range(8):
        if time.time() > deadline - 5:
            break

        xr, yr, rr = random_init()
        x1, y1, r1 = vectorized_relax(xr, yr, rr, max_steps=2000)
        x2, y2, r2 = local_search(x1, y1, r1, max_iter=150)
        score = evaluate(np.column_stack([x2, y2]), r2)
        if score is not None and score > best_score:
            best_score = score
            best_pos = np.column_stack([x2, y2])
            best_rad = r2.copy()

    # Final polish: try to increase radii
    if best_pos is not None and time.time() < deadline - 2:
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            new_rad = best_rad * (1.0 + 0.001)
            if evaluate(best_pos, new_rad) is not None:
                best_rad = new_rad
            else:
                break

    if best_pos is None:
        # Fallback
        x0, y0, r0 = hex_init()
        best_pos = np.column_stack([x0, y0])
        best_rad = r0

    return np.column_stack([best_pos, best_rad])


if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(result)
