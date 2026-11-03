"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit

    best_value = -np.inf
    best_result = None

    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-5 or x + r > w + 1e-5 or y - r < -1e-5 or y + r > h + 1e-5:
                return -np.inf
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if d < ri + rj - 1e-5:
                    return -np.inf
        return float(np.sum(circles[:, 2]))

    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        circles = np.column_stack([xs, ys, rs])
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = x.copy()

    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        x_beta = beta * x
        result = np.where(x_beta > 0,
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result

    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))

    def objective_and_grad(x, w, h, beta=100.0):
        """
        Force-directed penalty objective with analytical gradients (vectorized).
        x: [x0, y0, r0, x1, y1, r1, ...]
        """
        c = x.reshape(-1, 3)
        n = len(c)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Boundary penalties (vectorized)
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties (vectorized)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)

        rsum = rs[:, None] + rs[None, :]
        iu = np.triu_indices(n, 1)

        v = rsum[iu] - d[iu]
        obj += np.sum(softplus(v, beta))
        g = softplus_grad(v, beta)

        d_safe = np.maximum(d[iu], 1e-12)
        dx_iu = dx[iu]
        dy_iu = dy[iu]

        idx_i = iu[0]
        idx_j = iu[1]
        gx = g * dx_iu / d_safe
        gy = g * dy_iu / d_safe

        np.add.at(grad[0::3], idx_i, -gx)
        np.add.at(grad[0::3], idx_j, gx)
        np.add.at(grad[1::3], idx_i, -gy)
        np.add.at(grad[1::3], idx_j, gy)
        np.add.at(grad[2::3], idx_i, g)
        np.add.at(grad[2::3], idx_j, g)

        grad[2::3] += -1.0

        return obj, grad

    def optimize_from(circles0, w, h, max_iter=400, beta=100.0):
        """L-BFGS-B optimization with softplus penalties."""
        x0 = circles0.flatten()

        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))

        try:
            res = minimize(
                lambda p: objective_and_grad(p, w, h, beta),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return x0

    def polish(circles0, w, h, max_iter=500, beta=200.0):
        """High-precision polish pass."""
        return optimize_from(circles0, w, h, max_iter=max_iter, beta=beta)

    # ---------------- Delaunay + LP dual refinement machinery ----------------
    def get_delaunay_pairs(centers):
        """Get Delaunay neighbor pairs from center positions."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
            return list(edges)
        except Exception:
            return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]

    def solve_lp_with_dual(centers, W, H, edges):
        """
        Solve LP for optimal radii given fixed centers and box size.
        max sum r_i s.t. r_i+r_j <= d_ij (edges), r_i <= dist_to_boundary, r_i >= 0.
        Returns (radii, duals). duals[k] >= 0 is shadow price of constraint k.
        Constraint order: pairs first, then 4 boundary per circle.
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []

        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)

        for i in range(n):
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(H - centers[i][1])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                duals = None
                if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                    m = res.ineqlin.marginals
                    if m is not None:
                        duals = -np.asarray(m)  # marginals are <=0 for A_ub; shadow price = -m
                        duals = np.maximum(duals, 0.0)
                return res.x, duals
        except Exception:
            pass
        return None, None

    def compute_forces(centers, radii, W, H, edges, duals):
        """Compute forces on centers from LP duals (shadow prices)."""
        n = len(centers)
        forces = np.zeros_like(centers)

        if duals is not None and len(duals) > 0:
            num_pair = len(edges)
            for k, (i, j) in enumerate(edges):
                if k >= len(duals):
                    break
                dual_val = duals[k]
                if dual_val < 1e-9:
                    continue
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                direction = d_vec / d
                forces[i] += direction * dual_val
                forces[j] -= direction * dual_val

            offset = num_pair
            for i in range(n):
                idx = offset + 4 * i
                if idx < len(duals) and duals[idx] > 1e-9:
                    forces[i][0] += duals[idx]
                idx = offset + 4 * i + 1
                if idx < len(duals) and duals[idx] > 1e-9:
                    forces[i][0] -= duals[idx]
                idx = offset + 4 * i + 2
                if idx < len(duals) and duals[idx] > 1e-9:
                    forces[i][1] += duals[idx]
                idx = offset + 4 * i + 3
                if idx < len(duals) and duals[idx] > 1e-9:
                    forces[i][1] -= duals[idx]
        else:
            # Fallback: tight-constraint detection
            for i, j in edges:
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                if abs(radii[i] + radii[j] - d) < 1e-5:
                    direction = d_vec / d
                    forces[i] += direction
                    forces[j] -= direction
            for i in range(n):
                if abs(radii[i] - centers[i][0]) < 1e-5:
                    forces[i][0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-5:
                    forces[i][0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-5:
                    forces[i][1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-5:
                    forces[i][1] -= 1.0

        return forces

    def delaunay_lp_refine(circles0, w, h, max_iters=80, step=0.02):
        """
        Refine a full circle configuration by alternating:
          1) solve LP for exact radii at current centers (Delaunay constraint set)
          2) move centers along LP-dual forces
        Returns best valid circle configuration found.
        """
        centers = circles0[:, :2].copy()
        radii = circles0[:, 2].copy()
        best_val = -np.inf
        best_circles = circles0.copy()

        for it in range(max_iters):
            if time.time() > deadline - 1:
                break
            edges = get_delaunay_pairs(centers)
            radii, duals = solve_lp_with_dual(centers, w, h, edges)
            if radii is None:
                break

            # Build candidate circles and validate
            cand = np.column_stack([centers, radii])
            val = evaluate(cand, w, h)
            if np.isfinite(val) and val > best_val:
                best_val = val
                best_circles = cand.copy()

            forces = compute_forces(centers, radii, w, h, edges, duals)
            centers = centers + step * forces
            # Keep centers strictly inside (small margin)
            centers[:, 0] = np.clip(centers[:, 0], 1e-4, w - 1e-4)
            centers[:, 1] = np.clip(centers[:, 1], 1e-4, h - 1e-4)

        return best_circles

    def hex_lattice_init(w, h):
        """Generate hexagonal lattice initialization for N=21 circles."""
        best_circles = None
        best_val = -np.inf

        row_configs = [
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 3, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 3, 4, 4, 4, 2],
            [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
            [4, 4, 4, 4, 4, 1],
            [1, 4, 4, 4, 4, 4],
            [4, 4, 4, 5, 4],
            [5, 4, 4, 4, 4],
            [4, 4, 5, 4, 4],
            [3, 4, 4, 5, 5],
            [5, 5, 4, 4, 3],
            [4, 3, 4, 5, 5],
            [5, 5, 3, 4, 4],
            [2, 4, 4, 4, 3, 4],
            [4, 3, 4, 4, 4, 2],
            [3, 3, 3, 4, 4, 4],
            [4, 4, 4, 3, 3, 3],
            [6, 4, 4, 4, 3],
            [3, 4, 4, 4, 6],
            [7, 4, 4, 4, 2],
            [2, 4, 4, 4, 7],
            [5, 4, 4, 4, 4],
            [4, 4, 4, 4, 5],
        ]

        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue

            n_rows = len(row_counts)
            max_cols = max(row_counts)

            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95

            if r_est < 0.01:
                continue

            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])

            circles = np.array(circles)
            val = evaluate(circles, w, h)

            if val > best_val:
                best_val = val
                best_circles = circles.copy()

        if best_circles is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / a, h / b) * 0.45
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best_circles = np.array(circles)

        return best_circles

    # Try multiple aspect ratios
    aspects = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7]

    for aspect in aspects:
        if time.time() > deadline - 5:
            break

        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        # Standard approach: hex lattice init + gradient optimization
        circles0 = hex_lattice_init(w, h)
        check_and_update(circles0.flatten(), w, h)

        if time.time() < deadline - 12:
            x1 = optimize_from(circles0, w, h, max_iter=300, beta=50.0)
            check_and_update(x1, w, h)

        if time.time() < deadline - 9:
            x2 = optimize_from(x1, w, h, max_iter=300, beta=100.0)
            check_and_update(x2, w, h)

        if time.time() < deadline - 6:
            x3 = polish(x2, w, h, max_iter=400, beta=200.0)
            check_and_update(x3, w, h)

        # NEW: Delaunay + LP-dual refinement on the best softplus-polished config
        if time.time() < deadline - 2:
            circles3 = np.column_stack([x3[0::3], x3[1::3], x3[2::3]])
            refined = delaunay_lp_refine(circles3, w, h, max_iters=80, step=0.02)
            check_and_update(refined.flatten(), w, h)

            # Second, tighter refinement from the refined start
            if time.time() < deadline - 1:
                refined2 = delaunay_lp_refine(refined, w, h, max_iters=60, step=0.01)
                check_and_update(refined2.flatten(), w, h)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.099
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]

    return result
# EVOLVE-BLOCK-END
