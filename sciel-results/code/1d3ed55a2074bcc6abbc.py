"""Circle packing in perimeter-4 rectangle using Successive Convex Relaxation."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import solve
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    best_score = -1
    best_result = None

    # Pre-compute pairs for overlap constraints
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    P = len(pairs)

    def validate_and_score(x, w, h):
        cx_v = x[:N]
        cy_v = x[N:2*N]
        r_v = x[2*N:]

        if np.any(cx_v - r_v < -1e-7) or np.any(cx_v + r_v - w > 1e-7):
            return -1, None
        if np.any(cy_v - r_v < -1e-7) or np.any(cy_v + r_v - h > 1e-7):
            return -1, None

        dx = cx_v[:, None] - cx_v[None, :]
        dy = cy_v[:, None] - cy_v[None, :]
        dist_sq = dx * dx + dy * dy
        r_sum = r_v[:, None] + r_v[None, :]

        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol_sq = dist_sq[mask] - r_sum[mask] ** 2
        if np.any(viol_sq < -1e-7):
            return -1, None

        score = np.sum(r_v)
        return score, np.column_stack((cx_v, cy_v, r_v))

    def solve_scr_qp(x0, w, h, max_iter=30, tol=1e-9):
        """
        Successive Convex Relaxation for circle packing.
        Linearize the non-convex constraint (d_ij >= r_i + r_j) as:
        d_ij >= r_i + r_j - lambda*(d_ij - d_ij^0)  at current point x0
        which is convex (linear) in x.
        Minimize -sum(r_i) subject to linearized constraints and bounds.
        """
        x = x0.copy()
        n = 3 * N

        # QP formulation: minimize 0.5*x'Qx + c'x subject to Ax <= b, bounds
        # Since objective is linear (-sum r_i), Q=0, c = objective gradient
        c = np.zeros(n)
        c[2*N:] = -1.0  # minimize -sum(r_i)

        # Build constraint matrix A for linearized overlap constraints
        # For each pair (i,j): (x_i - x_j) >= r_i + r_j - lambda*(dist_ij)
        # Rearranged: (x_i - x_j) - (r_i + r_j) + lambda*dist_ij >= 0
        # As Ax <= b form: -(x_i - x_j) + (r_i + r_j) - lambda*dist_ij <= 0
        
        # Also boundary constraints:
        # 0 <= cx_i <= w  =>  -cx_i <= 0, cx_i <= w
        # 0 <= cy_i <= h  =>  -cy_i <= 0, cy_i <= h
        # 0 <= r_i <= r_max
        
        r_max = min(w, h) / 2.0

        # Count constraints: 2*P (overlaps) + 4*N (boundaries) = 2P + 4N
        n_cons = 2 * P + 4 * N
        A = np.zeros((n_cons, n))
        b = np.zeros(n_cons)

        # Overlap constraints (linearized)
        for k, (i, j) in enumerate(pairs):
            # Compute current distance
            dx0 = x[i] - x[N + j]
            dy0 = x[N + i] - x[N + j]
            d0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
            if d0 < 1e-12:
                d0 = 1e-12
            lambda_ij = 1.0  # Standard linearization parameter

            # Constraint: (cx_i - cx_j) - (r_i + r_j) + lambda*(dist_ij) >= 0
            # => -(cx_i - cx_j) + (r_i + r_j) - lambda*dist_ij <= 0
            # dist_ij = sqrt((cx_i-cx_j)^2 + (cy_i-cy_j)^2) is nonlinear
            # Linearize dist_ij around current point: dist_ij ≈ d0 + (dx_i-dx_j)/d0 * (cx_i-cx_j) + (dy_i-dy_j)/d0 * (cy_i-cy_j)
            # So: -(cx_i-cx_j) + (r_i+r_j) - lambda*(d0 + (dx0/d0)(cx_i-cx_j) + (dy0/d0)(cy_i-cy_j)) <= 0
            # Group terms:
            # cx_i: -1 - lambda*dx0/d0
            # cx_j: +1 + lambda*dx0/d0
            # cy_i: -lambda*dy0/d0
            # cy_j: +lambda*dy0/d0
            # r_i: +1
            # r_j: +1
            # constant: -lambda*d0

            A_row = 0
            A[A_row, i] = -1.0 - lambda_ij * dx0 / d0
            A[A_row, N + j] = 1.0 + lambda_ij * dx0 / d0
            A[A_row, N + i] = -lambda_ij * dy0 / d0
            A[A_row, N + j] += lambda_ij * dy0 / d0
            A[A_row, 2*N + i] = 1.0
            A[A_row, 2*N + j] = 1.0
            b[A_row] = -lambda_ij * d0
            A_row += 1

        # Boundary constraints: -cx_i <= 0
        for i in range(N):
            A[A_row, i] = -1.0
            b[A_row] = 0.0
            A_row += 1
        # cx_i <= w
        for i in range(N):
            A[A_row, i] = 1.0
            b[A_row] = w
            A_row += 1
        # -cy_i <= 0
        for i in range(N):
            A[A_row, N + i] = -1.0
            b[A_row] = 0.0
            A_row += 1
        # cy_i <= h
        for i in range(N):
            A[A_row, N + i] = 1.0
            b[A_row] = h
            A_row += 1

        # r_i >= 0 and r_i <= r_max are handled by bounds

        bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N

        # Use SLSQP which can handle linear constraints
        best_x = x.copy()
        best_obj = np.sum(x[2*N:])

        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Rebuild A and b with linearization at current x
            for k, (i, j) in enumerate(pairs):
                dx0 = x[i] - x[N + j]
                dy0 = x[N + i] - x[N + j]
                d0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
                if d0 < 1e-12:
                    d0 = 1e-12
                lambda_ij = 1.0

                A_row = k
                A[A_row, i] = -1.0 - lambda_ij * dx0 / d0
                A[A_row, N + j] = 1.0 + lambda_ij * dx0 / d0
                A[A_row, N + i] = -lambda_ij * dy0 / d0
                A[A_row, N + j] += lambda_ij * dy0 / d0
                A[A_row, 2*N + i] = 1.0
                A[A_row, 2*N + j] = 1.0
                b[A_row] = -lambda_ij * d0

            # Solve QP via SLSQP
            try:
                res = minimize(
                    lambda xx: -np.sum(xx[2*N:]),
                    lambda xx: -np.ones(n),
                    x,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': lambda xx, A=A, b=b: b - A @ xx}],
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.success or res.fun < best_obj - 1e-10:
                    x_new = res.x
                    change = np.linalg.norm(x_new - x)
                    x = x_new
                    best_obj = -res.fun
                    if change < tol:
                        break
                else:
                    break
            except Exception:
                break

        return x

    def refine_with_l_bfgs_b(x0, w, h):
        """Final polish using L-BFGS-B with moderate penalty."""
        r_max = min(w, h) / 2.0
        bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N

        def objective_and_grad(x):
            cx_v = x[:N]
            cy_v = x[N:2*N]
            r_v = x[2*N:]

            obj = -np.sum(r_v)
            grad = np.zeros(3*N)
            grad[2*N:] = -1.0

            # Overlap penalty
            dx = cx_v[:, None] - cx_v[None, :]
            dy = cy_v[:, None] - cy_v[None, :]
            dist_sq = dx * dx + dy * dy
            dist_sq[dist_sq < 1e-12] = 1e-12
            dist = np.sqrt(dist_sq)

            r_sum = r_v[:, None] + r_v[None, :]

            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            d_pairs = dist[mask]
            rs_pairs = r_sum[mask]

            viol = rs_pairs - d_pairs
            viol_pos = np.maximum(0, viol)

            penalty = 100.0
            obj += penalty * np.sum(viol_pos ** 2)

            ii, jj = np.where(mask)
            active = viol_pos > 0
            if len(active) > 0:
                ii_act = ii[active]
                jj_act = jj[active]
                viol_act = viol_pos[active]
                d_act = d_pairs[active]

                factor = -2.0 * penalty * viol_act / d_act
                dx_act = cx_v[ii_act] - cx_v[jj_act]
                dy_act = cy_v[ii_act] - cy_v[jj_act]

                np.add.at(grad, ii_act, factor * dx_act)
                np.add.at(grad, jj_act, -factor * dx_act)
                np.add.at(grad, N + ii_act, factor * dy_act)
                np.add.at(grad, N + jj_act, -factor * dy_act)

                np.add.at(grad, 2*N + ii_act, 2.0 * penalty * viol_act)
                np.add.at(grad, 2*N + jj_act, 2.0 * penalty * viol_act)

            # Boundary penalty
            boundary_viol = 0.0
            # Left wall: cx_i - r_i >= 0
            left_v = np.maximum(0, r_v - cx_v)
            boundary_viol += np.sum(left_v ** 2)
            # Right wall: w - cx_i - r_i >= 0
            right_v = np.maximum(0, cx_v + r_v - w)
            boundary_viol += np.sum(right_v ** 2)
            # Bottom wall: cy_i - r_i >= 0
            bottom_v = np.maximum(0, r_v - cy_v)
            boundary_viol += np.sum(bottom_v ** 2)
            # Top wall: h - cy_i - r_i >= 0
            top_v = np.maximum(0, cy_v + r_v - h)
            boundary_viol += np.sum(top_v ** 2)

            obj += 100.0 * boundary_viol

            # Gradient of boundary penalty
            np.add.at(grad, np.arange(N), -2.0 * 100.0 * left_v)
            np.add.at(grad, np.arange(N), 2.0 * 100.0 * right_v)
            np.add.at(grad, N + np.arange(N), -2.0 * 100.0 * bottom_v)
            np.add.at(grad, N + np.arange(N), 2.0 * 100.0 * top_v)
            np.add.at(grad, 2*N + np.arange(N), 2.0 * 100.0 * (left_v + bottom_v))
            np.add.at(grad, 2*N + np.arange(N), 2.0 * 100.0 * (right_v + top_v))

            return obj, grad

        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            return validate_and_score(res.x, w, h)
        except Exception:
            return -1, None

    # Generate initial configurations from hexagonal lattice
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.95, 1.05]

    for w_ratio in aspect_ratios:
        if time.time() > deadline - 15:
            break
        w = w_ratio
        h = 2.0 - w
        if h <= 0.1:
            continue

        for orientation in [0, 1]:
            if time.time() > deadline - 10:
                break

            # Generate hex lattice points in unit space
            pts = []
            for i in range(6):
                for j in range(6):
                    if orientation == 0:
                        x = i * math.sqrt(3) + (j % 2) * math.sqrt(3) / 2
                        y = j * 1.5
                    else:
                        x = i * 1.5
                        y = j * math.sqrt(3) + (i % 2) * math.sqrt(3) / 2
                    pts.append((x, y))

            pts = np.array(pts)
            pts_centered = pts - np.mean(pts, axis=0)

            # Select 21 closest to center
            dists = np.linalg.norm(pts_centered, axis=1)
            idx_sorted = np.argsort(dists)
            selected_idx = idx_sorted[:N]
            pts_sel = pts_centered[selected_idx]

            min_x, min_y = np.min(pts_sel, axis=0)
            max_x, max_y = np.max(pts_sel, axis=0)
            width_pts = max_x - min_x
            height_pts = max_y - min_y

            if width_pts < 1e-6 or height_pts < 1e-6:
                continue

            sx = w / width_pts
            sy = h / height_pts
            s = min(sx, sy) * 0.9

            cx_init = pts_sel[:, 0] * s + w / 2
            cy_init = pts_sel[:, 1] * s + h / 2
            r_init = (math.sqrt(3) / 2) * s * 0.9

            x0 = np.concatenate([cx_init, cy_init, np.full(N, r_init)])

            # Phase 1: SCR to find a good feasible point
            if time.time() > deadline - 20:
                break
            x_scr = solve_scr_qp(x0, w, h, max_iter=25)

            # Phase 2: Polish with L-BFGS-B
            score, res = refine_with_l_bfgs_b(x_scr, w, h)

            if score > best_score:
                best_score = score
                best_result = res

    # Fallback
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
