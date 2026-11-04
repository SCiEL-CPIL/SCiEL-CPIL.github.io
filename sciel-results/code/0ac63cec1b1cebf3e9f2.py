import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin

    best_result = None
    best_score = -np.inf

    # Adaptive smooth-max: start coarse, tighten over iterations
    _SMOOTH_EPS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    def _check_validity(circles, W, H):
        """Check if circles are valid: inside rectangle, no overlaps."""
        if len(circles) != N:
            return False
        for i in range(N):
            x, y, r = circles[i]
            if r <= 0:
                return False
            if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return False
        return True

    def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
        """
        Directly resolve overlaps and boundary violations using pairwise repulsion.
        Returns (circles, is_valid).
        """
        n = len(circles)
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()

        for it in range(max_iter):
            moved = False

            # Resolve boundary violations by clamping
            for i in range(n):
                if r[i] <= 0:
                    r[i] = 1e-4
                x[i] = np.clip(x[i], r[i], W - r[i])
                y[i] = np.clip(y[i], r[i], H - r[i])

            # Resolve overlaps via pairwise separation
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    min_dist = r[i] + r[j]
                    if dist_sq < (min_dist - tol) ** 2:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-12:
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle) * 1e-6
                            dy = np.sin(angle) * 1e-6
                            dist = 1e-6
                        overlap = min_dist - dist
                        total_r = r[i] + r[j]
                        wi = r[i] / total_r if total_r > 0 else 0.5
                        wj = r[j] / total_r if total_r > 0 else 0.5
                        push = overlap + 1e-8
                        x[i] += wi * push * dx / dist
                        y[i] += wi * push * dy / dist
                        x[j] -= wj * push * dx / dist
                        y[j] -= wj * push * dy / dist
                        moved = True

            # Re-clamp after separation, shrinking radius if circle doesn't fit
            for i in range(n):
                max_r_x = min(x[i], W - x[i])
                max_r_y = min(y[i], H - y[i])
                max_r = max(1e-4, min(max_r_x, max_r_y))
                if r[i] > max_r:
                    r[i] = max_r
                x[i] = np.clip(x[i], r[i], W - r[i])
                y[i] = np.clip(y[i], r[i], H - r[i])

            if not moved:
                break

        result = np.column_stack([x, y, r])
        valid = _check_validity(result, W, H)
        return result, valid

    def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
        """
        Objective: minimize -sum(r) + penalties for violations.
        Returns (obj, grad) with analytical gradient.
        """
        n = N
        circles = params.reshape(n, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]

        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0

        def _smooth_max_grad(v):
            smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
            grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
            return smax, grad_smax

        # x - r >= 0  => violation = r - x
        v = r - x
        smax, gsmax = _smooth_max_grad(v)
        obj += boundary_weight * np.sum(smax ** 2)
        grad[0::3] += -2.0 * boundary_weight * smax * gsmax
        grad[2::3] += 2.0 * boundary_weight * smax * gsmax

        # y - r >= 0
        v = r - y
        smax, gsmax = _smooth_max_grad(v)
        obj += boundary_weight * np.sum(smax ** 2)
        grad[1::3] += -2.0 * boundary_weight * smax * gsmax
        grad[2::3] += 2.0 * boundary_weight * smax * gsmax

        # x + r <= W
        v = x + r - W
        smax, gsmax = _smooth_max_grad(v)
        obj += boundary_weight * np.sum(smax ** 2)
        grad[0::3] += 2.0 * boundary_weight * smax * gsmax
        grad[2::3] += 2.0 * boundary_weight * smax * gsmax

        # y + r <= H
        v = y + r - H
        smax, gsmax = _smooth_max_grad(v)
        obj += boundary_weight * np.sum(smax ** 2)
        grad[1::3] += 2.0 * boundary_weight * smax * gsmax
        grad[2::3] += 2.0 * boundary_weight * smax * gsmax

        # r >= 0
        v = -r
        smax, gsmax = _smooth_max_grad(v)
        obj += boundary_weight * np.sum(smax ** 2)
        grad[2::3] += -2.0 * boundary_weight * smax * gsmax

        # Overlap penalties: vectorized
        dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
        dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
        dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
        np.fill_diagonal(dist_mat, 1e12)
        dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

        iu, ju = np.triu_indices(n, k=1)
        dist_ij = dist_mat_safe[iu, ju]
        dx_ij = dx_mat[iu, ju]
        dy_ij = dy_mat[iu, ju]
        r_sum = r[iu] + r[ju]

        v_vec = r_sum - dist_ij
        smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
        obj += overlap_weight * np.sum(smax_vec ** 2)

        factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
        nx_ij = -dx_ij / dist_ij
        ny_ij = -dy_ij / dist_ij

        grad_x_i = factor_vec * nx_ij
        grad_y_i = factor_vec * ny_ij
        grad_x_j = -factor_vec * nx_ij
        grad_y_j = -factor_vec * ny_ij

        np.add.at(grad, 3 * iu, grad_x_i)
        np.add.at(grad, 3 * iu + 1, grad_y_i)
        np.add.at(grad, 3 * iu + 2, factor_vec)
        np.add.at(grad, 3 * ju, grad_x_j)
        np.add.at(grad, 3 * ju + 1, grad_y_j)
        np.add.at(grad, 3 * ju + 2, factor_vec)

        return obj, grad

    def _local_optimize(circles, W, H, max_iter=3000):
        """Use L-BFGS-B with analytical gradients to locally optimize."""
        params = circles.flatten()
        r_max = min(W, H) / 2
        bounds = []
        for i in range(N):
            bounds.append((1e-6, W - 1e-6))
            bounds.append((1e-6, H - 1e-6))
            bounds.append((1e-6, r_max))

        result = minimize(
            _objective_and_grad, params, args=(W, H, 1e-4, 200.0, 200.0),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
        )
        return result.x.reshape(N, 3)

    def _polish(circles, W, H, max_iter=8000):
        """Multi-pass polish with progressively tighter smooth-max eps."""
        params = circles.flatten()
        r_max = min(W, H) / 2
        bounds = []
        for i in range(N):
            bounds.append((1e-6, W - 1e-6))
            bounds.append((1e-6, H - 1e-6))
            bounds.append((1e-6, r_max))

        # Progressive refinement: 5 passes with decreasing eps
        eps_schedule = _SMOOTH_EPS
        n_passes = len(eps_schedule)
        iter_per_pass = max(500, max_iter // n_passes)

        for p in range(n_passes):
            eps_val = eps_schedule[p]
            result = minimize(
                _objective_and_grad, params, args=(W, H, eps_val, 200.0, 200.0),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': iter_per_pass, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
            )
            params = result.x

        return result.x.reshape(N, 3)

    def _high_precision_polish(circles, W, H, max_iter=12000):
        """Ultra-tight final polish for the best candidate."""
        params = circles.flatten()
        r_max = min(W, H) / 2
        bounds = []
        for i in range(N):
            bounds.append((1e-6, W - 1e-6))
            bounds.append((1e-6, H - 1e-6))
            bounds.append((1e-6, r_max))

        # Final pass with very tight eps
        result = minimize(
            _objective_and_grad, params, args=(W, H, 1e-7, 500.0, 500.0),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
        )
        return result.x.reshape(N, 3)

    # Define the optimization loop
    for restart in range(50):
        if time.time() - start_time > timeout:
            break

        # Initialization
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        # 5x5 grid = 25 cells, use 21
        # Or 4x6 = 24 cells
        # Let's use a 5x5 grid, drop 4 random ones? Or just use first 21.

        # Better: Random initialization within the box
        W_init = 1.0
        H_init = 1.0

        # Random positions
        x0 = np.random.rand(N) * W_init
        y0 = np.random.rand(N) * H_init

        # Initial radii: small, or based on grid
        # Let's use a grid-based initial radius
        # Approximate radius for N circles in 1x1 square is ~ 1/sqrt(N)
        r0 = np.ones(N) * (0.5 / np.sqrt(N))

        # Perturb
        x0 += np.random.normal(0, 0.01, N)
        y0 += np.random.normal(0, 0.01, N)
        r0 += np.random.normal(0, 0.01, N)
        r0 = np.maximum(r0, 0.01)

        # Clip to box
        x0 = np.clip(x0, r0, W_init - r0)
        y0 = np.clip(y0, r0, H_init - r0)

        circles0 = np.column_stack([x0, y0, r0])

        # First, separate circles to get a valid starting point
        circles0, valid0 = _separate_circles(circles0, W_init, H_init)

        # Optimize
        try:
            # Local optimization
            circles1 = _local_optimize(circles0, W_init, H_init)
            
            # Polish
            circles2 = _polish(circles1, W_init, H_init)
            
            # High precision polish
            circles3 = _high_precision_polish(circles2, W_init, H_init)
            
            # Check validity
            if _check_validity(circles3, W_init, H_init):
                score = np.sum(circles3[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circles3.copy()
                    
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
