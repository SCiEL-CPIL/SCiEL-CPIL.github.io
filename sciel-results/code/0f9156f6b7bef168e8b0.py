import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    best_radius = 0.0
    best_centers = None

    def compute_radius(centers):
        """Compute the largest common radius for given centers."""
        if len(centers) < 2:
            return float(np.min(0.5 - np.abs(centers)))
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        dist_to_walls = 0.5 - np.abs(centers)
        min_wall_dist = np.min(dist_to_walls)
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def _soft_min_neg(x, beta):
        """
        Soft approximation of -min(x) using log-sum-exp.
        -min(x) ≈ (1/beta) * log(sum(exp(-beta*x)))
        Gradient: -exp(-beta*x) / sum(exp(-beta*x))
        """
        x = np.asarray(x)
        # Numerical stability: subtract max of -beta*x = -beta*min(x)
        m = -beta * np.min(x)
        ex = np.exp(-beta * x - m)
        s = np.sum(ex)
        val = (m + np.log(s)) / beta
        grad = -ex / s
        return val, grad

    def soft_min_obj_and_grad(centers_flat, beta):
        """
        Objective: minimize -soft_min(all distances and wall distances).
        This is equivalent to maximizing the soft-min of all constraints.
        """
        centers = centers_flat.reshape(N, 3)
        # Clip to ensure we stay in valid region for gradient computation
        centers = np.clip(centers, -0.4999, 0.4999)

        # Pairwise distances (only upper triangle to save memory)
        # But for simplicity and gradient, let's use full matrix and mask diagonal
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal with large value so they don't affect soft-min
        np.fill_diagonal(dists, 100.0)

        # Wall distances
        wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
        wall_dists_flat = wall_dists.flatten()

        # All constraints: pairwise dists / 2 (radius) and wall dists (radius)
        # We want to maximize min(all constraints), so minimize -soft_min(all constraints)
        # Pairwise contribute dist/2, walls contribute dist
        pair_contrib = dists / 2.0  # shape (N, N)
        all_vals = np.concatenate([pair_contrib.flatten(), wall_dists_flat])

        obj, grad = _soft_min_neg(all_vals, beta)

        # Build gradient with respect to centers
        grad_centers = np.zeros((N, 3))

        # Pairwise part
        pair_grad = grad[:N * N].reshape(N, N)
        # d(dists[i,j])/d(centers[i]) = (centers[i] - centers[j]) / dists[i,j]
        # d(dists[i,j]/2)/d(centers[i]) = (centers[i] - centers[j]) / (2 * dists[i,j])
        # Sum over j
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # pair_grad[i, j] is d(obj)/d(pair_contrib[i,j])
                # pair_contrib[i,j] = dists[i,j] / 2
                # d(pair_contrib[i,j])/d(centers[i]) = (centers[i] - centers[j]) / (2*d)
                grad_centers[i] += pair_grad[i, j] * (centers[i] - centers[j]) / (2.0 * d)
                # Also pair_grad[j, i] contributes to centers[j]
                # But we're summing over j for fixed i, so pair_grad[j,i] will be handled when i=j loop
                # Actually, we need to be careful. Let me restructure.

        # Let me redo the gradient computation more carefully
        grad_centers = np.zeros((N, 3))

        # For pairwise: obj depends on dists[i,j]/2 for all i,j (i!=j)
        # d(obj)/d(centers[i]) = sum_{j!=i} grad_pair[i,j] * d(dists[i,j]/2)/d(centers[i])
        #                        + sum_{j!=i} grad_pair[j,i] * d(dists[j,i]/2)/d(centers[i])
        # d(dists[i,j]/2)/d(centers[i]) = (centers[i] - centers[j]) / (2 * dists[i,j])
        # d(dists[j,i]/2)/d(centers[i]) = (centers[j] - centers[i]) / (2 * dists[j,i]) = -(centers[i] - centers[j]) / (2 * dists[i,j])

        # So the contribution from pair (i,j) to grad_centers[i] is:
        # grad_pair[i,j] * (centers[i]-centers[j])/(2*d) + grad_pair[j,i] * (-(centers[i]-centers[j])/(2*d))
        # = (grad_pair[i,j] - grad_pair[j,i]) * (centers[i]-centers[j])/(2*d)

        # Since dists is symmetric and our soft-min treats all values equally,
        # grad_pair should be symmetric: grad_pair[i,j] = grad_pair[j,i]
        # So this would be zero? That can't be right.

        # Let me think again. The objective is:
        # f = (1/beta) * log(sum_{all k} exp(-beta * v_k))
        # where v_k are the constraint values.
        # df/dv_k = -exp(-beta*v_k) / sum(exp(-beta*v_k)) = -p_k where p_k = exp(-beta*v_k)/sum
        # So grad = -p where p is the softmax distribution.

        # For pairwise constraint v_{ij} = dists[i,j]/2 (for i<j, but we include both i,j and j,i)
        # Actually, in our formulation, we have all N*N pairwise entries (with diagonal masked).
        # v_{ij} = dists[i,j]/2 for i != j
        # dv_{ij}/d(centers[i]) = (centers[i] - centers[j]) / (2 * dists[i,j])
        # dv_{ij}/d(centers[j]) = (centers[j] - centers[i]) / (2 * dists[i,j])

        # So: d(f)/d(centers[i]) = sum_{j!=i} (-p_{ij}) * (centers[i]-centers[j])/(2*d_{ij})
        #                         + sum_{j!=i} (-p_{ji}) * (centers[j]-centers[i])/(2*d_{ji})
        #                         = sum_{j!=i} -p_{ij} * (centers[i]-centers[j])/(2*d_{ij})
        #                         + sum_{j!=i} p_{ji} * (centers[i]-centers[j])/(2*d_{ij})
        # = sum_{j!=i} (p_{ji} - p_{ij}) * (centers[i]-centers[j])/(2*d_{ij})

        # Since p_{ij} = p_{ji} (because v_{ij} = v_{ji}), this is zero!
        # This means the pairwise part doesn't contribute to the gradient?
        # That seems wrong. Let me check.

        # Actually, I think the issue is that when I flatten dists/2, I'm including both (i,j) and (j,i).
        # But v_{ij} = v_{ji}, so they have the same value and same probability.
        # The gradient contribution from (i,j) to center i is -p_{ij} * (c_i - c_j)/(2d)
        # The gradient contribution from (j,i) to center i is -p_{ji} * (c_j - c_i)/(2d) = p_{ji} * (c_i - c_j)/(2d)
        # Sum: (-p_{ij} + p_{ji}) * (c_i - c_j)/(2d) = 0 since p_{ij} = p_{ji}

        # This is a problem. The pairwise distances are symmetric, so including both (i,j) and (j,i)
        # cancels out the gradient. I should only include upper triangle.

        # Let me redo this with only upper triangle pairwise distances.
        grad_centers = np.zeros((N, 3))

        # Upper triangle pairwise distances
        idx_i, idx_j = np.triu_indices(N, k=1)
        pair_dists = dists[idx_i, idx_j]  # shape (M,)
        pair_contrib = pair_dists / 2.0  # shape (M,)

        # Wall distances
        wall_dists_flat = wall_dists.flatten()  # shape (3N,)

        # All constraint values
        all_vals = np.concatenate([pair_contrib, wall_dists_flat])

        obj, grad_all = _soft_min_neg(all_vals, beta)

        # Pairwise gradient
        M = len(pair_contrib)
        grad_pair = grad_all[:M]  # shape (M,)

        for k in range(M):
            i, j = idx_i[k], idx_j[k]
            d = pair_dists[k]
            if d < 1e-10:
                continue
            g = grad_pair[k]
            # d(v_k)/d(centers[i]) = (centers[i] - centers[j]) / (2*d)
            # d(v_k)/d(centers[j]) = (centers[j] - centers[i]) / (2*d)
            grad_centers[i] += g * (centers[i] - centers[j]) / (2.0 * d)
            grad_centers[j] += g * (centers[j] - centers[i]) / (2.0 * d)

        # Wall gradient
        grad_wall = grad_all[M:]  # shape (3N,)
        # wall_dists[i, k] = 0.5 - |centers[i, k]|
        # d(wall_dists[i,k])/d(centers[i,k]) = -sign(centers[i,k])
        for i in range(N):
            for k in range(3):
                idx = i * 3 + k
                g = grad_wall[idx]
                c = centers[i, k]
                if abs(c) < 1e-10:
                    # At zero, subgradient is in [-1, 1], use 0
                    grad_centers[i, k] += g * 0.0
                else:
                    grad_centers[i, k] += g * (-np.sign(c))

        return obj, grad_centers.flatten()

    def soft_min_obj_only(centers_flat, beta):
        return soft_min_obj_and_grad(centers_flat, beta)[0]

    def soft_min_grad_only(centers_flat, beta):
        return soft_min_obj_and_grad(centers_flat, beta)[1]

    def run_l_bfgs(centers_init, beta, time_limit=10.0):
        """Run L-BFGS-B to minimize the soft-min objective."""
        centers_init = np.clip(centers_init, -0.4999, 0.4999)
        x0 = centers_init.flatten()

        # Bounds: centers in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)

        def obj(x):
            return soft_min_obj_only(x, beta)

        def grad(x):
            return soft_min_grad_only(x, beta)

        start_t = time.time()
        max_iter = 500

        # Custom callback to check time
        stop_flag = [False]
        def callback(xk):
            if time.time() > start_t + time_limit:
                stop_flag[0] = True
            return stop_flag[0]

        try:
            res = minimize(
                obj, x0, jac=grad, method='L-BFGS-B',
                bounds=bounds, options={
                    'maxiter': max_iter,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50,
                    'callback': callback,
                    'disp': False
                }
            )
            best_x = res.x
        except Exception:
            best_x = x0

        best_centers = best_x.reshape(N, 3)
        best_centers = np.clip(best_centers, -0.5, 0.5)
        return best_centers

    def high_precision_polish(centers, time_limit=10.0):
        """
        High-precision polish using very high beta (approaching true min)
        and tight tolerances.
        """
        centers = np.clip(centers, -0.4999, 0.4999)
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)

        # Use very high beta for near-exact min
        beta = 200.0

        def obj(x):
            return soft_min_obj_only(x, beta)

        def grad(x):
            return soft_min_grad_only(x, beta)

        start_t = time.time()

        try:
            res = minimize(
                obj, x0, jac=grad, method='L-BFGS-B',
                bounds=bounds, options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'disp': False
                }
            )
            if time.time() > start_t + time_limit:
                # Timeout, return current best
                best_x = x0
            else:
                best_x = res.x
        except Exception:
            best_x = x0

        best_centers = best_x.reshape(N, 3)
        best_centers = np.clip(best_centers, -0.5, 0.5)
        return best_centers

    # --- Main Logic ---

    # 1. Generate initial candidates
    candidates = []

    # Grid 5x5x4
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])

    # Grid 4x5x5
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])

    # Random
    np.random.seed(42)
    candidates.append(np.random.uniform(-0.45, 0.45, (N, 3)))

    # Jittered Grid
    base_grid = candidates[0].copy()
    np.random.seed(123)
    jitter = np.random.normal(0, 0.01, (N, 3))
    candidates.append(np.clip(base_grid + jitter, -0.5, 0.5))

    # 2. Initial optimization with moderate beta
    for i, init_centers in enumerate(candidates):
        if time.time() > deadline - 60.0:
            break

        opt_centers = run_l_bfgs(init_centers, beta=20.0, time_limit=15.0)
        r = compute_radius(opt_centers)

        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()

    # 3. Iterative refinement: increase beta and re-optimize
    current_best_centers = best_centers.copy() if best_centers is not None else candidates[0]
    current_best_radius = best_radius

    if current_best_radius == 0.0:
        current_best_centers = candidates[0]
        current_best_radius = compute_radius(current_best_centers)

    # Progressive beta increase for higher precision
    betas = [30.0, 50.0, 80.0, 120.0, 180.0]
    for beta in betas:
        if time.time() > deadline - 15.0:
            break

        time_left = deadline - time.time() - 10.0
        if time_left < 3.0:
            break

        opt_centers = run_l_bfgs(current_best_centers, beta=beta, time_limit=time_left * 0.8)
        r = compute_radius(opt_centers)

        if r > current_best_radius:
            current_best_radius = r
            current_best_centers = opt_centers.copy()

    # 4. Final high-precision polish
    if time.time() < deadline - 3.0 and current_best_centers is not None:
        time_left = deadline - time.time() - 2.0
        final_centers = high_precision_polish(current_best_centers, time_limit=time_left)
        r_final = compute_radius(final_centers)
        if r_final > current_best_radius:
            current_best_radius = r_final
            current_best_centers = final_centers

    if current_best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        current_best_centers = grid[:N]

    return current_best_centers
