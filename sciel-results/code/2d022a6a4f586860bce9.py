# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin (150s limit)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=100.0):
        """Smooth objective using soft-min via log-sum-exp for gradient-based methods."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def smooth_objective_grad(params, alpha=100.0):
        """Smooth objective with analytical gradient for L-BFGS-B."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        n = len(x)

        # Compute distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        f = -np.sum(radii_soft)

        # Gradient computation
        grad = np.zeros(2 * n)

        for i in range(n):
            # Boundary gradient
            b_exp_sum = np.sum(exp_b[i, :])
            d_dxi = 0.0
            d_dyi = 0.0
            if b_exp_sum > 0:
                d_dxi = -(1/alpha) * (exp_b[i, 0] * (-alpha) + exp_b[i, 1] * alpha) / b_exp_sum
                d_dyi = -(1/alpha) * (exp_b[i, 2] * (-alpha) + exp_b[i, 3] * alpha) / b_exp_sum

            # Distance gradient
            d_exp_sum = np.sum(exp_scaled[i, :])
            if d_exp_sum > 0:
                # Vectorized gradient calculation for distances
                # d(dist[i,j])/dx[i] = dx[i,j] / dist[i,j]
                # We need sum_j exp_scaled[i,j] * (-alpha * dx[i,j]/dist[i,j]) / d_exp_sum
                # Note: dist[i,i] is inf, exp is 0, so it's safe.
                # But dist[i,j] can be 0 if overlapping? No, we have soft min.
                # If dist is very small, gradient is large.
                
                # Avoid division by zero
                safe_dist = np.where(dist_matrix[i, :] > 1e-10, dist_matrix[i, :], 1.0)
                d_dx = dx[i, :] / safe_dist
                d_dy = dy[i, :] / safe_dist
                np.fill_diagonal(d_dx, 0.0)
                np.fill_diagonal(d_dy, 0.0)
                
                d_dxi += -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dx)) / d_exp_sum
                d_dyi += -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dy)) / d_exp_sum

            # Combine
            ab_exp_sum = np.sum(exp_ab[i, :])
            if ab_exp_sum > 0 and radii_soft[i] > 0:
                # dr/dx = (exp_ab[0] * d(-alpha*a)/dx + exp_ab[1] * d(-alpha*b)/dx) / ab_exp_sum * (-1/alpha)
                # d(-alpha*a)/dx = -alpha * d_dxi
                # d(-alpha*b)/dx = -alpha * 0.5 * d_dxi (since b = min_dist/2)
                
                dr_dxi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * d_dxi) + exp_ab[i, 1] * (-alpha * 0.5 * d_dxi)) / ab_exp_sum
                dr_dyi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * d_dyi) + exp_ab[i, 1] * (-alpha * 0.5 * d_dyi)) / ab_exp_sum
                
                grad[2*i] = -dr_dxi
                grad[2*i+1] = -dr_dyi

        return f, grad

    def push_apart(params, iterations=50):
        """Push overlapping circles apart to improve validity."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        radii = compute_radii(x, y)

        for _ in range(iterations):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist and dist > 1e-10:
                        push = (min_dist - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        x[i] -= push * nx
                        y[i] -= push * ny
                        x[j] += push * nx
                        y[j] += push * ny
                        moved = True
            if not moved:
                break

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)

        new_params = np.zeros(2 * N)
        new_params[0::2] = x
        new_params[1::2] = y
        return new_params

    def random_init():
        """Random initialization with some spacing."""
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Strategy: Multi-stage Multi-Start
    # Stage 1: Coarse search with many random starts
    # Stage 2: Medium search on top candidates
    # Stage 3: Fine search on top candidates
    # Stage 4: Powell polish

    np.random.seed(42)
    
    # Stage 1: 50 random starts, alpha=20, maxiter=200
    candidates = []
    for i in range(50):
        if time.time() > deadline - 90:
            break
        p0 = random_init()
        try:
            res = minimize(
                lambda p: smooth_objective_grad(p, alpha=20.0), 
                p0, 
                method='L-BFGS-B', 
                bounds=bounds,
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            # Evaluate true objective
            obj = objective(res.x)
            candidates.append((obj, res.x))
        except Exception:
            pass
    
    # Sort and keep top 10
    candidates.sort(key=lambda x: x[0])
    top_candidates = [c[1] for c in candidates[:10]]
    
    best_params = top_candidates[0] if top_candidates else random_init()
    best_obj = objective(best_params)
    
    # Stage 2: 10 starts, alpha=100, maxiter=500
    for p0 in top_candidates:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                lambda p: smooth_objective_grad(p, alpha=100.0), 
                p0, 
                method='L-BFGS-B', 
                bounds=bounds,
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            obj = objective(res.x)
            if obj < best_obj:
                best_params = res.x
                best_obj = obj
        except Exception:
            pass

    # Stage 3: 5 starts (best 5 from stage 2? we just kept best so far, let's restart from best few)
    # Actually, let's just take the best 5 from the pool of Stage 2 results if we stored them.
    # For simplicity, let's just run Stage 3 on the current best and a few jittered versions.
    
    stage3_starts = [best_params]
    for _ in range(4):
        jittered = best_params + np.random.randn(2*N) * 0.02
        jittered = np.clip(jittered, 0.01, 0.99)
        stage3_starts.append(jittered)
        
    for p0 in stage3_starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                lambda p: smooth_objective_grad(p, alpha=400.0), 
                p0, 
                method='L-BFGS-B', 
                bounds=bounds,
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            obj = objective(res.x)
            if obj < best_obj:
                best_params = res.x
                best_obj = obj
        except Exception:
            pass

    # Stage 4: Powell polish
    if time.time() < deadline - 10:
        try:
            cleaned = push_apart(best_params, iterations=50)
            clean_obj = objective(cleaned)
            if clean_obj < best_obj:
                best_params = cleaned
                best_obj = clean_obj

            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Final cleanup
    if time.time() < deadline - 3:
        try:
            final = push_apart(best_params, iterations=100)
            final_obj = objective(final)
            if final_obj < best_obj:
                best_params = final
                best_obj = final_obj
        except Exception:
            pass

    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)
    radii *= 0.99999

    for i in range(N):
        if x[i] < radii[i]:
            x[i] = radii[i]
        if x[i] > 1 - radii[i]:
            x[i] = 1 - radii[i]
        if y[i] < radii[i]:
            y[i] = radii[i]
        if y[i] > 1 - radii[i]:
            y[i] = 1 - radii[i]

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
