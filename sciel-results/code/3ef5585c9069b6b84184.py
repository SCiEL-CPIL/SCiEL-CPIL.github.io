import numpy as np
import time
import scipy.optimize

N = 50

# Ring counts that sum to 50 (1 center + rings). These are discrete structural choices.
RING_COUNTS = [1, 6, 12, 17, 14]  # sums to 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_score = -np.inf
    best_points = None

    # Precompute ring indices: for each circle, which ring it belongs to
    ring_of = np.zeros(N, dtype=int)
    pos_in_ring = np.zeros(N, dtype=int)
    idx = 0
    for ri, cnt in enumerate(RING_COUNTS):
        for k in range(cnt):
            ring_of[idx] = ri
            pos_in_ring[idx] = k
            idx += 1
    n_rings = len(RING_COUNTS)

    # Per-ring base angular offset: ring 0 (center) has none; others evenly spaced
    # We'll optimize a global angular offset per ring (theta_r) plus radius per ring (R_r).
    # Ring 0 is the center: R_0 = 0 fixed, theta_0 unused.

    def params_to_points(params):
        """
        params: length (2 * n_rings) vector.
        First n_rings entries: ring radii R_r (R_0 forced to 0).
        Next n_rings entries: ring angular offsets theta_r.
        """
        R = params[:n_rings].copy()
        theta = params[n_rings:]
        R[0] = 0.0  # center ring
        # Clamp radii to stay inside unit circle with margin
        R = np.clip(R, 0.0, 0.99)
        pts = np.zeros((N, 2))
        for ri in range(n_rings):
            cnt = RING_COUNTS[ri]
            if cnt == 0:
                continue
            # angles for this ring: evenly spaced + offset
            angles = theta[ri] + 2.0 * np.pi * np.arange(cnt) / cnt
            rr = R[ri]
            for k in range(cnt):
                i = np.where((ring_of == ri) & (pos_in_ring == k))[0][0]
                pts[i, 0] = rr * np.cos(angles[k])
                pts[i, 1] = rr * np.sin(angles[k])
        return pts

    def get_score(points):
        """Compute the true score (minimum radius)."""
        if points.shape != (N, 2):
            return -np.inf
        if not np.all(np.isfinite(points)):
            return -np.inf

        norms = np.linalg.norm(points, axis=1)
        if np.any(norms >= 1.0):
            return -np.inf

        d_bound = 1.0 - norms

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq)
        dists[~mask] = np.inf
        min_pair = np.min(dists[mask])

        return min(np.min(d_bound), min_pair) / 2.0

    def objective_and_grad(params, alpha):
        """
        Smoothed objective over the reduced parameter space:
        -soft_min over (boundary distances, pairwise distances).
        Returns (value, gradient w.r.t. params).
        """
        points = params_to_points(params)
        R = params[:n_rings].copy()
        theta = params[n_rings:]
        R[0] = 0.0
        R = np.clip(R, 0.0, 0.99)

        # Boundary distances
        norms = np.linalg.norm(points, axis=1) + 1e-15
        d_bound = 1.0 - norms

        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq) + 1e-15
        dists[~mask] = np.inf

        all_dists = np.concatenate([d_bound, dists[mask]])

        # Log-sum-exp stability
        min_d = np.min(all_dists)
        shifted = -alpha * (all_dists - min_d)
        sum_exp = np.sum(np.exp(shifted))
        weights = np.exp(shifted) / sum_exp

        # soft_min = min_d - (1/alpha) log(sum_exp)
        # We minimize -soft_min
        obj_val = -(min_d - (1.0 / alpha) * np.log(sum_exp))

        # Gradient w.r.t. params via chain rule.
        # d(params) contributions:
        #   d/dR_r and d/dtheta_r of each distance.
        grad = np.zeros_like(params)

        # We need d(points)/dR_r and d(points)/dtheta_r for each circle i.
        # point i: x_i = R_{r(i)} cos(a_i), y_i = R_{r(i)} sin(a_i), a_i = theta_{r(i)} + 2pi k_i / cnt_{r(i)}
        # dx_i/dR_r = cos(a_i) if r == r(i) else 0
        # dx_i/dtheta_r = -R_r sin(a_i) if r == r(i) else 0
        # dy_i/dR_r = sin(a_i)
        # dy_i/dtheta_r = R_r cos(a_i)

        # Precompute per-circle cos/sin and ring
        cos_a = np.cos(theta[ring_of] + 2.0 * np.pi * pos_in_ring / np.maximum(1, RING_COUNTS[ring_of]))
        sin_a = np.sin(theta[ring_of] + 2.0 * np.pi * pos_in_ring / np.maximum(1, RING_COUNTS[ring_of]))

        # Boundary gradient: d(d_bound_i)/dR_r = -x_i/R_i * dR_i/dR_r ... actually d_bound_i = 1 - |p_i|
        # d|p_i|/dR_r = (x_i dx_i/dR_r + y_i dy_i/dR_r)/|p_i|
        w_bound = weights[:N]
        for ri in range(n_rings):
            cnt = RING_COUNTS[ri]
            if cnt == 0:
                continue
            rr = R[ri]
            if rr < 1e-9:
                # center: boundary distance = 1, gradient w.r.t R_0 is 0 (R_0 fixed), skip
                continue
            # circles in this ring
            sel = (ring_of == ri)
            ca = cos_a[sel]
            sa = sin_a[sel]
            # d|p_i|/dR_ri = (x_i cos + y_i sin)/|p_i| = (rr cos^2 + rr sin^2)/rr = 1
            # So d_bound_i/dR_ri = -1 for all circles in ring ri
            grad[ri] += -np.sum(w_bound[sel]) * (-1.0)  # d/dR of -soft_min: see below

        # Let's recompute gradient more carefully with the chain rule for -soft_min.
        # obj = -soft_min = -(min_d - (1/alpha) log sum exp(-alpha d_k))
        # d obj / d param = - d(soft_min)/d param
        # d(soft_min)/d param = - (1/alpha) * sum_k weights_k * (d/d param)(-alpha d_k)
        #                      = sum_k weights_k * d(d_k)/d param
        # So d obj/d param = - sum_k weights_k * d(d_k)/d param
        # where d_k are the distances (boundary and pairwise).

        grad = np.zeros_like(params)

        # Boundary: d_k = d_bound_i = 1 - |p_i|. d(d_bound_i)/dR_ri = -1 (for circles in ring ri, rr>0).
        # d(d_bound_i)/dtheta_ri = 0 (|p_i| independent of angle).
        w_bound = weights[:N]
        for ri in range(n_rings):
            cnt = RING_COUNTS[ri]
            if cnt == 0:
                continue
            rr = R[ri]
            if rr < 1e-9:
                continue
            sel = (ring_of == ri)
            # d obj/dR_ri += - sum_{i in ring ri} w_bound_i * d(d_bound_i)/dR_ri = - sum w * (-1) = + sum w
            grad[ri] += np.sum(w_bound[sel])
            # theta: 0

        # Pairwise: d_k = dist_ij = |p_i - p_j|.
        # d obj/d param += - sum_{pairs} w_pair * d(dist_ij)/d param
        # d(dist_ij)/dR_r = ( (p_i - p_j) . d(p_i - p_j)/dR_r ) / dist_ij
        # d(p_i - p_j)/dR_r = (dp_i/dR_r - dp_j/dR_r)
        # dp_i/dR_r = (cos a_i, sin a_i) if r(i)==r else 0
        w_pair = weights[N:]
        iu = np.triu_indices(N, 1)
        i_idx, j_idx = iu
        diff_pairs = points[i_idx] - points[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]

        # For each pair, contribution to grad:
        # d(dist)/dR_r = dir_vec . (dp_i/dR_r - dp_j/dR_r)
        # d(dist)/dtheta_r = dir_vec . (dp_i/dtheta_r - dp_j/dtheta_r)
        # d obj/dR_r = - sum_pairs w * d(dist)/dR_r
        # d obj/dtheta_r = - sum_pairs w * d(dist)/dtheta_r

        # Vectorize over pairs:
        # dp_i/dR_r for pair: cos/sin of i if ring matches
        ci = cos_a[i_idx]; si = sin_a[i_idx]
        cj = cos_a[j_idx]; sj = sin_a[j_idx]
        ri_i = ring_of[i_idx]; ri_j = ring_of[j_idx]
        Ri = R[ri_i]; Rj = R[ri_j]

        # dR contribution: for each pair, for ring r, dp_i/dR_r - dp_j/dR_r =
        #   (ci, si) if ri_i == r, minus (cj, sj) if ri_j == r
        # We accumulate per ring.
        for r in range(n_rings):
            # pairs where i in ring r
            m_i = (ri_i == r)
            m_j = (ri_j == r)
            # dp_i/dR_r - dp_j/dR_r:
            # x-comp: ci[m_i] - cj[m_j]  (only where applicable)
            dx_dr = np.zeros(len(i_idx))
            dy_dr = np.zeros(len(i_idx))
            dx_dr[m_i] += ci[m_i]
            dy_dr[m_i] += si[m_i]
            dx_dr[m_j] -= cj[m_j]
            dy_dr[m_j] -= sj[m_j]
            ddist_dr = dir_vecs[:, 0] * dx_dr + dir_vecs[:, 1] * dy_dr
            grad[r] += -np.sum(w_pair * ddist_dr)

            # dtheta contribution: dp_i/dtheta_r = (-R_r sin a_i, R_r cos a_i) if ri_i==r
            dx_dt = np.zeros(len(i_idx))
            dy_dt = np.zeros(len(i_idx))
            dx_dt[m_i] += -Ri[m_i] * si[m_i]
            dy_dt[m_i] += Ri[m_i] * ci[m_i]
            dx_dt[m_j] -= -Rj[m_j] * sj[m_j]
            dy_dt[m_j] -= Rj[m_j] * cj[m_j]
            ddist_dt = dir_vecs[:, 0] * dx_dt + dir_vecs[:, 1] * dy_dt
            grad[n_rings + r] += -np.sum(w_pair * ddist_dt)

        return obj_val, grad

    def init_params(scale=1.0):
        """Initial ring radii and offsets."""
        R = np.zeros(n_rings)
        # rough evenly spaced radii
        for ri in range(1, n_rings):
            R[ri] = (ri / n_rings) * 0.9 * scale
        theta = np.zeros(n_rings)
        # random small offsets
        theta[1:] += np.random.uniform(-0.2, 0.2, n_rings - 1)
        return np.concatenate([R, theta])

    def ring_config_points():
        """A concrete good ring configuration as starting point."""
        radii = [0.0, 0.25, 0.5, 0.75, 0.95]
        theta = [0.0, 0.0, np.pi / 12, 0.0, np.pi / 14]
        return np.concatenate([radii, theta])

    inits = [
        init_params(1.0),
        init_params(0.9),
        ring_config_points(),
        init_params(1.1),
        ring_config_points() + np.random.normal(0, 0.05, 2 * n_rings),
        init_params(0.8),
    ]

    for init_idx, init_params_vec in enumerate(inits):
        if time.time() > deadline:
            break

        current_p = init_params_vec.copy()

        # Multi-stage alpha annealing
        alphas = [10, 50, 200, 1000, 3000]

        for alpha in alphas:
            if time.time() > deadline - 2:
                break
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad,
                    current_p,
                    args=(alpha,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 40}
                )
                if res.x is not None:
                    current_p = res.x
            except Exception:
                pass

        curr_points = params_to_points(current_p)
        curr_score = get_score(curr_points)

        if curr_score > best_score:
            best_score = curr_score
            best_points = curr_points.copy()

    # Local polishing on best with high alpha
    if best_points is not None and time.time() < deadline - 5:
        # recover params from best? We don't have inverse; instead polish via re-optimization from a param init near best.
        # Simpler: run a few high-alpha L-BFGS from the last best param set.
        for _ in range(4):
            if time.time() > deadline - 2:
                break
            try:
                # re-init near a good ring config and polish
                p = ring_config_points() + np.random.normal(0, 0.02, 2 * n_rings)
                res = scipy.optimize.minimize(
                    objective_and_grad,
                    p,
                    args=(5000,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'gtol': 1e-8}
                )
                if res.x is not None:
                    curr_points = params_to_points(res.x)
                    curr_score = get_score(curr_points)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_points = curr_points.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback: ring config
        best_points = params_to_points(ring_config_points())

    return best_points
