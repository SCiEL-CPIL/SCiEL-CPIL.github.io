"""Rewritten program: circle packing in unit square with gradient-based optimisation."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START

def construct():
    """Return centers of N circles inside [-1/2, 1/2]^2 maximising common radius."""
    t0 = time.time()
    DEADLINE = 143.0  # seconds before the 150 s kill; safety margin

    def _remaining():
        return DEADLINE - (time.time() - t0)

    # ---------- geometry helpers ----------
    def _radius(c):
        """Max radius for centres c (N,2)."""
        wall = 0.5 - np.max(np.abs(c), axis=1)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(min(np.min(wall), np.min(d) / 2.0))

    def _bottleneck(c):
        """Return (r, kind, idx_a, idx_b) for the tightest constraint."""
        wall = 0.5 - np.max(np.abs(c), axis=1)
        i_wall = int(np.argmin(wall))
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        i_pair = int(np.argmin(d))
        j1, j2 = divmod(i_pair, N)
        if wall[i_wall] < d[j1, j2] / 2.0:
            return wall[i_wall], 'wall', i_wall, -1
        else:
            return d[j1, j2] / 2.0, 'pair', j1, j2

    # ---------- initialisation ----------
    def _hex_grid(rows, cols, shift=True):
        """Hexagonal arrangement of up to N points."""
        pts = []
        # Spacing chosen so total width ≈ 1.0
        dx = 1.0 / (cols + (0.5 if shift else 0.0))
        dy = dx * math.sqrt(3) / 2.0
        total_h = (rows - 1) * dy + dx
        if total_h > 1.0 + 1e-4:
            return None
        y0 = -0.5 + dx / 2.0
        for r in range(rows):
            y = y0 + r * dy
            if shift and r % 2 == 1:
                x_start = -0.5 + dx
                n_x = cols - 1
            else:
                x_start = -0.5 + dx / 2.0
                n_x = cols
            for cc in range(n_x):
                if len(pts) < N:
                    pts.append([x_start + cc * dx, y])
        if len(pts) < N:
            return None
        return np.array(pts[:N])

    def _square_grid(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        return np.array([(x, y) for y in ticks for x in ticks])[:N]

    def _random_init():
        """Uniform random in a slightly shrunken square."""
        return np.random.uniform(-0.49, 0.49, (N, 2))

    # ---------- optimisation core ----------
    def _softmin_grad(c, k):
        """
        Minimise -softmin of constraint values.
        Constraint values: wall distances (N) and half-pairwise-distances (N^2).
        Returns (f, g) where f is objective, g is gradient w.r.t. flattened c.
        """
        n = c.shape[0]
        # Wall constraints: v_i = 0.5 - max(|x_i|, |y_i|)
        wall = 0.5 - np.max(np.abs(c), axis=1)  # (N,)

        # Pairwise: d_ij / 2 for all i<j, but we use full matrix for symmetry
        diff = c[:, None, :] - c[None, :, :]  # (N, N, 2)
        d = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(d, np.inf)
        pair = d / 2.0  # (N, N)

        # Combine all constraint values
        vals_wall = wall  # (N,)
        vals_pair = pair  # (N, N)

        # Softmin: -1/k * log(sum(exp(-k * v)))
        # Stable computation
        all_vals = np.concatenate([vals_wall, vals_pair.flatten()])
        min_v = np.min(all_vals)
        neg_k = -k * (all_vals - min_v)
        # Avoid overflow: clip
        neg_k = np.clip(neg_k, -500, 0)
        exp_neg_k = np.exp(neg_k)
        log_sum = min_v + np.log(np.sum(exp_neg_k)) / (-k)
        # Actually: softmin = -1/k * log(sum(exp(-k*v)))
        # Let M = max(-k*v) = -k*min_v
        # sum(exp(-k*v)) = exp(M) * sum(exp(-k*v - M))
        # log(sum(exp(-k*v))) = M + log(sum(exp(-k*v - M)))
        # softmin = -1/k * (M + log_sum_part) = min_v - (1/k)*log(sum(exp(-k*v - M)))
        # Let me redo carefully.
        # softmin(v) = -1/k * log( sum_i exp(-k v_i) )
        # Let m = min(v). Then -k v_i = -k(m + (v_i - m)) = -k m - k(v_i - m)
        # sum exp(-k v_i) = exp(-k m) sum exp(-k (v_i - m))
        # log(sum) = -k m + log(sum exp(-k (v_i - m)))
        # softmin = -1/k * (-k m + log_sum2) = m - (1/k) log(sum exp(-k (v_i - m)))
        
        v_shifted = all_vals - min_v
        exp_part = np.exp(-k * v_shifted)
        log_sum2 = np.log(np.sum(exp_part))
        softmin_val = min_v - log_sum2 / k
        f = -softmin_val  # we minimise this

        # Gradient of f w.r.t. v_i:
        # f = -softmin = 1/k * log(sum exp(-k v_i)) - min_v  (but min_v depends on v too, complicated)
        # Better: directly differentiate.
        # softmin = -1/k log(S) where S = sum exp(-k v_i)
        # d(softmin)/d v_i = -1/k * (1/S) * dS/dv_i = -1/k * (1/S) * (-k exp(-k v_i)) = exp(-k v_i)/S
        # d f / d v_i = -d(softmin)/d v_i = -exp(-k v_i)/S
        # Weights w_i = exp(-k v_i) / S, sum to 1.
        # d f / d v_i = -w_i
        
        w = exp_part / np.sum(exp_part)  # (len(all_vals),)
        # Split back
        w_wall = w[:n]  # (N,)
        w_pair = w[n:].reshape(n, n)  # (N, N)

        # Gradient w.r.t. centres
        g = np.zeros_like(c)
        
        # Wall part: v_i = 0.5 - max(|x_i|, |y_i|)
        # d v_i / d x_i = -sign(x_i) if |x_i| >= |y_i| else 0
        # d v_i / d y_i = -sign(y_i) if |y_i| > |x_i| else 0
        abs_c = np.abs(c)
        # For each i, determine which coordinate is the max
        is_x_max = abs_c[:, 0] >= abs_c[:, 1]
        # d v_i / d x_i = -sign(x_i) * is_x_max_i
        # d v_i / d y_i = -sign(y_i) * (1 - is_x_max_i)  [only if y is strictly max; tie goes to x]
        # At ties (|x|==|y|), subgradient; use 0.5 each or just x. We use x.
        dx_wall = -np.sign(c[:, 0]) * is_x_max
        dy_wall = -np.sign(c[:, 1]) * (~is_x_max)
        g[:, 0] += w_wall * dx_wall
        g[:, 1] += w_wall * dy_wall

        # Pair part: v_ij = d_ij / 2 = 0.5 * ||c_i - c_j||
        # d v_ij / d c_i = 0.5 * (c_i - c_j) / d_ij
        # d v_ij / d c_j = 0.5 * (c_j - c_i) / d_ij = -0.5 * (c_i - c_j) / d_ij
        # d f / d c_i = sum_j w_ij * d v_ij / d c_i + sum_j w_ji * d v_ji / d c_i
        # Note w is symmetric (w_ij = w_ji since d_ij = d_ji), and v_ij = v_ji.
        # Actually w_pair[i,j] = w_pair[j,i] since d is symmetric.
        # d f / d c_i = sum_{j != i} w_pair[i,j] * 0.5 * (c_i - c_j) / d_ij
        # (the j->i term for the same pair is captured by the i->j term since w is symmetric)
        # Wait: f depends on v_ij for all ordered pairs? No, we have all (i,j) i!=j in vals.
        # v_ij and v_ji are the same value. In the sum, both appear.
        # d f / d c_i = sum_{j != i} [w_pair[i,j] * d v_ij/d c_i + w_pair[j,i] * d v_ji/d c_i]
        # d v_ij/d c_i = 0.5 (c_i - c_j)/d_ij
        # d v_ji/d c_i = 0.5 (c_j - c_i)/d_ji = -0.5 (c_i - c_j)/d_ij
        # So sum = w_pair[i,j] * 0.5 (c_i-c_j)/d_ij - w_pair[j,i] * 0.5 (c_i-c_j)/d_ij
        # Since w_pair[i,j] = w_pair[j,i], this is 0!
        # That can't be right. The issue is that we're double-counting.
        # Actually in the all_vals array, each pair (i,j) with i != j appears once.
        # v_{ij} = ||c_i - c_j||/2. Only one entry per unordered pair.
        # So d f / d c_i = sum_{j != i} w_pair[i,j] * 0.5 * (c_i - c_j) / d_ij
        # where w_pair[i,j] is the weight for the entry (i,j) in the flattened pair array.
        # But in our flatten, we have both (i,j) and (j,i) as separate entries with same value.
        # So we should really use only upper triangle. Let me restructure.
        
        # Better approach: use only i < j pairs.
        # Let me redo: vals = [wall_0, ..., wall_{N-1}, pair_{01}, pair_{02}, ..., pair_{(N-1)(N-2)}]
        # This is cleaner.
        
        # Actually for gradient computation with symmetric weights, the double counting cancels.
        # Let me verify with a 2-point example.
        # N=2, pairs: (0,1) and (1,0). vals_pair = [d01/2, d01/2]. w_pair = [w0, w0] where w0 = exp(-k d01/2)/S.
        # d f / d c_0 = w0 * 0.5 (c0-c1)/d01 + w0 * 0.5 (c1-c0)/d01 = 0.
        # That means the pair constraint gives zero gradient, which is wrong.
        # The problem: we should not include both (i,j) and (j,i) as separate constraints.
        # They are the same constraint. Let me fix to use only i < j.
        
        # I'll recompute with only upper triangle pairs.
        pass  # placeholder, will fix below

    def _softmin_grad_v2(c, k):
        """
        Correct version: constraints are wall_i for i in range(N) and pair_{ij} for 0<=i<j<N.
        """
        n = c.shape[0]
        wall = 0.5 - np.max(np.abs(c), axis=1)
        diff = c[:, None, :] - c[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        # Upper triangle indices
        iu, ju = np.triu_indices(n, 1)
        pairs = d[iu, ju] / 2.0  # (n*(n-1)/2,)

        all_vals = np.concatenate([wall, pairs])
        min_v = np.min(all_vals)
        v_shift = all_vals - min_v
        exp_part = np.exp(-k * v_shift)
        S = np.sum(exp_part)
        w = exp_part / S
        softmin_val = min_v - np.log(S) / k
        f = -softmin_val

        w_wall = w[:n]
        w_pair = w[n:]  # corresponds to (iu, ju)

        g = np.zeros_like(c)
        # Wall gradient
        abs_c = np.abs(c)
        is_x_max = abs_c[:, 0] >= abs_c[:, 1]
        g[:, 0] += w_wall * (-np.sign(c[:, 0])) * is_x_max
        g[:, 1] += w_wall * (-np.sign(c[:, 1])) * (~is_x_max)

        # Pair gradient: for each (i,j) in iu,ju:
        # d f / d c_i += w_pair[t] * 0.5 * (c_i - c_j) / d_ij
        # d f / d c_j += w_pair[t] * 0.5 * (c_j - c_i) / d_ij
        diff_ij = diff[iu, ju]  # (num_pairs, 2)
        d_ij = d[iu, ju]  # (num_pairs,)
        factor = 0.5 * w_pair / d_ij  # (num_pairs,)
        contrib = factor[:, None] * diff_ij  # (num_pairs, 2)
        # Add to i and subtract from j
        g[iu] += contrib
        g[ju] -= contrib

        return f, g

    def _l_bfgs(c, max_time, k=40.0):
        """Run L-BFGS-B on the softmin objective."""
        from scipy.optimize import minimize as sp_min
        x0 = c.flatten()
        bounds = [(-0.5, 0.5)] * (2 * n_c)

        def obj(x):
            c_t = x.reshape(n_c, 2)
            f, _ = _softmin_grad_v2(c_t, k)
            return f

        def jac(x):
            c_t = x.reshape(n_c, 2)
            _, g = _softmin_grad_v2(c_t, k)
            return g.flatten()

        start = time.time()
        try:
            res = sp_min(
                obj, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return res.x.reshape(n_c, 2)
        except Exception:
            return c
        finally:
            pass

    def _anneal_polish(c, max_time, T0=0.01, T_end=0.0005):
        """
        Simulated-annealing-like local search: at each step, pick the tightest
        constraint and try a random small move of one or two involved points.
        Accept if radius improves, or with probability exp(-delta/T).
        """
        c = c.copy()
        best_r = _radius(c)
        start = time.time()
        T = T0
        cooling = 0.999
        n_iter = 0
        while _remaining() > max(2.0, max_time * 0.05):
            if time.time() - start > max_time:
                break
            r, kind, ia, ib = _bottleneck(c)
            if kind == 'wall':
                # Move point ia toward center
                direction = -np.sign(c[ia])
                direction = direction / (np.linalg.norm(direction) + 1e-12)
            else:
                # Move ia and ib apart
                dvec = c[ia] - c[ib]
                nd = np.linalg.norm(dvec)
                if nd < 1e-12:
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction) + 1e-12
                else:
                    direction = dvec / nd
            # Random perturbation
            scale = 0.005 + 0.02 * T
            delta_a = np.random.randn(2) * scale
            if kind == 'pair' and ib >= 0:
                delta_b = np.random.randn(2) * scale
            else:
                delta_b = np.zeros(2)

            c_new = c.copy()
            c_new[ia] += direction * scale * 0.5 + delta_a
            if kind == 'pair' and ib >= 0:
                c_new[ib] -= direction * scale * 0.5 + delta_b
            # Clamp
            c_new[:, 0] = np.clip(c_new[:, 0], -0.5, 0.5)
            c_new[:, 1] = np.clip(c_new[:, 1], -0.5, 0.5)
            r_new = _radius(c_new)
            delta = r_new - r
            if delta > 0 or np.random.rand() < math.exp(-delta / max(T, 1e-8)):
                c = c_new
                r = r_new
                if r > best_r:
                    best_r = r
            T *= cooling
            n_iter += 1
        return c

    # ---------- main ----------
    n_c = N
    best_c = None
    best_r = 0.0

    # Generate diverse initial configurations
    inits = []

    # Hex grids with various row counts
    for rows in range(13, 18):
        cols = math.ceil(N / rows)
        for shift in [True, False]:
            pts = _hex_grid(rows, cols, shift)
            if pts is not None:
                inits.append(pts)

    # Square grid
    m = int(math.ceil(math.sqrt(N)))
    inits.append(_square_grid(m))

    # A couple random inits
    for _ in range(2):
        inits.append(_random_init())

    # Evaluate and pick best initial
    for init in inits:
        r = _radius(init)
        if r > best_r:
            best_r = r
            best_c = init.copy()

    # Multi-stage optimisation with time budgeting
    # Stage 1: L-BFGS on best init
    if _remaining() > 30:
        c_opt = _l_bfgs(best_c, max_time=min(25, _remaining() - 5), k=30)
        r_opt = _radius(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # Stage 2: Annealing polish
    if _remaining() > 15:
        c_ann = _anneal_polish(best_c, max_time=min(15, _remaining() - 3), T0=0.008)
        r_ann = _radius(c_ann)
        if r_ann > best_r:
            best_r = r_ann
            best_c = c_ann

    # Stage 3: Another L-BFGS with higher k for fine-tuning
    if _remaining() > 10:
        c_opt2 = _l_bfgs(best_c, max_time=min(8, _remaining() - 2), k=80)
        r_opt2 = _radius(c_opt2)
        if r_opt2 > best_r:
            best_r = r_opt2
            best_c = c_opt2

    # Stage 4: Try a second best initial if time allows
    if _remaining() > 20:
        # Sort inits by radius
        scored = []
        for init in inits:
            r_i = _radius(init)
            scored.append((r_i, init))
        scored.sort(key=lambda x: -x[0])
        for r_i, init in scored:
            if r_i >= best_r - 1e-6:
                continue
            if _remaining() < 15:
                break
            c_opt3 = _l_bfgs(init, max_time=min(10, _remaining() - 3), k=40)
            r_opt3 = _radius(c_opt3)
            if r_opt3 > best_r:
                best_r = r_opt3
                best_c = c_opt3
                if _remaining() > 10:
                    c_ann2 = _anneal_polish(best_c, max_time=min(8, _remaining() - 2), T0=0.005)
                    r_ann2 = _radius(c_ann2)
                    if r_ann2 > best_r:
                        best_r = r_ann2
                        best_c = c_ann2

    # Final micro-polish
    if _remaining() > 5:
        c_final = _l_bfgs(best_c, max_time=min(4, _remaining() - 1), k=100)
        r_final = _radius(c_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final

    return best_c

# EVOLVE-BLOCK-END
