import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes D/M where D = H(x-y), M = max(H(x), H(x+y), H(x+2y)).
    Uses smooth surrogate for max, logit parameterization, and multi-phase optimization.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 595.0  # 5s safety margin from the 630s kill

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]

    # Flatten for binning
    d_val = (xs - ys).flatten()       # x - y
    m0_val = xs.flatten()              # x
    m1_val = (xs + ys).flatten()       # x + y
    m2_val = (xs + 2 * ys).flatten()   # x + 2y

    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)

    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def shannon(p_dist):
        p_nonzero = p_dist[p_dist > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    # Smooth surrogate: soft-max of entropies using log-sum-exp
    def soft_max(values, temperature=0.01):
        """Approximate max using log-sum-exp for differentiability"""
        shifted = np.array(values) / temperature
        max_shift = np.max(shifted)
        if not np.isfinite(max_shift):
            return max_shift * temperature
        exp_shifted = np.exp(shifted - max_shift)
        log_sum = np.log(np.sum(exp_shifted)) + max_shift
        return log_sum * temperature

    def compute_entropies_soft(p_flat, temperature=0.01):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        M_soft = soft_max([H_m0, H_m1, H_m2], temperature)
        return H_d, M_soft

    def objective_smooth(p_flat, temperature=0.01):
        H_d, M_soft = compute_entropies_soft(p_flat, temperature)
        if M_soft < 1e-9:
            return 1e9
        return -H_d / M_soft

    # Logit parameterization: map unconstrained z to probability simplex
    def z_to_p(z):
        """Convert unconstrained vector to probability distribution via softmax"""
        # Subtract max for numerical stability
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        p = exp_z / np.sum(exp_z)
        return p

    def objective_z(z, temperature=0.01):
        p = z_to_p(z)
        return objective_smooth(p, temperature)

    # --- Build structured starting points ---
    starts = []

    # 1. Uniform
    starts.append(np.ones(100) / 100.0)

    # 2. Single x-line
    for x0 in range(n):
        p_flat = np.zeros(100)
        p_flat[x0 * 10:(x0 + 1) * 10] = 1.0 / 10.0
        starts.append(p_flat)

    # 3. Single y-line
    for y0 in range(n):
        p_flat = np.zeros(100)
        p_flat[y0 * 10:(y0 + 1) * 10] = 1.0 / 10.0
        starts.append(p_flat)

    # 4. Main diagonal
    p_diag = np.zeros(100)
    for i in range(n):
        p_diag[i * 10 + i] = 1.0 / 10.0
    starts.append(p_diag)

    # 5. Anti-diagonal
    p_adiag = np.zeros(100)
    for i in range(n):
        p_adiag[i * 10 + (9 - i)] = 1.0 / 10.0
    starts.append(p_adiag)

    # 6. Concentrated on two x-values with spread y
    for x0 in range(n):
        for x1 in range(x0 + 1, n):
            p_flat = np.zeros(100)
            p_flat[x0 * 10:(x0 + 1) * 10] = 0.5 / 10.0
            p_flat[x1 * 10:(x1 + 1) * 10] = 0.5 / 10.0
            starts.append(p_flat)

    # 7. Random Dirichlet with varied concentration
    rng = np.random.default_rng(42)
    for trial in range(60):
        alpha = rng.uniform(0.05, 5.0, 100)
        p_flat = rng.dirichlet(alpha)
        starts.append(p_flat)

    # 8. Checkerboard patterns
    for offset in range(2):
        p_flat = np.zeros(100)
        for y in range(n):
            for x in range(n):
                if (x + y) % 2 == offset:
                    p_flat[y * 10 + x] = 1.0
        s = p_flat.sum()
        if s > 0:
            p_flat /= s
            starts.append(p_flat)

    # 9. Fourier-like patterns: concentrate on arithmetic progressions
    for step in [1, 2, 3]:
        for phase in range(step):
            p_flat = np.zeros(100)
            count = 0
            for y in range(n):
                for x in range(n):
                    if (x - step * y) % step == phase:
                        p_flat[y * 10 + x] = 1.0
                        count += 1
            if count > 0:
                p_flat /= count
                starts.append(p_flat)

    # 10. Sparse patterns: concentrate on small subsets
    for subset_size in [3, 5, 7]:
        for trial in range(5):
            indices = rng.choice(100, size=subset_size, replace=False)
            p_flat = np.zeros(100)
            p_flat[indices] = 1.0 / subset_size
            starts.append(p_flat)

    # --- Optimize from each start ---
    best_score = -1e9
    p_best = starts[0].copy()

    # Pre-evaluate all starts
    for p_flat in starts:
        score = get_score(p_flat)
        if score > best_score:
            best_score = score
            p_best = p_flat.copy()

    # Phase 1: SLSQP with smooth objective (lower temperature for smoother landscape)
    bounds = [(0.0, 1.0) for _ in range(100)]
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]

    # Use smooth objective with annealing temperature
    temperatures = [0.1, 0.05, 0.01, 0.001]
    max_slsqp_per_temp = 30

    for temp in temperatures:
        if time.time() > deadline - 60.0:
            break

        slsqp_count = 0
        for i, start_p in enumerate(starts):
            if time.time() > deadline - 30.0:
                break
            if slsqp_count >= max_slsqp_per_temp:
                break

            try:
                res = minimize(
                    lambda p: objective_smooth(p, temp),
                    start_p,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )

                p_res = res.x
                p_res = np.clip(p_res, 0, None)
                s = np.sum(p_res)
                if s > 1e-9:
                    p_res /= s

                score = get_score(p_res)
                if score > best_score:
                    best_score = score
                    p_best = p_res.copy()

                slsqp_count += 1

            except Exception:
                pass

    # Phase 2: Logit-space optimization (unconstrained)
    if time.time() < deadline - 40.0:
        # Convert best p to logit space
        p_best_clipped = np.clip(p_best, 1e-10, None)
        p_best_clipped /= np.sum(p_best_clipped)
        z_init = np.log(p_best_clipped)

        try:
            res_z = minimize(
                lambda z: objective_z(z, 0.001),
                z_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            p_z = z_to_p(res_z.x)
            score_z = get_score(p_z)
            if score_z > best_score:
                best_score = score_z
                p_best = p_z.copy()
        except Exception:
            pass

    # Phase 3: High-precision polish with Nelder-Mead on exact objective
    if time.time() < deadline - 15.0:
        try:
            res_nm = minimize(
                lambda p: -get_score(p) if get_score(p) > -1e9 else 1e9,
                p_best.copy(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-15, 'xatol': 1e-16}
            )
            p_nm = res_nm.x
            p_nm = np.clip(p_nm, 0, None)
            s = np.sum(p_nm)
            if s > 1e-9:
                p_nm /= s
            score_nm = get_score(p_nm)
            if score_nm > best_score:
                best_score = score_nm
                p_best = p_nm.copy()
        except Exception:
            pass

    # Phase 4: Final fine-tuning with very tight SLSQP
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                lambda p: -get_score(p) if get_score(p) > -1e9 else 1e9,
                p_best.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            p_final = res_final.x
            p_final = np.clip(p_final, 0, None)
            s = np.sum(p_final)
            if s > 1e-9:
                p_final /= s
            score_final = get_score(p_final)
            if score_final > best_score:
                best_score = score_final
                p_best = p_final.copy()
        except Exception:
            pass

    # --- Final validation and normalization ---
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)

    return P
