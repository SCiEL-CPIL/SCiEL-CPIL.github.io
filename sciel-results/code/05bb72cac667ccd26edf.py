# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Optimizes the 1D base positions to minimize the union volume.
    Uses a fast exact volume calculation based on piecewise linear integration.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    # Precompute indices
    m = np.arange(N)

    def _exact_volume_fast(z):
        """
        Compute the exact volume of the union of tubes.
        The volume is integral_0^1 L(c)^2 dc.
        L(c) is the length of the union of intervals [z_i + c*i/N, z_i + c*i/N + 1/N].
        L(c) is piecewise linear in c. We find all critical points and integrate exactly.
        """
        critical_cs = [0.0, 1.0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = (i - j) / N
                if abs(denom) < 1e-12:
                    continue
                c1 = (z[j] - z[i]) / denom
                c2 = (z[j] - z[i] + 1.0/N) / denom

                if 0.0 <= c1 <= 1.0:
                    critical_cs.append(c1)
                if 0.0 <= c2 <= 1.0:
                    critical_cs.append(c2)

        critical_cs = np.array(critical_cs)
        critical_cs = np.sort(np.unique(critical_cs))

        volume = 0.0

        for k in range(len(critical_cs) - 1):
            c_start = critical_cs[k]
            c_end = critical_cs[k+1]

            if c_end - c_start < 1e-12:
                continue

            c_mid = 0.5 * (c_start + c_end)

            starts = z + c_mid * m / N
            ends = starts + 1.0 / N

            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]

            gaps = s_sorted[1:] - e_sorted[:-1]
            active_gaps = gaps > 0

            slope_min_s = order[0] / N
            slope_max_e = order[-1] / N

            slope_gaps = 0.0
            for idx in range(N - 1):
                if active_gaps[idx]:
                    slope_gaps += (order[idx+1] - order[idx]) / N

            A = slope_max_e - slope_min_s - slope_gaps

            L_mid = (e_sorted[-1] - s_sorted[0]) - np.sum(gaps[active_gaps])
            B = L_mid - A * c_mid

            if abs(A) < 1e-12:
                volume += B * B * (c_end - c_start)
            else:
                val_end = A * c_end + B
                val_start = A * c_start + B
                volume += (val_end**3 - val_start**3) / (3.0 * A)

        return volume

    def _smooth_volume(z, eps=0.01):
        """
        Smooth approximation of the volume using softplus for gaps.
        Uses Gauss-Legendre quadrature for the c-integral.
        """
        c_quad, w_quad = np.polynomial.legendre.leggauss(64)
        c_quad = 0.5 * (c_quad + 1)
        w_quad = 0.5 * w_quad

        cost = 0.0
        for c, w in zip(c_quad, w_quad):
            starts = z + c * m / N
            ends = starts + 1.0 / N
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            gaps = s_sorted[1:] - e_sorted[:-1]
            soft_gaps = eps * np.logaddexp(0, gaps / eps)
            L = e_sorted[-1] - s_sorted[0] - np.sum(soft_gaps)
            cost += w * L * L
        return cost

    def _smooth_volume_grad(z, eps=0.01):
        """
        Smooth volume with numerical gradient via central differences.
        """
        cost = _smooth_volume(z, eps)
        grad = np.zeros_like(z)
        h = 1e-6
        for i in range(len(z)):
            zp = z.copy()
            zp[i] += h
            zm = z.copy()
            zm[i] -= h
            grad[i] = (_smooth_volume(zp, eps) - _smooth_volume(zm, eps)) / (2 * h)
        return cost, grad

    # Initial guess: Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    keich = keich - keich[0]
    x0 = keich[1:].copy()

    best_z = x0.copy()
    z_full_init = np.zeros(N)
    z_full_init[1:] = best_z
    best_cost = _exact_volume_fast(z_full_init)

    rng = np.random.default_rng(12345)

    def _try_optimize(start_z, run_id):
        """Try to optimize from start_z using multi-stage approach."""
        nonlocal best_z, best_cost
        if time.time() > deadline - safety_margin:
            return

        current_z = start_z.copy()
        z_full = np.zeros(N)
        z_full[1:] = current_z

        # Stage 1: Smooth L-BFGS-B with moderate softness
        try:
            def smooth_obj(params):
                zf = np.zeros(N)
                zf[1:] = params
                return _smooth_volume(zf, 0.03)

            def smooth_grad(params):
                zf = np.zeros(N)
                zf[1:] = params
                _, g = _smooth_volume_grad(zf, 0.03)
                return g[1:]

            res1 = scipy.optimize.minimize(
                smooth_obj,
                current_z,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res1.fun < _smooth_volume(z_full, 0.03):
                current_z = res1.x
        except Exception:
            pass

        # Stage 2: Nelder-Mead on smooth objective with tighter softness
        try:
            def smooth_obj_tight(params):
                zf = np.zeros(N)
                zf[1:] = params
                return _smooth_volume(zf, 0.005)

            res2 = scipy.optimize.minimize(
                smooth_obj_tight,
                current_z,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            if res2.fun < _smooth_volume(z_full, 0.005):
                current_z = res2.x
        except Exception:
            pass

        # Stage 3: Exact Nelder-Mead polish
        try:
            def exact_obj(params):
                zf = np.zeros(N)
                zf[1:] = params
                return _exact_volume_fast(zf)

            res3 = scipy.optimize.minimize(
                exact_obj,
                current_z,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res3.fun < best_cost:
                best_cost = res3.fun
                best_z = res3.x.copy()
        except Exception:
            pass

    # Main optimization loop with multiple restarts
    num_restarts = 40
    for run in range(num_restarts):
        if time.time() > deadline - safety_margin:
            break

        if run == 0:
            start = best_z.copy()
        else:
            # Adaptive noise scale: start larger, decrease over time
            progress = run / num_restarts
            scale = 0.08 * (1.0 - 0.5 * progress) + 0.01
            start = best_z + rng.normal(0, scale, N - 1)

        _try_optimize(start, run)

        # Periodic re-evaluation of best with tighter settings
        if run % 10 == 9 and time.time() < deadline - safety_margin:
            try:
                def exact_obj(params):
                    zf = np.zeros(N)
                    zf[1:] = params
                    return _exact_volume_fast(zf)

                res_re = scipy.optimize.minimize(
                    exact_obj,
                    best_z,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_re.fun < best_cost:
                    best_cost = res_re.fun
                    best_z = res_re.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - safety_margin:
        try:
            def exact_obj(params):
                zf = np.zeros(N)
                zf[1:] = params
                return _exact_volume_fast(zf)

            res_fin = scipy.optimize.minimize(
                exact_obj,
                best_z,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res_fin.fun < best_cost:
                best_cost = res_fin.fun
                best_z = res_fin.x.copy()
        except Exception:
            pass

    # Construct full z array
    z_full = np.zeros(N)
    z_full[1:] = best_z

    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = z_full[i_flat]
    base_y = z_full[j_flat]

    positions = np.stack([base_x, base_y], axis=1)

    return positions

# EVOLVE-BLOCK-END
