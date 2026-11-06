"""Broadband antireflection coating design: CMA-ES global search + fast local polish."""
import numpy as np
import time


def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """Compute reflectance (fraction, 0-1) at each wavelength via characteristic matrix method."""
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)

    k = 2.0 * np.pi / wavelengths_nm

    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)

    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d

        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11

        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11

    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)

    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    return R


def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
               max_layer_nm, max_total_nm):
    """Objective: mean reflectance in percent. Returns 100 on infeasibility."""
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0

    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0

    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped,
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0


def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-12, time_deadline=None):
    """Nelder-Mead simplex with bounds."""
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)

    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step

    fvals = np.array([f(simplex[i]) for i in range(n + 1)])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break

        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]

        if np.max(fvals) - np.min(fvals) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)

        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1], fvals[-1] = x_e, f_e
            else:
                simplex[-1], fvals[-1] = x_r, f_r
        elif f_r < fvals[-2]:
            simplex[-1], fvals[-1] = x_r, f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1], fvals[-1] = x_c, f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]),
                                         lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])

    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def _polish_with_gradient_descent(f, x0, bounds, max_iter=2000, time_deadline=None,
                                  grad_tol=1e-10):
    """High-precision gradient descent with numerical gradient and backtracking line search."""
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)

    def numerical_gradient(x):
        eps = 1e-7
        grad = np.zeros(n)
        for i in range(n):
            xp = x.copy(); xp[i] += eps; xp = np.clip(xp, lo_bounds, hi_bounds)
            xm = x.copy(); xm[i] -= eps; xm = np.clip(xm, lo_bounds, hi_bounds)
            grad[i] = (f(xp) - f(xm)) / (2 * eps)
        return grad

    lr = 1.0
    for _ in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break

        grad = numerical_gradient(x)
        if np.max(np.abs(grad)) < grad_tol:
            break

        direction = -grad
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-12:
            break
        direction = direction / norm_dir

        step = lr
        f_old = fval
        improved = False
        for _ls in range(60):
            x_trial = np.clip(x + step * direction, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            if f_trial < f_old - 1e-4 * step * np.dot(grad, direction):
                x, fval, improved = x_trial, f_trial, True
                break
            step *= 0.5

        if not improved:
            break

        if fval < f_old - 1e-3:
            lr = min(lr * 1.5, 20.0)
        elif fval > f_old + 1e-4:
            lr = max(lr * 0.5, 0.005)

    return x, fval


def _cmaes(f, x0, bounds, max_iter=150, pop_size=30, time_deadline=None):
    """Bounded CMA-ES (simplified) for global search over thicknesses."""
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    pop_size = max(pop_size, 2 * n + 1)

    sigma = 0.1 * np.max(hi - lo)
    x = x0.copy()
    fval = f(x)

    C = np.diag((hi - lo) ** 2 / 4.0)
    mu = pop_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 - mu_eff + c_sigma) / (n + 4 - mu_eff)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    p_sigma = np.zeros(n)
    p_c = np.zeros(n)

    try:
        B = np.linalg.cholesky(C)
    except Exception:
        B = np.sqrt(np.diag(C)).reshape(n, 1)

    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break

        population = np.empty((pop_size, n))
        for i in range(pop_size):
            y = B @ np.random.randn(n)
            population[i] = np.clip(x + sigma * y, lo, hi)

        fvals = np.array([f(population[i]) for i in range(pop_size)])
        idx = np.argsort(fvals)
        population, fvals = population[idx], fvals[idx]

        new_x = np.sum(weights[:, None] * population[:mu], axis=0)

        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
            B.T @ (new_x - x) / sigma)
        if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1 / mu_eff) * np.sqrt(
                (1 - c_sigma) / (1 - c_sigma ** (2 * (it + 1)))):
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x - x) / sigma
        C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (c_c * (2 - c_c) - 1) * C)
        delta_h = 1 if np.linalg.norm(p_c) / chi_n < (1.4 + 1 / mu_eff) * np.sqrt(
            (1 - c_c) / (1 - c_c ** (2 * (it + 1)))) else 0
        C = (1 - c_1 - c_mu + c_mu * delta_h) * C + c_mu * np.sum(
            weights[:, None, None] * np.outer(population[:mu] - x, population[:mu] - x) / sigma ** 2,
            axis=0)

        C = (C + C.T) / 2
        try:
            B = np.linalg.cholesky(C + 1e-12 * np.eye(n))
        except Exception:
            B = np.sqrt(np.diag(np.maximum(np.diag(C), 1e-12))).reshape(n, 1)

        x = np.clip(new_x, lo, hi)
        fval = f(x)

        if np.all(np.abs(x - lo) < 1e-6) or np.all(np.abs(x - hi) < 1e-6):
            break

    return x, fval


def _perturb_and_swap(f, x0, labels, materials, bounds, wavelengths_nm, ambient_idx,
                      substrate_idx, max_layer_nm, max_total_nm, time_deadline, max_iter=150):
    """Local search: perturb thicknesses, swap adjacent materials, add/remove layers."""
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    current_labels = labels[:]

    for _ in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break

        best_x, best_f, best_labels = x, fval, current_labels[:]

        for _ in range(3):
            x_trial = x.copy()
            x_trial[np.random.randint(n)] += np.random.uniform(-20, 20)
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            if np.sum(x_trial) <= max_total_nm:
                ft = f(x_trial)
                if ft < best_f:
                    best_f, best_x = ft, x_trial

        if n > 1:
            i = np.random.randint(n - 1)
            lt = current_labels[:]
            lt[i], lt[i + 1] = lt[i + 1], lt[i]
            it = [materials[l] for l in lt]
            ft = _objective(x, it, wavelengths_nm, ambient_idx, substrate_idx,
                            max_layer_nm, max_total_nm)
            if ft < best_f:
                best_f, best_labels = ft, lt

        if n < 20 and np.sum(x) < max_total_nm - 10:
            new_label = 'L' if np.random.random() < 0.5 else 'H'
            pos = np.random.randint(n + 1)
            lt = current_labels[:pos] + [new_label] + current_labels[pos:]
            it = [materials[l] for l in lt]
            xt = np.clip(np.insert(x, pos, 50.0), 0, max_layer_nm)
            if np.sum(xt) <= max_total_nm:
                ft = _objective(xt, it, wavelengths_nm, ambient_idx, substrate_idx,
                                max_layer_nm, max_total_nm)
                if ft < best_f:
                    best_f, best_x, best_labels = ft, xt, lt

        if n > 1:
            pos = np.random.randint(n)
            lt = current_labels[:pos] + current_labels[pos + 1:]
            it = [materials[l] for l in lt]
            xt = np.delete(x, pos)
            if np.sum(xt) <= max_total_nm:
                ft = _objective(xt, it, wavelengths_nm, ambient_idx, substrate_idx,
                                max_layer_nm, max_total_nm)
                if ft < best_f:
                    best_f, best_x, best_labels = ft, xt, lt

        if best_f < fval:
            x, fval, current_labels = best_x, best_f, best_labels
        else:
            x_trial = np.clip(x + np.random.uniform(-10, 10, size=n), lo_bounds, hi_bounds)
            if np.sum(x_trial) <= max_total_nm:
                ft = f(x_trial)
                if ft < fval:
                    x, fval = x_trial, ft

    return x, fval, current_labels


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before the 150 s kill

    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)

    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])

    best_labels = None
    best_thicknesses = None
    best_score = np.inf

    # Alternating patterns (odd layer counts) for L-first and H-first.
    patterns = []
    for n in range(3, min(max_layers + 1, 21), 2):
        patterns.append((n, ['L' if i % 2 == 0 else 'H' for i in range(n)]))
        patterns.append((n, ['H' if i % 2 == 0 else 'L' for i in range(n)]))

    def pattern_priority(p):
        n = p[0]
        if n in (5, 7, 9):
            return 0
        if n in (3, 11):
            return 1
        if n in (13, 15):
            return 2
        return 3

    patterns.sort(key=pattern_priority)

    for n_layers, labels in patterns:
        if time.time() > deadline - 15:
            break

        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers

        def make_f(idx):
            def f(x):
                return _objective(x, idx, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f

        f = make_f(indices)

        # Physics-motivated initial guesses.
        starts = []
        starts.append(np.array([center_wl / (4.0 * n) for n in indices]))
        starts.append(np.array([band_nm[0] / (4.0 * n) for n in indices]))
        starts.append(np.array([band_nm[1] / (4.0 * n) for n in indices]))
        starts.append(np.array([
            (0.3 * band_nm[0] + 0.4 * center_wl + 0.3 * band_nm[1]) / (4.0 * n)
            for n in indices]))
        starts.append(np.array([center_wl / (4.0 * n) for n in indices]) *
                      np.random.uniform(0.7, 1.3, size=n_layers))
        rnd = np.random.uniform(0, max_layer_nm * 0.5, size=n_layers)
        if np.sum(rnd) > max_total_nm:
            rnd = rnd * (max_total_nm / np.sum(rnd))
        starts.append(rnd)

        for x0 in starts:
            if time.time() > deadline - 12:
                break

            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))

            f0 = f(x0)
            if f0 < best_score:
                best_score, best_labels, best_thicknesses = f0, labels.copy(), x0.copy()
            if f0 >= 100.0:
                continue

            # Phase 1: global CMA-ES (small population, time-bounded).
            cma_deadline = time.time() + min(6.0, max(0.5, deadline - 12 - time.time()))
            try:
                x_c, f_c = _cmaes(f, x0, bounds, max_iter=120, pop_size=30,
                                  time_deadline=cma_deadline)
            except Exception:
                x_c, f_c = x0, f0
            if f_c < best_score:
                best_score, best_labels, best_thicknesses = f_c, labels.copy(), x_c.copy()

            # Phase 2: Nelder-Mead local refinement.
            nm_deadline = time.time() + min(4.0, max(0.5, deadline - 10 - time.time()))
            try:
                x_nm, f_nm = _nelder_mead(f, x_c, bounds, max_iter=400,
                                          time_deadline=nm_deadline)
            except Exception:
                x_nm, f_nm = x_c, f_c
            if f_nm < best_score:
                best_score, best_labels, best_thicknesses = f_nm, labels.copy(), x_nm.copy()

            # Phase 3: high-precision gradient polish.
            if f_nm < best_score + 0.5 and time.time() < deadline - 8:
                gd_deadline = time.time() + min(3.0, max(0.5, deadline - 8 - time.time()))
                try:
                    x_gd, f_gd = _polish_with_gradient_descent(f, x_nm, bounds, max_iter=2000,
                                                               time_deadline=gd_deadline)
                except Exception:
                    x_gd, f_gd = x_nm, f_nm
                if f_gd < best_score:
                    best_score, best_labels, best_thicknesses = f_gd, labels.copy(), x_gd.copy()

            # Phase 4: perturb + swap local search.
            if f_gd < best_score + 0.5 and time.time() < deadline - 5:
                ps_deadline = time.time() + min(3.0, max(0.5, deadline - 5 - time.time()))
                try:
                    x_ps, f_ps, labels_ps = _perturb_and_swap(
                        f, x_gd, labels, materials, bounds, wavelengths_nm,
                        ambient_idx, substrate_idx, max_layer_nm, max_total_nm,
                        time_deadline=ps_deadline, max_iter=120)
                except Exception:
                    x_ps, f_ps, labels_ps = x_gd, f_gd, labels
                if f_ps < best_score:
                    best_score, best_labels, best_thicknesses = f_ps, labels_ps, x_ps.copy()

            if best_score < 0.1:
                break

        if best_score < 0.1:
            break

    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]

    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))

    return best_labels, thicknesses.tolist()
