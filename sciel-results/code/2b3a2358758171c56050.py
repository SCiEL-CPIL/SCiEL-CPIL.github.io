"""Broadband antireflection coating: DE global + Nelder-Mead + gradient polish on fixed labels."""
import numpy as np
import time


def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """Reflectance (fraction) via characteristic-matrix method."""
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


def _make_objective(indices, wls, ambient_idx, substrate_idx, max_layer_nm, max_total_nm):
    def obj(x):
        t = np.clip(np.asarray(x, dtype=float), 0.0, max_layer_nm)
        if t.sum() > max_total_nm:
            return 100.0
        R = _compute_reflectance(wls, indices, t, ambient_idx, substrate_idx)
        return float(np.mean(R)) * 100.0
    return obj


def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, deadline=None):
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)

    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] > lo[i] else 0.1
        simplex[i + 1][i] += step

    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        if np.max(fvals) - np.min(fvals) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)

        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = f(x_r)
        if f_r < fvals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1], fvals[-1] = x_e, f_e
            else:
                simplex[-1], fvals[-1] = x_r, f_r
        elif f_r < fvals[-2]:
            simplex[-1], fvals[-1] = x_r, f_r
        else:
            x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1], fvals[-1] = x_c, f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
                    fvals[i] = f(simplex[i])

    bi = np.argmin(fvals)
    return simplex[bi], fvals[bi]


def _gradient_polish(f, x0, bounds, max_iter=1500, deadline=None):
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    fval = f(x)

    def grad(x):
        g = np.zeros(n)
        for i in range(n):
            xp = x.copy(); xp[i] += 1e-5
            xm = x.copy(); xm[i] -= 1e-5
            g[i] = (f(np.clip(xp, lo, hi)) - f(np.clip(xm, lo, hi))) / 2e-5
        return g

    lr = 1.0
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        g = grad(x)
        if np.max(np.abs(g)) < 1e-8:
            break
        d = -g
        nd = np.linalg.norm(d)
        if nd < 1e-10:
            break
        d = d / nd
        step = lr
        f_old = fval
        for _ in range(50):
            xt = np.clip(x + step * d, lo, hi)
            ft = f(xt)
            if ft < f_old - 1e-4 * step * float(np.dot(g, d)):
                x, fval = xt, ft
                break
            step *= 0.5
        if fval < f_old - 1e-3:
            lr = min(lr * 1.5, 10.0)
        elif fval > f_old + 1e-4:
            lr = max(lr * 0.5, 0.01)
    return x, fval


def _differential_evolution(f, bounds, pop_size=40, max_iter=80, deadline=None, cr=0.8, F=0.8):
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fvals = np.array([f(pop[i]) for i in range(pop_size)])
    bi = int(np.argmin(fvals))
    best_x, best_f = pop[bi].copy(), fvals[bi]

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        for i in range(pop_size):
            if deadline is not None and time.time() > deadline:
                break
            a, b, c = np.random.choice(pop_size, size=3, replace=False)
            mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lo, hi)
            trial = pop[i].copy()
            for j in range(n):
                if np.random.random() < cr:
                    trial[j] = mutant[j]
            ft = f(trial)
            if ft < fvals[i]:
                pop[i], fvals[i] = trial, ft
                if ft < best_f:
                    best_f, best_x = ft, trial.copy()
    return best_x, best_f


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock kill

    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)

    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    wls_coarse = np.linspace(band_nm[0], band_nm[1], 201)
    wls_fine = np.linspace(band_nm[0], band_nm[1], 401)

    best_labels = None
    best_thick = None
    best_score = np.inf

    # Fixed air-side-L alternating sequences (dominant broadband-AR family).
    layer_counts = [11, 13, 9, 15, 7, 17, 5, 19, 3]

    for n_layers in layer_counts:
        if time.time() > deadline - 25.0:
            break
        if n_layers > max_layers:
            continue

        labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        indices = [materials[lb] for lb in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers

        obj_coarse = _make_objective(indices, wls_coarse, ambient_idx, substrate_idx,
                                     max_layer_nm, max_total_nm)
        obj_fine = _make_objective(indices, wls_fine, ambient_idx, substrate_idx,
                                   max_layer_nm, max_total_nm)

        init = np.array([center_wl / (4.0 * n) for n in indices])

        # Phase 1: DE on the coarse grid (cheap -> more global search).
        de_deadline = min(deadline - 40.0, start_time + 50.0)
        try:
            x_de, f_de = _differential_evolution(obj_coarse, bounds, pop_size=40,
                                                 max_iter=80, deadline=de_deadline)
        except Exception:
            x_de, f_de = init, obj_coarse(init)

        # Phase 2: Nelder-Mead on the fine grid.
        nm_deadline = min(deadline - 15.0, start_time + 90.0)
        try:
            x_nm, f_nm = _nelder_mead(obj_fine, x_de, bounds, max_iter=1500,
                                      tol=1e-9, deadline=nm_deadline)
        except Exception:
            x_nm, f_nm = x_de, obj_fine(x_de)

        # Phase 3: gradient polish on the fine grid.
        gp_deadline = min(deadline - 3.0, start_time + 130.0)
        try:
            x_gp, f_gp = _gradient_polish(obj_fine, x_nm, bounds, max_iter=1500,
                                          deadline=gp_deadline)
        except Exception:
            x_gp, f_gp = x_nm, f_nm

        # Phase 4: multi-start local polish from a few perturbations of the best.
        starts = [x_gp, x_nm, x_de]
        rng = np.random.RandomState(12345)
        for _ in range(3):
            pert = x_gp + rng.normal(0.0, 15.0, size=n_layers)
            starts.append(np.clip(pert, 0.0, max_layer_nm))
        for s in starts:
            if time.time() > gp_deadline:
                break
            try:
                xp, fp = _gradient_polish(obj_fine, s, bounds, max_iter=400,
                                          deadline=gp_deadline)
            except Exception:
                continue
            if fp < f_gp:
                x_gp, f_gp = xp, fp

        candidates = [(x_de, obj_fine(x_de)), (x_nm, f_nm), (x_gp, f_gp)]
        x_opt, f_opt = min(candidates, key=lambda t: t[1])

        if f_opt < best_score:
            best_score = f_opt
            best_labels = labels
            best_thick = np.asarray(x_opt, dtype=float)

        if best_score < 0.5:
            break

    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]

    t = np.clip(best_thick, 0.0, max_layer_nm)
    if t.sum() > max_total_nm:
        t = t * (max_total_nm / t.sum())
    return best_labels, t.tolist()
