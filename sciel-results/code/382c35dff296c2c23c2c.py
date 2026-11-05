"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np


def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """Vectorized reflectance over all wavelengths via characteristic-matrix method."""
    n_wls = len(wls)
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(n_wls, r * r)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = n_mat[materials[i]]
        d = thicknesses[i]
        if d == 0.0:
            continue
        beta = 2.0 * math.pi * n * d / wls
        c = np.cos(beta)
        s = np.sin(beta)
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2


def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """Mean reflectance in percent with constraint penalties."""
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0
    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0
    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0


def _feasible(mats, ths, problem):
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    if len(mats) != len(ths) or len(mats) > problem["max_layers"]:
        return False
    for t in ths:
        if t < -1e-9 or t > max_layer + 1e-9:
            return False
    return sum(ths) <= max_total + 1e-6


def _nelder_mead(objective, x0, lo, hi, deadline, max_iter=300, tol=1e-8):
    """Nelder-Mead with bound clipping and deadline check."""
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        span = hi[i] - lo[i]
        step = 0.05 * span if span > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
    f_vals = [objective(x) for x in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
    order = np.argsort(f_vals)
    bi = order[0]
    return np.clip(simplex[bi], lo, hi), f_vals[bi]


def design(problem):
    """Design a broadband antireflection coating via GA + local refinement."""
    start_time = time.time()
    deadline = start_time + 135.0

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]

    best_val = float('inf')
    best_mats = None
    best_ths = None

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        if not _feasible(mats, ths, problem):
            return
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(mats)
            best_ths = list(ths)
        return val

    # Build a diverse population of alternating stacks (global search).
    population = []  # (mats, thicknesses)
    rng = np.random.default_rng(20240521)
    for n_layers in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                mats.append('L' if (i % 2 == 0) == (start_char == 'L') else 'H')
            # Quarter-wave seed at the band center
            center = 0.5 * (band_start + band_end)
            ths = [center / (4.0 * n_mat[m]) for m in mats]
            population.append((mats, ths))
            # A few perturbed seeds
            for _ in range(2):
                ths2 = [max(0.0, min(max_layer, t * (0.7 + 0.6 * rng.random())))
                        for t in ths]
                population.append((mats, ths2))

    # Evaluate initial population
    pop_fit = []
    for mats, ths in population:
        if _feasible(mats, ths, problem):
            pop_fit.append(_mean_reflectance_percent(mats, ths, problem, wls, n_mat))
        else:
            pop_fit.append(1e6)
    update_best_from_pop = [(population[i], pop_fit[i]) for i in range(len(population))]
    for mats, ths in population:
        if _feasible(mats, ths, problem):
            update_best(mats, ths)

    # Genetic algorithm loop
    pop_size = len(population)
    elite_keep = max(4, pop_size // 4)
    mutation_rate = 0.35
    for gen in range(60):
        if time.time() > deadline - 25.0:
            break
        # Sort by fitness
        idx = np.argsort(pop_fit)
        population = [population[i] for i in idx]
        pop_fit = [pop_fit[i] for i in idx]

        new_pop = list(population[:elite_keep])
        while len(new_pop) < pop_size:
            if time.time() > deadline - 25.0:
                break
            # Tournament selection
            a = population[int(rng.integers(0, pop_size))]
            b = population[int(rng.integers(0, pop_size))]
            parent = a if pop_fit[int(np.where(np.array(population) == a)[0][0])] <= pop_fit[int(np.where(np.array(population) == b)[0][0])] else b
            # Simpler: pick best of two random
            i1 = int(rng.integers(0, pop_size))
            i2 = int(rng.integers(0, pop_size))
            parent = population[i1] if pop_fit[i1] <= pop_fit[i2] else population[i2]
            child_mats = list(parent[0])
            child_ths = list(parent[1])
            # Mutation: perturb some thicknesses
            for k in range(len(child_ths)):
                if rng.random() < mutation_rate:
                    child_ths[k] = max(0.0, min(max_layer, child_ths[k] * (0.6 + 0.8 * rng.random())))
            new_pop.append((child_mats, child_ths))
        population = new_pop
        pop_fit = []
        for mats, ths in population:
            if _feasible(mats, ths, problem):
                pop_fit.append(_mean_reflectance_percent(mats, ths, problem, wls, n_mat))
            else:
                pop_fit.append(1e6)
            update_best(mats, ths)

    # Local refinement of the best GA individuals
    idx = np.argsort(pop_fit)
    n_refine = min(6, len(population))
    for k in range(n_refine):
        if time.time() > deadline - 10.0:
            break
        mats, ths = population[idx[k]]
        if not _feasible(mats, ths, problem):
            continue
        lo = np.zeros(len(ths))
        hi = np.full(len(ths), max_layer)
        x0 = np.array(ths, dtype=float)
        # Enforce total constraint during objective via projection
        def objective(x):
            x = np.clip(x, lo, hi)
            total = np.sum(x)
            if total > max_total:
                x = x * (max_total / total)
                x = np.clip(x, lo, hi)
            return _mean_reflectance_percent(mats, x.tolist(), problem, wls, n_mat)
        try:
            x_opt, f_opt = _nelder_mead(objective, x0, lo, hi, deadline, max_iter=250)
            x_opt = np.clip(x_opt, lo, hi)
            total = np.sum(x_opt)
            if total > max_total:
                x_opt = x_opt * (max_total / total)
                x_opt = np.clip(x_opt, lo, hi)
            update_best(mats, x_opt.tolist())
        except Exception:
            pass

    # Fallback
    if best_mats is None:
        center = 0.5 * (band_start + band_end)
        best_mats = ['L', 'H']
        best_ths = [center / (4.0 * n_L), center / (4.0 * n_H)]
        if sum(best_ths) > max_total:
            s = max_total / sum(best_ths)
            best_ths = [t * s for t in best_ths]

    # Final constraint enforcement
    best_ths = [max(0.0, min(t, max_layer)) for t in best_ths]
    total = sum(best_ths)
    if total > max_total:
        s = max_total / total
        best_ths = [t * s for t in best_ths]

    return best_mats, best_ths
# EVOLVE-BLOCK-END
