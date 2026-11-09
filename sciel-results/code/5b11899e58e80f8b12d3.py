"""Broadband antireflection coating design: variable-length stack with GA + multi-start Nelder-Mead."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings("ignore")


def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    K = wavs.shape[0]
    A = np.ones(K)
    B = np.zeros(K)
    C = np.zeros(K)
    D = np.ones(K)
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return float(100.0 * np.mean(R))


def design(problem):
    start_time = time.time()
    deadline = start_time + 130  # 150s kill, 20s safety margin

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]

    # Quarter-wave thicknesses at center wavelength
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)

    def clamp_ths(ths, n_layers):
        """Clamp thicknesses to valid range and total."""
        ths = np.clip(np.asarray(ths, dtype=float), 0.0, max_layer)
        if len(ths) < n_layers:
            ths = np.resize(ths, n_layers)
        else:
            ths = ths[:n_layers]
        tot = ths.sum()
        if tot > max_total:
            ths = ths * (max_total / tot)
        return ths

    def evaluate(materials, thicknesses):
        """Return mean reflectance, or inf if invalid."""
        if not materials or len(materials) == 0:
            return _mean_reflectance(materials, thicknesses, problem)
        ths = clamp_ths(thicknesses, len(materials))
        if ths.sum() > max_total + 1e-6:
            return float('inf')
        return _mean_reflectance(materials, list(ths), problem)

    # Initialize with a few good starting designs
    best_score = float('inf')
    best_materials = []
    best_ths = []

    def update(materials, ths, score):
        nonlocal best_score, best_materials, best_ths
        if score < best_score:
            best_score = score
            best_materials = list(materials)
            best_ths = list(ths)

    # Seed 1: Maier 7-layer
    maier_pattern = ("L", "H", "H", "L", "H", "H", "L")
    T = qw_L + qw_H
    maier_frac = (0.09, 0.175, 0.21, 0.35, 0.21, 0.175, 0.09)
    init_ths = clamp_ths(np.array([f * T for f in maier_frac]), 7)
    s = evaluate(maier_pattern, init_ths)
    update(maier_pattern, init_ths, s)

    # Seed 2: 5-layer symmetric (L H L H L)
    pattern5 = ("L", "H", "L", "H", "L")
    ths5 = clamp_ths(np.array([qw_L, qw_H, qw_L, qw_H, qw_L]), 5)
    s = evaluate(pattern5, ths5)
    update(pattern5, ths5, s)

    # Seed 3: 3-layer (L H L)
    pattern3 = ("L", "H", "L")
    ths3 = clamp_ths(np.array([qw_L, qw_H, qw_L]), 3)
    s = evaluate(pattern3, ths3)
    update(pattern3, ths3, s)

    # Seed 4: 9-layer symmetric
    pattern9 = ("L", "H", "H", "L", "H", "L", "H", "H", "L")
    ths9 = clamp_ths(np.array([0.5*qw_L, qw_H, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H, 0.5*qw_L]), 9)
    s = evaluate(pattern9, ths9)
    update(pattern9, ths9, s)

    # Seed 5: 11-layer
    pattern11 = ("L", "H", "H", "L", "H", "L", "H", "L", "H", "H", "L")
    ths11 = clamp_ths(np.array([0.3*qw_L, 0.8*qw_H, 0.8*qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, 0.8*qw_H, 0.8*qw_H, 0.3*qw_L]), 11)
    s = evaluate(pattern11, ths11)
    update(pattern11, ths11, s)

    # Local polish on each seed
    def local_polish(materials, ths):
        """Nelder-Mead polish on thicknesses for fixed material pattern."""
        n = len(materials)
        if n == 0:
            return materials, ths, evaluate(materials, ths)

        def obj(x):
            x = clamp_ths(x, n)
            if x.sum() > max_total + 1e-6:
                return best_score + 10.0
            return _mean_reflectance(materials, list(x), problem)

        try:
            res = minimize(
                obj, np.asarray(ths, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-8, "adaptive": True},
            )
            cand = clamp_ths(res.x, n)
            s = obj(cand)
            if s < best_score:
                update(materials, cand, s)
            return materials, cand, s
        except Exception:
            return materials, ths, evaluate(materials, ths)

    # Polish all seeds
    for pat, ths in [(maier_pattern, init_ths), (pattern5, ths5), (pattern3, ths3),
                     (pattern9, ths9), (pattern11, ths11)]:
        if time.time() > deadline - 15:
            break
        local_polish(pat, ths)

    # Genetic algorithm over both layer count and material sequence
    # Representation: (n_layers, material_string, thickness_vector)
    def random_design():
        """Generate a random valid design."""
        n = np.random.randint(3, 16)  # 3 to 15 layers
        # Generate material sequence with some structure
        # Use a mix of random and pattern-based
        r = np.random.random()
        if r < 0.3:
            # Symmetric pattern
            half = n // 2
            half_mat = ['L' if np.random.random() < 0.5 else 'H' for _ in range(half)]
            if n % 2 == 0:
                mat = half_mat + half_mat[::-1]
            else:
                mid = 'L' if np.random.random() < 0.5 else 'H'
                mat = half_mat + [mid] + half_mat[::-1]
        else:
            # Random
            mat = ['L' if np.random.random() < 0.5 else 'H' for _ in range(n)]
        # Ensure first and last are L (good for air interface)
        mat[0] = 'L'
        mat[-1] = 'L'
        # Random thicknesses around quarter-wave
        ths = np.array([qw_L if m == 'L' else qw_H for m in mat]) * np.random.uniform(0.5, 1.5, n)
        ths = clamp_ths(ths, n)
        return mat, ths

    def crossover(mat1, ths1, mat2, ths2):
        """Crossover two designs."""
        n1, n2 = len(mat1), len(mat2)
        # Choose a crossover point
        if np.random.random() < 0.5:
            # Swap material sequences
            return list(mat1), list(ths1)
        else:
            return list(mat2), list(ths2)

    def mutate(mat, ths, rate=0.1):
        """Mutate a design."""
        n = len(mat)
        mat = list(mat)
        ths = list(ths)
        # Mutate materials
        for i in range(n):
            if np.random.random() < rate:
                mat[i] = 'H' if mat[i] == 'L' else 'L'
        # Mutate thicknesses
        for i in range(n):
            if np.random.random() < rate:
                ths[i] = np.clip(ths[i] * np.random.uniform(0.7, 1.3), 0, max_layer)
        # Ensure first and last are L
        mat[0] = 'L'
        mat[-1] = 'L'
        ths = clamp_ths(ths, n)
        return mat, ths

    # GA loop
    pop_size = 20
    max_generations = 30
    population = []

    # Initialize population with diverse designs
    population.append((list(best_materials), list(best_ths), best_score))
    for i in range(pop_size - 1):
        mat, ths = random_design()
        s = evaluate(mat, ths)
        population.append((mat, ths, s))

    for gen in range(max_generations):
        if time.time() > deadline - 20:
            break

        # Sort by fitness
        population.sort(key=lambda x: x[2])

        # Elitism: keep top 5
        elite = population[:5]

        # Selection and reproduction
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            parent1 = population[idx1] if population[idx1][2] < population[idx2][2] else population[idx2]

            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            parent2 = population[idx3] if population[idx3][2] < population[idx4][2] else population[idx4]

            child_mat, child_ths = crossover(parent1[0], parent1[1], parent2[0], parent2[1])
            child_mat, child_ths = mutate(child_mat, child_ths)

            # Update layer count occasionally
            if np.random.random() < 0.1:
                new_n = np.random.randint(max(3, len(child_mat) - 3), min(18, len(child_mat) + 3))
                if new_n != len(child_mat):
                    # Adjust material sequence
                    if new_n < len(child_mat):
                        child_mat = child_mat[:new_n]
                    else:
                        extra = new_n - len(child_mat)
                        child_mat = child_mat + ['L' if np.random.random() < 0.5 else 'H' for _ in range(extra)]
                    child_mat[0] = 'L'
                    child_mat[-1] = 'L'
                    # Regenerate thicknesses
                    child_ths = np.array([qw_L if m == 'L' else qw_H for m in child_mat]) * np.random.uniform(0.5, 1.5, new_n)
                    child_ths = clamp_ths(child_ths, new_n)

            s = evaluate(child_mat, child_ths)
            new_pop.append((child_mat, child_ths, s))

            # Update best
            if s < best_score:
                update(child_mat, child_ths, s)

        population = new_pop

    # Final local polish on the best design found by GA
    if time.time() < deadline - 10:
        local_polish(best_materials, best_ths)

    # Additional polish: try a few more random starts and polish them
    for i in range(5):
        if time.time() > deadline - 5:
            break
        mat, ths = random_design()
        s = evaluate(mat, ths)
        if s < best_score + 0.5:  # Only polish if reasonably close
            local_polish(mat, ths)

    # Final safety: ensure constraints are met
    final_ths = clamp_ths(best_ths, len(best_materials))
    if sum(final_ths) > max_total:
        final_ths = final_ths * (max_total / sum(final_ths))
    for i in range(len(final_ths)):
        final_ths[i] = max(0.0, min(max_layer, final_ths[i]))

    return list(best_materials), list(final_ths)
