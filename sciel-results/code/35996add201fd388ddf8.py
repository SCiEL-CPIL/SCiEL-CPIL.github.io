"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap
    using a pixel-level genetic algorithm with a two-tier PWE surrogate
    (5x5 fast ranking + 11x11 elite refinement).
    """
    start_time = time.time()
    deadline = start_time + 200  # safety margin (evaluator kills at 230s)

    size = problem["pixels"]
    half = size // 2
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    # K-points in the irreducible BZ (units of pi/a)
    def make_k_points(n_per_edge=5):
        pts = [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0)]
        for i in range(n_per_edge + 1):
            for j in range(n_per_edge + 1):
                kx = i / n_per_edge * 0.5
                ky = j / n_per_edge * 0.5
                if kx <= ky + 1e-9 and kx * kx + ky * ky <= 1.0 + 1e-9:
                    dup = False
                    for pkx, pky in pts:
                        if abs(pkx - kx) < 1e-9 and abs(pky - ky) < 1e-9:
                            dup = True
                            break
                    if not dup:
                        pts.append((kx, ky))
        return pts

    K_POINTS = make_k_points(n_per_edge=5)

    def fast_tm_gap(cell_array, n_pw):
        """
        TM band gap ratio via PWE with n_pw x n_pw plane-wave window.
        cell_array: 32x32 numpy array of 0/1
        """
        N = size
        eps = np.where(cell_array == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        F_full = np.fft.fft2(inv_eps) / (N * N)

        f = np.zeros((2 * n_pw + 1, 2 * n_pw + 1), dtype=complex)
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                f[m + n_pw, n + n_pw] = F_full[m % N, n % N]

        p_range = np.arange(-n_pw, n_pw + 1)
        F_f = np.fft.fft2(f)
        min_omega5 = float('inf')
        max_omega4 = 0.0
        n_modes = 2 * n_pw + 1

        for kx, ky in K_POINTS:
            kx_arr = kx + p_range
            ky_arr = ky + p_range
            K = kx_arr[:, None] ** 2 + ky_arr[None, :] ** 2
            F_K = np.fft.fft2(K)
            A = np.real(np.fft.ifft2(F_f * F_K))
            A = (A + A.T) / 2.0
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                if len(eigvals) >= 5:
                    omega4 = eigvals[3]
                    omega5 = eigvals[4]
                    if omega4 > 1e-10 and omega5 > 1e-10:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
            except Exception:
                pass

        if min_omega5 == float('inf') or max_omega4 < 1e-10:
            return 0.0
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap_ratio

    def quadrant_to_cell(quadrant):
        cell = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(half):
            for j in range(half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][size - 1 - j] = val
                cell[size - 1 - i][j] = val
                cell[size - 1 - i][size - 1 - j] = val
        return cell

    def make_superellipse(a, n_exp):
        q = [[0 for _ in range(half)] for _ in range(half)]
        cx, cy = 7.5, 7.5
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                dx, dy = abs(x - cx) / a, abs(y - cy) / a
                if dx ** n_exp + dy ** n_exp < 1.0:
                    q[i][j] = 1
        return q

    def make_circle(r):
        q = [[0 for _ in range(half)] for _ in range(half)]
        cx, cy = 7.5, 7.5
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                    q[i][j] = 1
        return q

    def make_ring(r_in, r_out):
        q = [[0 for _ in range(half)] for _ in range(half)]
        cx, cy = 7.5, 7.5
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if r_in < r < r_out:
                    q[i][j] = 1
        return q

    def make_cross(w):
        q = [[0 for _ in range(half)] for _ in range(half)]
        cx, cy = 7.5, 7.5
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x - cx) < w or abs(y - cy) < w:
                    q[i][j] = 1
        return q

    def make_square(a):
        q = [[0 for _ in range(half)] for _ in range(half)]
        cx, cy = 7.5, 7.5
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x - cx) < a and abs(y - cy) < a:
                    q[i][j] = 1
        return q

    def create_initial_population(pop_size, rng):
        population = [
            make_superellipse(5.0, 4.0),
            make_circle(4.5),
            make_square(4.5),
            make_ring(2.0, 5.5),
            make_cross(2.5),
            make_superellipse(4.5, 3.0),
            make_ring(3.0, 6.0),
            make_circle(5.0),
        ]
        while len(population) < pop_size:
            base = population[rng.randint(0, len(population) - 1)]
            q = [row[:] for row in base]
            for i in range(half):
                for j in range(half):
                    if rng.random() < 0.15:
                        q[i][j] = 1 - q[i][j]
            population.append(q)
        return population[:pop_size]

    def crossover(q1, q2, rate, rng):
        child1 = [row[:] for row in q1]
        child2 = [row[:] for row in q2]
        for i in range(half):
            for j in range(half):
                if rng.random() < rate:
                    child1[i][j], child2[i][j] = child2[i][j], child1[i][j]
        return child1, child2

    def mutate(quadrant, rate, rng):
        q = [row[:] for row in quadrant]
        for i in range(half):
            for j in range(half):
                if rng.random() < rate:
                    q[i][j] = 1 - q[i][j]
        return q

    def evaluate_quadrant_fast(quadrant):
        cell = quadrant_to_cell(quadrant)
        cell_arr = np.array(cell, dtype=np.float64)
        return fast_tm_gap(cell_arr, n_pw=2)  # 5x5 window

    def evaluate_quadrant_accurate(quadrant):
        cell = quadrant_to_cell(quadrant)
        cell_arr = np.array(cell, dtype=np.float64)
        return fast_tm_gap(cell_arr, n_pw=5)  # 11x11 window

    rng = random.Random(123)
    pop_size = 18
    population = create_initial_population(pop_size, rng)

    scores = [evaluate_quadrant_fast(q) for q in population]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_quadrant = [row[:] for row in population[best_idx]]

    generation = 0
    max_generations = 60
    stagnation_count = 0
    stagnation_limit = 8

    while generation < max_generations:
        if time.time() > deadline - 40:
            break

        sorted_indices = np.argsort(scores)[::-1]
        elite_count = max(4, pop_size // 3)
        new_population = [population[i][:] for i in sorted_indices[:elite_count]]

        while len(new_population) < pop_size:
            if time.time() > deadline - 40:
                break
            t_size = min(6, len(sorted_indices))
            p1_idx = sorted_indices[rng.randint(0, t_size - 1)]
            p2_idx = sorted_indices[rng.randint(0, t_size - 1)]
            c1, c2 = crossover(population[p1_idx], population[p2_idx], rate=0.25, rng=rng)
            c1 = mutate(c1, rate=0.03, rng=rng)
            c2 = mutate(c2, rate=0.03, rng=rng)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population[:pop_size]
        scores = [evaluate_quadrant_fast(q) for q in population]

        gen_best_idx = int(np.argmax(scores))
        if scores[gen_best_idx] > best_score + 1e-9:
            best_score = scores[gen_best_idx]
            best_quadrant = [row[:] for row in population[gen_best_idx]]
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Stagnation: inject fresh diverse topologies to escape local basin
        if stagnation_count >= stagnation_limit and time.time() < deadline - 40:
            for _ in range(3):
                new_q = create_initial_population(1, rng)[0]
                new_q = mutate(new_q, rate=0.2, rng=rng)
                population[-1] = new_q
                scores[-1] = evaluate_quadrant_fast(new_q)
            stagnation_count = 0

        generation += 1

    # Elite refinement with accurate 11x11 solver + local pixel-flip search
    if time.time() < deadline - 8:
        # Re-evaluate best with accurate solver
        acc_best = evaluate_quadrant_accurate(best_quadrant)
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            for i in range(half):
                if time.time() > deadline - 8:
                    break
                for j in range(half):
                    if time.time() > deadline - 8:
                        break
                    q = [row[:] for row in best_quadrant]
                    q[i][j] = 1 - q[i][j]
                    score = evaluate_quadrant_accurate(q)
                    if score > acc_best:
                        acc_best = score
                        best_quadrant = q
                        improved = True

    # Also try a few top candidates from the GA with accurate refinement
    if time.time() < deadline - 8:
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices[:5]:
            if time.time() > deadline - 15:
                break
            q = [row[:] for row in population[idx]]
            acc = evaluate_quadrant_accurate(q)
            if acc > acc_best:
                acc_best = acc
                best_quadrant = q
                # local refine
                improved = True
                while improved and time.time() < deadline - 8:
                    improved = False
                    for i in range(half):
                        if time.time() > deadline - 8:
                            break
                        for j in range(half):
                            if time.time() > deadline - 8:
                                break
                            q2 = [row[:] for row in best_quadrant]
                            q2[i][j] = 1 - q2[i][j]
                            s2 = evaluate_quadrant_accurate(q2)
                            if s2 > acc_best:
                                acc_best = s2
                                best_quadrant = q2
                                improved = True

    best_cell = quadrant_to_cell(best_quadrant)

    # Verify symmetry
    for i in range(size):
        for j in range(size):
            if best_cell[i][j] != best_cell[size - 1 - i][j]:
                raise ValueError("Row symmetry failed")
            if best_cell[i][j] != best_cell[i][size - 1 - j]:
                raise ValueError("Col symmetry failed")
            if best_cell[i][j] != best_cell[j][i]:
                raise ValueError("Transpose symmetry failed")

    return best_cell
# EVOLVE-BLOCK-END
