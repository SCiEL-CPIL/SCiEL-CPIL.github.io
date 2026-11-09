"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. A pool of physically-motivated layer sequences (quarter-wave stacks, symmetric
     multi-layer, graded, and "random-ish" patterns) so we explore many structures.
  3. Stage A: coarse global search over layer counts + a few thickness parameters
     (coarse grid + random restarts) to find good basins.
  4. Stage B: Nelder-Mead local polish on the best candidates (and a few others).
  5. Explicit deadline guard: return the best feasible result found so far.
"""

import time
import math
import random
import numpy as np


def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Build per-layer optical path lengths
    # delta_j = 2*pi*n_j*d_j/lambda
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # 150s kill, 15s safety

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    def make_initial_thicknesses(pattern):
        ths = []
        for m in pattern:
            ths.append(qw[m])
        return ths

    def constrain_thicknesses(ths, pattern):
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        # Re-check sum after clamping
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        return ths

    def nelder_mead(materials_pattern, init_ths, problem, deadline, max_iters=500):
        """Nelder-Mead simplex local optimization."""
        ths = list(init_ths)
        ths = constrain_thicknesses(ths, materials_pattern)
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Build initial simplex
        step = 10.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)

        # Evaluate simplex
        scores = [_mean_reflectance(materials_pattern, simplex[i], problem) for i in range(n + 1)]

        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            # Sort by score
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]

            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = constrain_thicknesses(list(xr), materials_pattern)
            fr = _mean_reflectance(materials_pattern, xr, problem)

            if fr < scores[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain_thicknesses(list(xe), materials_pattern)
                fe = _mean_reflectance(materials_pattern, xe, problem)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                # Contraction
                if fr < scores[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = constrain_thicknesses(list(xc), materials_pattern)
                    fc = _mean_reflectance(materials_pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain_thicknesses(simplex[i], materials_pattern)
                        scores[i] = _mean_reflectance(materials_pattern, simplex[i], problem)
                    # Check convergence
                    if np.max(scores) - np.min(scores) < 1e-6:
                        break

        return materials_pattern, best_ths, best_score

    def simulated_annealing(materials_pattern, init_ths, problem, deadline, max_iters=400):
        ths = list(init_ths)
        ths = constrain_thicknesses(ths, materials_pattern)
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        current_score = best_score
        n = len(ths)

        T_start = 2.0
        T_end = 0.005
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)

        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            T = T_start * (cooling_rate ** it)

            idx = random.randrange(n)
            step = random.uniform(-100, 100)
            new_ths = list(ths)
            new_ths[idx] = max(0.0, min(max_layer, ths[idx] + step))
            new_ths = constrain_thicknesses(new_ths, materials_pattern)
            new_score = _mean_reflectance(materials_pattern, new_ths, problem)

            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / (T + 1e-9)):
                ths = new_ths
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_ths = list(ths)

        return materials_pattern, best_ths, best_score

    # Generate candidate patterns
    candidates = []

    # 1-layer
    candidates.append(("L",))
    candidates.append(("H",))

    # 2-layer
    for a in ("L", "H"):
        for b in ("L", "H"):
            if a != b:
                candidates.append((a, b))

    # 3-layer
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                candidates.append((a, b, c))

    # 4-layer symmetric: A B B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            if a != b:
                candidates.append((a, b, b, a))

    # 5-layer symmetric: A B C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                candidates.append((a, b, c, b, a))

    # 6-layer symmetric: A B C C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                if a != c:
                    candidates.append((a, b, c, c, b, a))

    # 7-layer symmetric: A B C D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    candidates.append((a, b, c, d, c, b, a))

    # 8-layer symmetric: A B C D D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    candidates.append((a, b, c, d, d, c, b, a))

    # 9-layer symmetric: A B C D E D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    for e in ("L", "H"):
                        candidates.append((a, b, c, d, e, d, c, b, a))

    # 10-layer symmetric: A B C D E E D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    for e in ("L", "H"):
                        candidates.append((a, b, c, d, e, e, d, c, b, a))

    # Random longer patterns
    random.seed(42)
    for _ in range(40):
        length = random.randint(4, min(max_layers, 14))
        pattern = tuple(random.choice("LH") for _ in range(length))
        candidates.append(pattern)

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    candidates = unique_candidates

    # Sort by length ascending
    candidates.sort(key=lambda p: len(p))

    best_result = None
    best_score = float('inf')

    # Phase 1: SA exploration on all candidates
    for pattern in candidates:
        if time.time() >= deadline - 60:
            break

        init_ths = make_initial_thicknesses(pattern)

        mats, ths, score = simulated_annealing(pattern, init_ths, problem, deadline - 60, max_iters=200)
        if score < best_score:
            best_score = score
            best_result = (mats, ths)

    # Phase 2: Polish the best few candidates
    if best_result is not None:
        # Collect top candidates for polishing
        top_patterns = []
        for pattern in candidates:
            if time.time() >= deadline - 30:
                break
            init_ths = make_initial_thicknesses(pattern)
            # Quick SA to get a starting point
            mats, ths, score = simulated_annealing(pattern, init_ths, problem, deadline - 30, max_iters=100)
            top_patterns.append((pattern, ths, score))

        top_patterns.sort(key=lambda x: x[2])

        for pattern, ths, score in top_patterns[:10]:
            if time.time() >= deadline - 10:
                break
            mats, polished_ths, polished_score = nelder_mead(pattern, ths, problem, deadline - 10, max_iters=300)
            if polished_score < best_score:
                best_score = polished_score
                best_result = (mats, polished_ths)

    if best_result is None:
        return ["L"], [qw["L"]]

    mats, ths = best_result
    return list(mats), list(ths)
