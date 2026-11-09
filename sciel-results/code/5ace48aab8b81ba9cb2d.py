"""Broadband antireflection coating design: parametric index-profile + multi-start NM.

Strategy:
  1. Parametric index profiles (cubic/quartic polynomials) mapped to L/H layers.
  2. Multi-start Nelder-Mead on thicknesses for each material pattern.
  3. Global DE on a few promising patterns.
  4. Final local polish.
"""
import time
import numpy as np


def _reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (%) over the band using characteristic matrix method."""
    if not materials:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        return 100.0 * ((ns - n0) / (ns + n0)) ** 2

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    n_wl = 401
    wl = np.linspace(band[0], band[1], n_wl)

    n_mat = np.array([problem["materials"][m] for m in materials])
    d = np.array(thicknesses, dtype=float)
    N = len(materials)

    # Phase delta for each layer at each wavelength
    delta = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wl[None, :]

    # Characteristic matrix product
    A = np.ones(n_wl, dtype=complex)
    B = np.zeros(n_wl, dtype=complex)
    C = np.zeros(n_wl, dtype=complex)
    D = np.ones(n_wl, dtype=complex)

    for i in range(N):
        c = np.cos(delta[i])
        s = np.sin(delta[i])
        n_i = n_mat[i]

        A_new = A * c + 1j * B * n_i * s
        B_new = 1j * A * s / n_i + B * c
        C_new = C * c + 1j * D * n_i * s
        D_new = 1j * C * s / n_i + D * c
        A, B, C, D = A_new, B_new, C_new, D_new

    # Reflectance
    num = n0 * (A + B * ns) - (C + D * ns)
    den = n0 * (A + B * ns) + (C + D * ns)
    r = num / den
    R = np.abs(r) ** 2
    return np.mean(R) * 100.0


def _valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


def _clamp_thicknesses(thicknesses, problem):
    """Clamp thicknesses to valid range and ensure total doesn't exceed max."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    clamped = [max(0.0, min(max_layer_nm, t)) for t in thicknesses]

    total = sum(clamped)
    if total > max_total_nm:
        scale = max_total_nm / total
        clamped = [t * scale for t in clamped]

    return clamped


def _nelder_mead(objective, x0, bounds, maxiter=150, deadline=None):
    """Nelder-Mead simplex optimization with bounds and deadline."""
    n = len(x0)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Initial simplex with larger steps
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        span = bounds[i][1] - bounds[i][0]
        step = 0.1 * span if span > 1 else 0.05
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)

    fvals = np.array([objective(simplex[i]) for i in range(n + 1)])
    best_idx = np.argmin(fvals)
    fbest = fvals[best_idx]
    xbest = simplex[best_idx].copy()

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(maxiter):
        if deadline and time.time() > deadline:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]

        if fvals[0] < fbest:
            fbest = fvals[0]
            xbest = simplex[0].copy()

        # Check convergence
        if np.allclose(fvals, fvals[0], rtol=1e-7, atol=1e-7):
            break

        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = objective(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - worst)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])

    return xbest, fbest


def _differential_evolution(objective, bounds, pop_size=12, max_iters=25, F=0.8, CR=0.7, deadline=None):
    """Differential evolution for global search."""
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fitness = np.array([objective(pop[i]) for i in range(pop_size)])
    best_idx = np.argmin(fitness)
    xbest = pop[best_idx].copy()
    fbest = fitness[best_idx]

    for gen in range(max_iters):
        if deadline and time.time() > deadline:
            break

        for i in range(pop_size):
            if deadline and time.time() > deadline:
                break

            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)

            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)

            trial = pop[i].copy()
            for j in range(n):
                if np.random.random() < CR:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            trial = np.clip(trial, lo, hi)

            ft = objective(trial)
            if ft < fitness[i]:
                pop[i] = trial
                fitness[i] = ft
                if ft < fbest:
                    fbest = ft
                    xbest = trial.copy()

    return xbest, fbest


def _parametric_profiles(problem, n_layers):
    """Generate material patterns from parametric index profiles.
    
    Maps a continuous index profile n(x) from x=0 (air side) to x=1 (substrate side)
    to discrete L/H layers. The profile is a polynomial: n(x) = sum a_i * x^i.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_mid = (nL + nH) / 2.0

    # Target: index should transition from near n0 (air=1.0) to near ns (glass=1.52)
    # Since we only have L=1.38 and H=2.35, the best we can do is use L near air
    # and H near glass, or mix them.
    
    profiles = []
    
    # Cubic polynomial: n(x) = a0 + a1*x + a2*x^2 + a3*x^3
    # We want n(0) ~ n0=1.0, n(1) ~ ns=1.52
    # But since our materials are 1.38 and 2.35, we'll map:
    # If n(x) < n_mid -> 'L', else 'H'
    
    # Profile 1: Linear from L to H
    for _ in range(3):
        a0 = np.random.uniform(nL, n_mid)
        a1 = np.random.uniform(0, nH - a0)
        coeffs = [a0, a1, 0, 0]
        xs = np.linspace(0, 1, n_layers)
        n_vals = a0 + a1 * xs
        pattern = ['L' if nv < n_mid else 'H' for nv in n_vals]
        profiles.append(pattern)
    
    # Profile 2: Cubic with more control
    for _ in range(5):
        a0 = np.random.uniform(nL, n_mid)
        a1 = np.random.uniform(-1, 1) * (nH - nL) * 0.5
        a2 = np.random.uniform(-1, 1) * (nH - nL) * 0.3
        a3 = np.random.uniform(-1, 1) * (nH - nL) * 0.2
        xs = np.linspace(0, 1, n_layers)
        n_vals = a0 + a1 * xs + a2 * xs**2 + a3 * xs**3
        # Clamp to [nL, nH]
        n_vals = np.clip(n_vals, nL, nH)
        pattern = ['L' if nv < n_mid else 'H' for nv in n_vals]
        profiles.append(pattern)
    
    # Profile 3: S-shaped (sigmoid-like) transition
    for _ in range(3):
        x0 = np.random.uniform(0.3, 0.7)
        k = np.random.uniform(5, 15)
        xs = np.linspace(0, 1, n_layers)
        sig = 1.0 / (1.0 + np.exp(-k * (xs - x0)))
        n_vals = nL + (nH - nL) * sig
        pattern = ['L' if nv < n_mid else 'H' for nv in n_vals]
        profiles.append(pattern)
    
    # Profile 4: Alternating with graded thickness (just generate patterns)
    for length in [n_layers]:
        for _ in range(3):
            pattern = []
            last = 'L' if np.random.random() < 0.5 else 'H'
            for _ in range(length):
                pattern.append(last)
                if np.random.random() < 0.7:
                    last = 'H' if last == 'L' else 'L'
            profiles.append(pattern)
    
    # Quarter-wave based patterns
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    qwL = centre / (4.0 * nL)
    qwH = centre / (4.0 * nH)
    
    qw_patterns = [
        ['L', 'H', 'L', 'H'][:n_layers],
        ['H', 'L', 'H', 'L'][:n_layers],
        ['L', 'H', 'L', 'H', 'L'][:n_layers],
        ['H', 'L', 'H', 'L', 'H'][:n_layers],
        ['L', 'H', 'L', 'H', 'L', 'H'][:n_layers],
    ]
    profiles.extend(qw_patterns)
    
    return profiles


def _initial_thicknesses(problem, materials):
    """Generate initial thicknesses based on quarter-wave at center wavelength."""
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    qwL = centre / (4.0 * nL)
    qwH = centre / (4.0 * nH)
    
    thicknesses = []
    for m in materials:
        if m == 'L':
            thicknesses.append(np.random.uniform(0.7 * qwL, 1.3 * qwL))
        else:
            thicknesses.append(np.random.uniform(0.7 * qwH, 1.3 * qwH))
    
    return _clamp_thicknesses(thicknesses, problem)


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # Try different numbers of layers
    best_materials = ['L']
    best_thicknesses = [centre / (4.0 * nL)]
    best_score = _reflectance(best_materials, best_thicknesses, problem)

    # Generate candidate patterns for each layer count
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        if time.time() > deadline - 30:
            break

        profiles = _parametric_profiles(problem, n_layers)
        
        # Evaluate initial scores for all profiles
        candidates = []
        for pattern in profiles:
            if len(pattern) != n_layers:
                continue
            thicknesses = _initial_thicknesses(problem, pattern)
            if not _valid(pattern, thicknesses, problem):
                continue
            score = _reflectance(pattern, thicknesses, problem)
            candidates.append((score, pattern, thicknesses))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])

        # Optimize top candidates with Nelder-Mead
        for score, mats, thicks in candidates[:6]:
            if time.time() > deadline - 20:
                break

            def objective(x, mats=mats):
                if time.time() > deadline:
                    return 100.0
                t = _clamp_thicknesses(x, problem)
                if not _valid(mats, t, problem):
                    return 100.0
                return _reflectance(mats, t, problem)

            x0 = np.array(thicks, dtype=float)
            bounds = [(0.0, max_layer_nm)] * len(x0)

            try:
                x_opt, f_opt = _nelder_mead(objective, x0, bounds, maxiter=100, deadline=deadline)
                t_opt = _clamp_thicknesses(x_opt, problem)
                if _valid(mats, t_opt, problem):
                    f_opt_check = _reflectance(mats, t_opt, problem)
                    if f_opt_check < best_score:
                        best_materials = mats
                        best_thicknesses = t_opt
                        best_score = f_opt_check
            except Exception:
                pass

    # DE on the best pattern found so far
    if time.time() < deadline - 25:
        n = len(best_materials)
        if n > 0:
            def objective(x):
                if time.time() > deadline:
                    return 100.0
                t = _clamp_thicknesses(x, problem)
                if not _valid(best_materials, t, problem):
                    return 100.0
                return _reflectance(best_materials, t, problem)

            x0 = np.array(best_thicknesses, dtype=float)
            bounds = [(0.0, max_layer_nm)] * n

            try:
                x_opt, f_opt = _differential_evolution(objective, bounds, pop_size=10, max_iters=20, deadline=deadline)
                t_opt = _clamp_thicknesses(x_opt, problem)
                if _valid(best_materials, t_opt, problem):
                    f_opt_check = _reflectance(best_materials, t_opt, problem)
                    if f_opt_check < best_score:
                        best_thicknesses = t_opt
                        best_score = f_opt_check
            except Exception:
                pass

    # Final local polish
    if time.time() < deadline - 5:
        n = len(best_materials)
        if n > 0:
            def objective(x):
                if time.time() > deadline:
                    return 100.0
                t = _clamp_thicknesses(x, problem)
                if not _valid(best_materials, t, problem):
                    return 100.0
                return _reflectance(best_materials, t, problem)

            x0 = np.array(best_thicknesses, dtype=float)
            bounds = [(0.0, max_layer_nm)] * n

            try:
                x_opt, f_opt = _nelder_mead(objective, x0, bounds, maxiter=80, deadline=deadline)
                t_opt = _clamp_thicknesses(x_opt, problem)
                if _valid(best_materials, t_opt, problem):
                    f_opt_check = _reflectance(best_materials, t_opt, problem)
                    if f_opt_check < best_score:
                        best_thicknesses = t_opt
                        best_score = f_opt_check
            except Exception:
                pass

    # Final validation
    best_thicknesses = _clamp_thicknesses(best_thicknesses, problem)
    if not _valid(best_materials, best_thicknesses, problem):
        best_materials = ['L']
        best_thicknesses = [centre / (4.0 * nL)]

    return best_materials, best_thicknesses
