"""Broadband antireflection coating design: gradient-profile + multi-start local search.

Strategy:
  1. Model the coating as a discretized refractive index profile (20 layers) that
     approximates a continuous exponential or power-law gradient from air to substrate.
  2. Use binary search over layer indices to assign each of the 20 physical layers
     to either 'L' (n=1.38) or 'H' (n=2.35) based on the target index at that position.
  3. Optimize the physical thicknesses of this fixed 20-layer structure using
     multi-start Nelder-Mead local search, initialized with the analytical quarter-wave
     stack solution and diverse random perturbations.
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


def _gradient_materials(problem, n_layers=20, profile='exp'):
    """
    Generate a 20-layer material sequence that approximates a continuous index gradient.
    
    The target index at position i (0-indexed, from air to substrate) is:
      - 'exp': n(i) = n0 * (ns/n0)^(i/(n_layers-1))
      - 'power': n(i) = n0 + (ns - n0) * (i/(n_layers-1))^p
    
    Each layer is assigned to 'L' or 'H' by binary search: pick whichever of
    the two available indices is closer to the target index at that position.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    materials = []
    for i in range(n_layers):
        t = i / (n_layers - 1) if n_layers > 1 else 0.5
        if profile == 'exp':
            n_target = n0 * (ns / n0) ** t
        elif profile == 'power':
            p = 1.5
            n_target = n0 + (ns - n0) * (t ** p)
        else:
            n_target = n0 + (ns - n0) * t
        
        # Binary search: pick closer of L or H
        if abs(n_target - nL) <= abs(n_target - nH):
            materials.append('L')
        else:
            materials.append('H')
    
    return materials


def _nelder_mead(objective, x0, bounds, maxiter=200, timeout=None, start_time=None):
    """Nelder-Mead simplex optimization with bounds."""
    if start_time is None:
        start_time = time.time()

    n = len(x0)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 0.05
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
        if timeout and time.time() - start_time > timeout:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]

        if fvals[0] < fbest:
            fbest = fvals[0]
            xbest = simplex[0].copy()

        # Check convergence
        if np.allclose(fvals, fvals[0], rtol=1e-6, atol=1e-6):
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


def _quarter_wave_thicknesses(materials, problem):
    """Compute quarter-wave thicknesses at the center wavelength."""
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    thicknesses = []
    for m in materials:
        n = problem["materials"][m]
        thicknesses.append(centre / (4.0 * n))
    return thicknesses


def _multi_start_nm(materials, problem, deadline, n_starts=8):
    """
    Multi-start Nelder-Mead optimization for thicknesses of a fixed material sequence.
    """
    max_layer_nm = problem["max_layer_nm"]
    n = len(materials)
    bounds = [(0.0, max_layer_nm)] * n

    # Initial guess: quarter-wave
    x0_qw = np.array(_quarter_wave_thicknesses(materials, problem), dtype=float)
    x0_qw = np.clip(x0_qw, 0, max_layer_nm)

    # Generate multiple starting points
    starts = [x0_qw.copy()]

    # Add perturbed quarter-wave
    for _ in range(n_starts - 1):
        perturbation = np.random.uniform(-0.3, 0.3, size=n)
        x_start = x0_qw * (1.0 + perturbation)
        x_start = np.clip(x_start, 0, max_layer_nm)
        starts.append(x_start)

    # Add random starts
    for _ in range(3):
        x_start = np.random.uniform(10, max_layer_nm, size=n)
        starts.append(x_start)

    best_x = None
    best_f = float('inf')

    for x_start in starts:
        if time.time() > deadline:
            break

        def objective(x):
            if time.time() > deadline:
                return 100.0
            t = _clamp_thicknesses(x, problem)
            if not _valid(materials, t, problem):
                return 100.0
            return _reflectance(materials, t, problem)

        try:
            x_opt, f_opt = _nelder_mead(objective, x_start, bounds, maxiter=150,
                                        timeout=max(0.1, deadline - time.time()))
            t_opt = _clamp_thicknesses(x_opt, problem)
            if _valid(materials, t_opt, problem):
                f_opt_check = _reflectance(materials, t_opt, problem)
                if f_opt_check < best_f:
                    best_f = f_opt_check
                    best_x = t_opt
        except Exception:
            pass

    if best_x is None:
        t = _clamp_thicknesses(x0_qw, problem)
        return materials, t, _reflectance(materials, t, problem)

    return materials, best_x, best_f


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # Generate gradient-based material sequences
    profiles = ['exp', 'power', 'linear']
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')

    for profile in profiles:
        if time.time() > deadline - 20:
            break

        materials = _gradient_materials(problem, n_layers=max_layers, profile=profile)

        # Validate
        if not _valid(materials, [max_layer_nm] * len(materials), problem):
            continue

        # Multi-start Nelder-Mead optimization
        opt_mats, opt_thicks, opt_score = _multi_start_nm(materials, problem, deadline, n_starts=6)

        if opt_score < best_score:
            best_score = opt_score
            best_materials = opt_mats
            best_thicknesses = opt_thicks

    # Also try a few alternative material sequences with different layer counts
    for n_layers in [15, 10, 8]:
        if time.time() > deadline - 15:
            break

        for profile in ['exp', 'power']:
            if time.time() > deadline - 10:
                break

            materials = _gradient_materials(problem, n_layers=n_layers, profile=profile)
            if not _valid(materials, [max_layer_nm] * len(materials), problem):
                continue

            opt_mats, opt_thicks, opt_score = _multi_start_nm(materials, problem, deadline, n_starts=4)

            if opt_score < best_score:
                best_score = opt_score
                best_materials = opt_mats
                best_thicknesses = opt_thicks

    # Fallback: quarter-wave 4-layer stack
    if best_materials is None:
        band = problem["band_nm"]
        centre = 0.5 * (band[0] + band[1])
        best_materials = ['L', 'H', 'L', 'H']
        best_thicknesses = [centre / (4.0 * nL), centre / (4.0 * nH),
                            centre / (4.0 * nL), centre / (4.0 * nH)]
        best_thicknesses = _clamp_thicknesses(best_thicknesses, problem)
        best_score = _reflectance(best_materials, best_thicknesses, problem)

    # Final validation
    best_thicknesses = _clamp_thicknesses(best_thicknesses, problem)
    if not _valid(best_materials, best_thicknesses, problem):
        band = problem["band_nm"]
        centre = 0.5 * (band[0] + band[1])
        best_materials = ['L']
        best_thicknesses = [centre / (4.0 * nL)]

    return best_materials, best_thicknesses
