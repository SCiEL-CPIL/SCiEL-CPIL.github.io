"""Antireflection coating design via graded-index profile + Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# --- Reflectance computation (characteristic-matrix method) ---

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, band_nm, n_wavelengths=401):
    """Compute mean normal-incidence reflectance (percent) over the band."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r0 * r0

    wls = np.linspace(band_nm[0], band_nm[1], n_wavelengths)
    k = 2.0 * math.pi / wls  # 1/nm

    # Build characteristic matrices for all layers at all wavelengths
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # delta_j = k * n_j * d_j

    n_L = materials['L']
    n_H = materials['H']

    # We'll precompute for each layer
    # Total matrix = product of all layer matrices, then combine with substrate
    # Y = M_total * n_s (for admittance)
    # r = (Y - n_0) / (Y + n_0)

    # For efficiency, compute per-wavelength
    reflectances = np.zeros(n_wavelengths)

    for i, wl in enumerate(wls):
        # Start with identity matrix
        M_total = np.array([[1.0 + 0j, 0.0 + 0j],
                            [0.0 + 0j, 1.0 + 0j]])
        for j in range(n_layers):
            n_j = n_L if j % 2 == 0 else n_H  # alternating L, H, L, H...
            d_j = thicknesses[j]
            delta = k[i] * n_j * d_j
            c = math.cos(delta)
            s = math.sin(delta)
            M_j = np.array([[c + 0j, 1j * s / n_j],
                            [1j * n_j * s, c + 0j]])
            M_total = M_total @ M_j

        # Combine with substrate
        n_s = substrate_index
        Y = M_total[0, 0] * n_s + M_total[0, 1]
        r = (Y - ambient_index) / (Y + ambient_index)
        reflectances[i] = abs(r) ** 2

    return float(np.mean(reflectances) * 100.0)


def _mean_reflectance_fast(thicknesses, layer_indices, materials, ambient_index, substrate_index, band_nm, n_wavelengths=401):
    """Fast version with precomputed layer indices."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r0 * r0

    wls = np.linspace(band_nm[0], band_nm[1], n_wavelengths)
    k = 2.0 * math.pi / wls

    n_L = materials['L']
    n_H = materials['H']
    n_s = substrate_index
    n_0 = ambient_index

    # Precompute n_j for each layer
    n_arr = np.array([n_L if idx == 0 else n_H for idx in layer_indices])

    reflectances = np.zeros(n_wavelengths)

    for i in range(n_wavelengths):
        M_total = np.array([[1.0 + 0j, 0.0 + 0j],
                            [0.0 + 0j, 1.0 + 0j]])
        ki = k[i]
        for j in range(n_layers):
            delta = ki * n_arr[j] * thicknesses[j]
            c = math.cos(delta)
            s = math.sin(delta)
            M_j = np.array([[c + 0j, 1j * s / n_arr[j]],
                            [1j * n_arr[j] * s, c + 0j]])
            M_total = M_total @ M_j

        Y = M_total[0, 0] * n_s + M_total[0, 1]
        r = (Y - n_0) / (Y + n_0)
        reflectances[i] = abs(r) ** 2

    return float(np.mean(reflectances) * 100.0)


# --- Nelder-Mead optimizer ---

def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-8):
    """Simple Nelder-Mead optimizer with bounds."""
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        step = 0.05 * (bounds[1][i] - bounds[0][i]) if bounds[1][i] > bounds[0][i] else 0.1
        x[i] += step
        simplex.append(x)

    # Evaluate initial simplex
    values = [objective(x) for x in simplex]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(max_iter):
        # Sort simplex by objective value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break

        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        # Clip to bounds
        x_r = np.clip(x_r, bounds[0], bounds[1])
        f_r = objective(x_r)

        if f_r < values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds[0], bounds[1])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds[0], bounds[1])
            f_c = objective(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    values[i] = objective(simplex[i])

    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


# --- Design function ---

def design(problem):
    """Design an antireflection coating using graded-index profile + Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    n_L = materials['L']
    n_H = materials['H']

    # --- Step 1: Define graded-index layer sequence ---
    # Use a graded profile: start with L (closer to air), end with H (closer to glass)
    # This approximates a continuous index gradient from ~1.0 to ~1.52
    # Alternating L/H with graded thicknesses

    n_layers = max_layers  # Use all 20 layers

    # Graded index profile: layer index alternates L, H, L, H, ...
    # But we want the effective index to gradually increase
    # Strategy: use pairs of (L, H) with increasing H thickness relative to L
    # This creates a gradual transition

    # Layer sequence: L, H, L, H, ..., L, H (10 pairs)
    layer_indices = []
    for i in range(n_layers // 2):
        layer_indices.append(0)  # L
        layer_indices.append(1)  # H
    if n_layers % 2 == 1:
        layer_indices.append(0)  # Extra L

    # --- Step 2: Initial guess from quarter-wave stack ---
    centre_wl = 0.5 * (band_nm[0] + band_nm[1])
    d_L_qw = centre_wl / (4.0 * n_L)
    d_H_qw = centre_wl / (4.0 * n_H)

    # Initial thicknesses: quarter-wave for all layers
    x0 = []
    for idx in layer_indices:
        if idx == 0:
            x0.append(d_L_qw)
        else:
            x0.append(d_H_qw)

    # Scale to respect total thickness constraint
    total = sum(x0)
    if total > max_total_nm:
        scale = max_total_nm / total
        x0 = [x * scale for x in x0]

    x0 = np.array(x0)
    bounds = (np.zeros(n_layers), np.full(n_layers, max_layer_nm))

    # --- Step 3: Objective function ---
    def objective(x):
        x = np.clip(x, 0, max_layer_nm)
        # Check total thickness
        if np.sum(x) > max_total_nm + 1e-6:
            return 1e6  # Penalty
        return _mean_reflectance_fast(x, layer_indices, materials, ambient_index, substrate_index, band_nm)

    # Evaluate initial solution
    best_x = x0.copy()
    best_val = objective(best_x)

    # --- Step 4: Multi-start Nelder-Mead ---
    n_starts = 5
    np.random.seed(42)

    for start in range(n_starts):
        if time.time() > deadline - 20.0:
            break

        if start == 0:
            x_init = x0.copy()
        else:
            # Perturb initial solution
            perturbation = 0.1 * (max_layer_nm - 0) * np.random.rand(n_layers)
            x_init = np.clip(x0 + perturbation, 0, max_layer_nm)
            # Scale to respect total
            if np.sum(x_init) > max_total_nm:
                x_init = x_init * (max_total_nm / np.sum(x_init))

        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(objective, x_init, bounds, max_iter=300, tol=1e-7)
            if f_opt < best_val:
                best_val = f_opt
                best_x = x_opt
        except Exception:
            pass

    # --- Step 5: Local refinement with smaller steps ---
    if time.time() < deadline - 10.0:
        # Fine-tune with smaller perturbation
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            perturbation = 0.02 * max_layer_nm * np.random.rand(n_layers)
            x_init = np.clip(best_x + perturbation, 0, max_layer_nm)
            if np.sum(x_init) > max_total_nm:
                x_init = x_init * (max_total_nm / np.sum(x_init))
            try:
                x_opt, f_opt = _nelder_mead(objective, x_init, bounds, max_iter=200, tol=1e-8)
                if f_opt < best_val:
                    best_val = f_opt
                    best_x = x_opt
            except Exception:
                pass

    # --- Step 6: Convert to material labels and thicknesses ---
    # Trim zero-thickness layers
    materials_list = []
    thicknesses_list = []
    for i in range(n_layers):
        if best_x[i] > 1e-6:  # Only include non-negligible layers
            if layer_indices[i] == 0:
                materials_list.append('L')
            else:
                materials_list.append('H')
            thicknesses_list.append(float(best_x[i]))

    # Ensure constraints
    if len(materials_list) > max_layers:
        materials_list = materials_list[:max_layers]
        thicknesses_list = thicknesses_list[:max_layers]

    total_thick = sum(thicknesses_list)
    if total_thick > max_total_nm:
        scale = max_total_nm / total_thick
        thicknesses_list = [t * scale for t in thicknesses_list]

    # Clamp each thickness
    thicknesses_list = [min(max(t, 0.0), max_layer_nm) for t in thicknesses_list]

    if len(materials_list) == 0:
        # Fallback: single quarter-wave layer
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]

    return materials_list, thicknesses_list

# EVOLVE-BLOCK-END
