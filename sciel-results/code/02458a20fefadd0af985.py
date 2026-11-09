"""Design a broadband antireflection coating using simulated annealing with a
fast characteristic-matrix evaluator."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np


def _compute_reflectance(materials, thicknesses, wavelengths, n0, ns):
    """
    Compute reflectance spectrum using the characteristic matrix method.
    Returns reflectance in percent.
    """
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return (r * r) * 100.0

    # Precompute refractive indices for each layer
    n = np.array([materials[m] for m in materials], dtype=np.float64)
    t = np.array(thicknesses, dtype=np.float64)

    # wavelengths: array of shape (N,)
    # We compute for each wavelength
    N = len(wavelengths)
    reflectances = np.empty(N, dtype=np.float64)

    for i, lam in enumerate(wavelengths):
        # Phase thickness delta_j = 2*pi*n_j*t_j / lam
        delta = 2.0 * np.pi * n * t / lam

        # Build characteristic matrix
        # M = product over layers of [[cos(delta_j), i*sin(delta_j)/n_j],
        #                              [i*n_j*sin(delta_j), cos(delta_j)]]
        # For normal incidence, s-polarization = p-polarization
        # M_total = M_1 * M_2 * ... * M_N
        # Then Y = M11/M22 (wait, no)
        # Actually:
        # [B, C]^T = M * [1, 0]^T  (at substrate, the field is [1,0] for
        #   normalized amplitude and derivative)
        # Wait, let me be more careful.
        #
        # The characteristic matrix for layer j is:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        #
        # Total matrix M = M_1 * M_2 * ... * M_N
        #
        # At the substrate side, the boundary condition gives:
        # [B, C]^T = M * [1, 0]^T = [M11, M21]^T
        #
        # The admittance Y = C/B = M21/M11
        #
        # Reflectance amplitude: r = (n0 - Y) / (n0 + Y)
        # Reflectance: R = |r|^2

        # Build the total matrix
        # Start with identity
        M11 = 1.0 + 0j
        M12 = 0.0 + 0j
        M21 = 0.0 + 0j
        M22 = 1.0 + 0j

        for j in range(n_layers):
            c = math.cos(delta[j])
            s = math.sin(delta[j])
            # M_j = [[c, i*s/n_j], [i*n_j*s, c]]
            m11 = c
            m12 = 1j * s / n[j]
            m21 = 1j * n[j] * s
            m22 = c

            # New M = M * M_j
            n11 = M11 * m11 + M12 * m21
            n12 = M11 * m12 + M12 * m22
            n21 = M21 * m11 + M22 * m21
            n22 = M21 * m12 + M22 * m22

            M11 = n11
            M12 = n12
            M21 = n21
            M22 = n22

        # B = M11, C = M21
        Y = M21 / M11
        r = (n0 - Y) / (n0 + Y)
        R = abs(r) ** 2
        reflectances[i] = R * 100.0

    return reflectances


def _mean_reflectance(materials, thicknesses, wavelengths, n0, ns):
    """Compute mean reflectance in percent."""
    refl = _compute_reflectance(materials, thicknesses, wavelengths, n0, ns)
    return float(np.mean(refl))


def _clamp_thicknesses(thicknesses, max_layer_nm, max_total_nm):
    """Ensure thicknesses are within bounds."""
    t = np.array(thicknesses, dtype=np.float64)
    t = np.clip(t, 0.0, max_layer_nm)
    total = np.sum(t)
    if total > max_total_nm:
        # Scale down proportionally
        t = t * (max_total_nm / total)
        t = np.clip(t, 0.0, max_layer_nm)
    return t.tolist()


def design(problem):
    """
    Design a broadband antireflection coating using simulated annealing.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    band = problem["band_nm"]
    lam_min = band[0]
    lam_max = band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # Wavelength grid: 401 equally spaced points
    wavelengths = np.linspace(lam_min, lam_max, 401)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial design: start with a few layers
    # Try a good initial guess: alternating L and H layers with quarter-wave thicknesses
    # centered at the middle of the band
    centre = 0.5 * (lam_min + lam_max)

    # Start with a reasonable initial stack
    # Let's try a 5-layer stack: L H L H L with quarter-wave thicknesses
    # or use a known good starting point

    # Actually, let's start with a simple but not trivial design
    # and use simulated annealing to optimize

    # Random seed for reproducibility within the run
    rng = random.Random(42)

    # Initial design: 6 layers alternating H and L
    # Start with quarter-wave thicknesses at center wavelength
    init_n_layers = 6
    materials = []
    thicknesses = []
    for i in range(init_n_layers):
        if i % 2 == 0:
            materials.append("H")
            thicknesses.append(centre / (4.0 * nH))
        else:
            materials.append("L")
            thicknesses.append(centre / (4.0 * nL))

    # Evaluate initial design
    best_materials = materials[:]
    best_thicknesses = thicknesses[:]
    best_score = _mean_reflectance(best_materials, best_thicknesses, wavelengths, n0, ns)

    current_materials = materials[:]
    current_thicknesses = thicknesses[:]
    current_score = best_score

    # Simulated annealing parameters
    T = 5.0  # Initial temperature (in percent reflectance units)
    T_min = 0.001
    cooling_rate = 0.995

    step_count = 0

    while T > T_min:
        # Check deadline
        if time.time() > deadline:
            break

        # Propose a move
        move_type = rng.random()

        if move_type < 0.6:
            # Perturb a random thickness
            idx = rng.randrange(len(current_materials))
            delta = rng.gauss(0, 30.0)  # nm
            new_t = current_thicknesses[idx] + delta
            new_t = max(0.0, min(new_t, max_layer_nm))
            new_thicknesses = current_thicknesses[:]
            new_thicknesses[idx] = new_t

            # Check total thickness
            total = sum(new_thicknesses)
            if total > max_total_nm:
                # Scale down
                scale = max_total_nm / total
                new_thicknesses = [max(0.0, min(t * scale, max_layer_nm)) for t in new_thicknesses]

            new_materials = current_materials[:]
            new_score = _mean_reflectance(new_materials, new_thicknesses, wavelengths, n0, ns)

        elif move_type < 0.8:
            # Change a random layer material
            idx = rng.randrange(len(current_materials))
            new_materials = current_materials[:]
            if new_materials[idx] == "L":
                new_materials[idx] = "H"
            else:
                new_materials[idx] = "L"
            new_thicknesses = current_thicknesses[:]
            new_score = _mean_reflectance(new_materials, new_thicknesses, wavelengths, n0, ns)

        elif move_type < 0.9:
            # Add a layer (if under max)
            if len(current_materials) < max_layers:
                idx = rng.randrange(len(current_materials) + 1)
                new_materials = current_materials[:]
                new_thicknesses = current_thicknesses[:]

                # Insert a new layer
                mat = rng.choice(["L", "H"])
                if mat == "L":
                    t = centre / (4.0 * nL)
                else:
                    t = centre / (4.0 * nH)
                t = max(0.0, min(t, max_layer_nm))

                new_materials.insert(idx, mat)
                new_thicknesses.insert(idx, t)

                # Check total
                total = sum(new_thicknesses)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    new_thicknesses = [max(0.0, min(t * scale, max_layer_nm)) for t in new_thicknesses]

                new_score = _mean_reflectance(new_materials, new_thicknesses, wavelengths, n0, ns)
            else:
                new_score = current_score + 10.0  # Reject
                new_materials = current_materials[:]
                new_thicknesses = current_thicknesses[:]

        else:
            # Remove a layer (if more than 1)
            if len(current_materials) > 1:
                idx = rng.randrange(len(current_materials))
                new_materials = current_materials[:]
                new_thicknesses = current_thicknesses[:]
                new_materials.pop(idx)
                new_thicknesses.pop(idx)
                new_score = _mean_reflectance(new_materials, new_thicknesses, wavelengths, n0, ns)
            else:
                new_score = current_score + 10.0
                new_materials = current_materials[:]
                new_thicknesses = current_thicknesses[:]

        # Accept or reject
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < math.exp(-delta_score / T):
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = new_score

            if current_score < best_score:
                best_score = current_score
                best_materials = current_materials[:]
                best_thicknesses = current_thicknesses[:]

        step_count += 1

        # Cool down
        T *= cooling_rate

    # Final verification
    best_thicknesses = _clamp_thicknesses(best_thicknesses, max_layer_nm, max_total_nm)

    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
