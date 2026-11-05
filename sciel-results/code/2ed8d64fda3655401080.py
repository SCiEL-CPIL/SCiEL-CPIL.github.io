"""Broadband antireflection coating design via multi-start local optimization."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix(n, d, wl):
    """Characteristic matrix for a single layer at normal incidence.
    M = [[cos(beta), (i/n)*sin(beta)], [i*n*sin(beta), cos(beta)]]
    """
    beta = 2.0 * np.pi * n * d / wl
    cb = np.cos(beta)
    sb = np.sin(beta)
    return np.array([[cb, 1j * sb / n],
                     [1j * n * sb, cb]])


def _compute_reflectance(materials, thicknesses, n0, ns, wl):
    """Mean reflectance (0..1) at wavelength wl for given stack.
    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    n0: ambient index, ns: substrate index
    """
    n_dict = {'L': 1.38, 'H': 2.35}
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    for lab, d in zip(materials, thicknesses):
        n = n_dict[lab]
        if d < 1e-9:
            continue
        M = M @ _char_matrix(n, d, wl)
    # Effective admittance: Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]
    Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    # Reflectance amplitude
    r = (n0 - Y) / (n0 + Y)
    return abs(r) ** 2


def _mean_reflectance_percent(materials, thicknesses, problem, wl_array):
    """Mean reflectance in percent over wavelength array."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    vals = np.array([_compute_reflectance(materials, thicknesses, n0, ns, w)
                     for w in wl_array])
    return 100.0 * np.mean(vals)


def _is_valid(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for d in thicknesses:
        if d < -1e-6 or d > max_layer + 1e-6:
            return False
    if sum(thicknesses) > max_total + 1e-6:
        return False
    return True


def _clamp_stack(materials, thicknesses, problem):
    """Clamp thicknesses into feasible range, drop empty layers."""
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    mats = []
    ths = []
    for lab, d in zip(materials, thicknesses):
        d = max(0.0, min(float(d), max_layer))
        if d > 1e-6:
            mats.append(lab)
            ths.append(d)
    if len(mats) > max_layers:
        mats = mats[:max_layers]
        ths = ths[:max_layers]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
    return mats, ths


def _score(materials, thicknesses, problem, wl_array):
    mats, ths = _clamp_stack(materials, thicknesses, problem)
    if len(mats) == 0:
        # bare substrate
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    return _mean_reflectance_percent(mats, ths, problem, wl_array)


def _nelder_mead(f, x0, problem, wl_array, max_iter=200, tol=1e-8, deadline=None):
    """Nelder-Mead on stacked variable: [d0, lab0(0/1), d1, lab1, ...]
    We encode as: for each layer, thickness and a continuous 'index' variable
    that we round to L or H. Use a penalty for index deviation.
    """
    n_layers = len(x0) // 2
    best = f(x0, problem, wl_array)
    best_x = list(x0)

    # Build simplex
    simplex = [list(x0)]
    for i in range(len(x0)):
        p = list(x0)
        step = 50.0 if i % 2 == 0 else 0.5  # thickness or index
        p[i] += step
        simplex.append(p)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Sort
        vals = [f(s, problem, wl_array) for s in simplex]
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]

        if abs(vals[-1] - vals[0]) < tol:
            break

        # Centroid of all but worst
        centroid = [np.mean([simplex[j][i] for j in range(len(simplex) - 1)])
                    for i in range(len(x0))]

        # Reflection
        alpha = 1.0
        xr = [centroid[i] + alpha * (centroid[i] - simplex[-1][i]) for i in range(len(x0))]
        fr = f(xr, problem, wl_array)
        if fr < vals[0]:
            # Expansion
            gamma = 2.0
            xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(len(x0))]
            fe = f(xe, problem, wl_array)
            if fe < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
        elif fr < vals[-2]:
            simplex[-1] = xr
        else:
            # Contraction
            rho = 0.5
            xc = [centroid[i] + rho * (simplex[-1][i] - centroid[i]) for i in range(len(x0))]
            fc = f(xc, problem, wl_array)
            if fc < vals[-1]:
                simplex[-1] = xc
            else:
                # Shrink
                for j in range(1, len(simplex)):
                    simplex[j] = [simplex[0][i] + 0.5 * (simplex[j][i] - simplex[0][i])
                                  for i in range(len(x0))]
                simplex[-1] = [simplex[0][i] + 0.5 * (simplex[-1][i] - simplex[0][i])
                               for i in range(len(x0))]

    # Find best in simplex
    for s in simplex:
        v = f(s, problem, wl_array)
        if v < best:
            best = v
            best_x = list(s)
    return best, best_x


def _decode(x, problem):
    """Decode continuous variable to (materials, thicknesses)."""
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    mats = []
    ths = []
    n_layers = len(x) // 2
    for i in range(n_layers):
        d = x[2 * i]
        idx_var = x[2 * i + 1]
        d = max(0.0, min(float(d), max_layer))
        if d < 1e-6:
            continue
        # index variable: 0 -> L, 1 -> H (use midpoint 0.5)
        lab = 'L' if idx_var < 0.5 else 'H'
        mats.append(lab)
        ths.append(d)
    if len(mats) > max_layers:
        mats = mats[:max_layers]
        ths = ths[:max_layers]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
    return mats, ths


def _f_encoded(x, problem, wl_array):
    """Objective for encoded variable."""
    mats, ths = _decode(x, problem)
    if len(mats) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    return _mean_reflectance_percent(mats, ths, problem, wl_array)


def design(problem):
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    wl_array = np.linspace(band[0], band[1], 401)

    # Bare substrate baseline
    r_bare = (n0 - ns) / (n0 + ns)
    best_score = 100.0 * r_bare * r_bare
    best_result = (['L'], [600.0 / (4.0 * 1.38)])  # quarter-wave L at 600nm

    # Try a few fixed good designs first
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    centre = 0.5 * (band[0] + band[1])

    candidate_stacks = [
        # Single quarter-wave L
        (['L'], [centre / (4.0 * nL)]),
        # Single quarter-wave H
        (['H'], [centre / (4.0 * nH)]),
        # Double quarter-wave L-H
        (['L', 'H'], [centre / (4.0 * nL), centre / (4.0 * nH)]),
        (['H', 'L'], [centre / (4.0 * nH), centre / (4.0 * nL)]),
        # 5-layer: L H L H L quarter-wave
        (['L', 'H', 'L', 'H', 'L'], [centre / (4.0 * nL), centre / (4.0 * nH),
                                     centre / (4.0 * nL), centre / (4.0 * nH),
                                     centre / (4.0 * nL)]),
        # 3-layer: L H L
        (['L', 'H', 'L'], [centre / (4.0 * nL), centre / (4.0 * nH),
                           centre / (4.0 * nL)]),
    ]

    for mats, ths in candidate_stacks:
        if not _is_valid(mats, ths, problem):
            continue
        s = _score(mats, ths, problem, wl_array)
        if s < best_score:
            best_score = s
            best_result = (mats, ths)

    # Multi-start local optimization on encoded variables
    # Start with a moderate number of layers
    n_layers_start = 8
    # Build initial x: alternating L/H with quarter-wave thicknesses
    x0 = []
    for i in range(n_layers_start):
        d = centre / (4.0 * (nL if i % 2 == 0 else nH))
        idx = 0.0 if i % 2 == 0 else 1.0
        x0.extend([d, idx])

    # Run Nelder-Mead
    best_enc_score = best_score
    best_enc_x = list(x0)

    # Multiple random starts
    np.random.seed(42)
    n_starts = 5
    for start in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        if start == 0:
            x = list(x0)
        else:
            x = []
            for i in range(n_layers_start):
                d = np.random.uniform(20.0, 200.0)
                idx = np.random.choice([0.0, 1.0])
                x.extend([d, idx])

        s, x_opt = _nelder_mead(_f_encoded, x, problem, wl_array,
                                 max_iter=150, deadline=deadline - 5.0)
        if s < best_enc_score:
            best_enc_score = s
            best_enc_x = x_opt

    # Decode best
    mats, ths = _decode(best_enc_x, problem)
    if len(mats) > 0:
        s = _score(mats, ths, problem, wl_array)
        if s < best_score:
            best_score = s
            best_result = (mats, ths)

    # Final polish: try small perturbations of thicknesses on best_result
    if time.time() < deadline - 10.0:
        mats0, ths0 = best_result
        if len(mats0) > 0:
            # Coordinate descent on thicknesses
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(len(ths0)):
                    base = list(ths0)
                    d0 = base[i]
                    for frac in [0.8, 0.9, 1.1, 1.2, 0.7, 1.3]:
                        base[i] = d0 * frac
                        s = _score(mats0, base, problem, wl_array)
                        if s < best_score - 1e-6:
                            best_score = s
                            best_result = (mats0, list(base))
                            improved = True
                    ths0 = best_result[1]

    return list(best_result[0]), list(best_result[1])

# EVOLVE-BLOCK-END
