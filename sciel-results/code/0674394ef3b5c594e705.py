"""Antireflection coating design: search L/H layer patterns + anneal thicknesses."""
import time
import math
import numpy as np

_DEADLINE_FRACTION = 0.92  # safety margin: stop searching before the wall-clock limit


def _reflectance_spectrum(wavelengths_nm, n_layers, t_layers, n_ambient, n_substrate, n_stack):
    """Mean reflectance (percent) over wavelengths via characteristic-matrix method."""
    k = 2.0 * math.pi / wavelengths_nm  # rad / nm
    # per-layer phase: delta_j = k * n_j * t_j
    delta = k[None, :] * n_stack[:, None] * t_layers[:, None]  # (N, W)
    c = np.cos(delta)
    s = np.sin(delta)
    # transfer matrix M_j = [[cos, i sin/n_j], [i n_j sin, cos]]
    m11 = c
    m12 = 1j * s / n_stack[:, None]
    m21 = 1j * n_stack[:, None] * s
    m22 = c
    # multiply N matrices in order (air -> substrate)
    M11 = np.ones_like(delta)
    M12 = np.zeros_like(delta)
    M21 = np.zeros_like(delta)
    M22 = np.ones_like(delta)
    for j in range(n_layers):
        nM11 = M11 * m11[j] + M12 * m21[j]
        nM12 = M11 * m12[j] + M12 * m22[j]
        nM21 = M21 * m11[j] + M22 * m21[j]
        nM22 = M21 * m12[j] + M22 * m22[j]
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22
    # substrate admittance
    y_s = n_substrate
    # r = (n_ambient*M11 + n_ambient*M12*y_s - M21 - M22*y_s) /
    #     (n_ambient*M11 + n_ambient*M12*y_s + M21 + M22*y_s)
    num = n_ambient * M11 + n_ambient * M12 * y_s - M21 - M22 * y_s
    den = n_ambient * M11 + n_ambient * M12 * y_s + M21 + M22 * y_s
    r = num / den
    R = np.abs(r) ** 2
    return float(np.mean(R) * 100.0)


def _anneal(labels, t0, n_ambient, n_substrate, n_stack, wavelengths_nm,
            max_layer_nm, max_total_nm, deadline, start_temp, min_temp, cooling):
    """Simulated annealing on thicknesses. Returns (best_t, best_val)."""
    t = np.array(t0, dtype=float).copy()
    n_layers = len(t)
    best_t = t.copy()
    best_val = _reflectance_spectrum(wavelengths_nm, n_layers, t, n_ambient, n_substrate, n_stack)

    temp = start_temp
    rng = np.random.default_rng(12345)
    while temp > min_temp:
        if time.time() > deadline:
            break
        # propose a new thickness for one or two random layers
        idx = rng.integers(0, n_layers)
        step = rng.uniform(0.0, 1.0) * 30.0
        new_t = t.copy()
        new_t[idx] = t[idx] + (2.0 * rng.uniform(0.0, 1.0) - 1.0) * step
        if new_t[idx] < 0.0:
            new_t[idx] = 0.0
        if new_t[idx] > max_layer_nm:
            new_t[idx] = max_layer_nm
        if new_t.sum() > max_total_nm:
            # scale down proportionally to satisfy total constraint
            new_t = new_t * (max_total_nm / new_t.sum())
        new_val = _reflectance_spectrum(wavelengths_nm, n_layers, new_t, n_ambient, n_substrate, n_stack)
        d = new_val - best_val
        if d < 0 or rng.uniform(0.0, 1.0) < math.exp(-d / max(temp, 1e-9)):
            t = new_t
            if new_val < best_val:
                best_val = new_val
                best_t = t.copy()
        temp *= cooling
    return best_t, best_val


def design(problem):
    start = time.time()
    deadline = start + (150.0 - 5.0) * _DEADLINE_FRACTION  # leave margin

    n_ambient = float(problem["ambient_index"])
    n_substrate = float(problem["substrate_index"])
    mat = problem["materials"]
    n_L = float(mat["L"])
    n_H = float(mat["H"])
    band = problem["band_nm"]
    wl = np.linspace(band[0], band[1], 401)
    max_layers = int(problem["max_layers"])
    max_total_nm = float(problem["max_total_nm"])
    max_layer_nm = float(problem["max_layer_nm"])

    centre = 0.5 * (band[0] + band[1])
    qL = centre / (4.0 * n_L)
    qH = centre / (4.0 * n_H)

    # Build candidate label patterns (air side -> substrate), symmetric/anti-symmetric.
    def make_patterns():
        pats = []
        # single layer
        pats.append(["L"])
        pats.append(["H"])
        # pairs
        pats.append(["L", "H"])
        pats.append(["H", "L"])
        # symmetric stacks of L/H quarter-wave pairs, up to budget
        for n in range(1, max_layers + 1):
            if n > 20:
                break
            # alternating starting with L
            alt = []
            for i in range(n):
                alt.append("L" if i % 2 == 0 else "H")
            pats.append(alt)
            # alternating starting with H
            alt2 = []
            for i in range(n):
                alt2.append("H" if i % 2 == 0 else "L")
            pats.append(alt2)
            # symmetric: L H ... H L (mirror)
            if n >= 2:
                half = (n + 1) // 2
                left = []
                for i in range(half):
                    left.append("L" if i % 2 == 0 else "H")
                sym = left + left[-2::-1] if n % 2 == 0 else left + left[-1::-1]
                # ensure length n
                sym = sym[:n]
                pats.append(sym)
        # dedupe
        seen = set()
        uniq = []
        for p in pats:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    patterns = make_patterns()

    best_labels = ["L"]
    best_t = np.array([qL])
    best_val = _reflectance_spectrum(wl, 1, best_t, n_ambient, n_substrate, [n_L])

    # quick screening: evaluate quarter-wave init for each pattern, keep promising ones
    screened = []
    for p in patterns:
        n = len(p)
        if n > max_layers:
            continue
        # quarter-wave init
        t_init = np.array([(qL if lab == "L" else qH) for lab in p], dtype=float)
        if t_init.sum() > max_total_nm:
            t_init = t_init * (max_total_nm / t_init.sum())
        if t_init.max() > max_layer_nm:
            continue
        n_stack = np.array([n_L if lab == "L" else n_H for lab in p], dtype=float)
        val = _reflectance_spectrum(wl, n, t_init, n_ambient, n_substrate, n_stack)
        screened.append((val, p, t_init, n_stack))
    screened.sort(key=lambda x: x[0])
    # keep top patterns to anneal
    top = screened[:12]

    for val, p, t_init, n_stack in top:
        if time.time() > deadline:
            break
        n = len(p)
        # anneal
        bt, bv = _anneal(p, t_init, n_ambient, n_substrate, n_stack, wl,
                         max_layer_nm, max_total_nm, deadline,
                         start_temp=2.0, min_temp=0.02, cooling=0.97)
        if bv < best_val:
            best_val = bv
            best_labels = list(p)
            best_t = bt.copy()

    # ensure final validity
    if best_t.sum() > max_total_nm:
        best_t = best_t * (max_total_nm / best_t.sum())
    best_t = np.clip(best_t, 0.0, max_layer_nm)
    if best_t.sum() > max_total_nm:
        best_t = best_t * (max_total_nm / best_t.sum())

    return best_labels, [float(x) for x in best_t]
