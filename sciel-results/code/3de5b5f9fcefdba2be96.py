"""Broadband antireflection coating design via fast characteristic-matrix model
and multi-start local search with an explicit wall-clock deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

# Fixed problem parameters (from the task contract)
N_AMB = 1.0
N_SUB = 1.52
N_L = 1.38
N_H = 2.35
WAVE_MIN, WAVE_MAX = 400.0, 800.0
MAX_LAYERS = 20
MAX_TOTAL_NM = 1000.0
MAX_LAYER_NM = 400.0
WAVES = np.linspace(WAVE_MIN, WAVE_MAX, 401)


def _compute_reflectance(labels, thicknesses):
    """Mean normal-incidence reflectance (fraction) over the band, using the
    2x2 characteristic-matrix method, vectorized over all wavelengths."""
    n = len(labels)
    if n == 0:
        # bare glass reflectance
        r = (N_AMB - N_SUB) / (N_AMB + N_SUB)
        return float(r * r)

    indices = np.array([N_L if lab == "L" else N_H for lab in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)

    k = 2.0 * np.pi / WAVES                      # (M,)
    # phase per layer: (M, n)
    phase = k[:, None] * indices[None, :] * d[None, :]

    c = np.cos(phase)
    s = np.sin(phase)

    # Layer matrix M_j = [[c_j, i s_j / n_j], [i n_j s_j, c_j]]
    inv_n = 1.0 / indices
    A = c                                  # (M, n)
    B = 1j * s * inv_n[None, :]            # (M, n)
    C = 1j * indices[None, :] * s          # (M, n)
    D = c                                  # (M, n)

    # Substrate matrix at normal incidence: [[1, 1/n_s], [n_s, 1]]
    A_sub = np.ones((len(WAVES), 1))
    B_sub = np.full((len(WAVES), 1), 1.0 / N_SUB)
    C_sub = np.full((len(WAVES), 1), N_SUB)
    D_sub = np.ones((len(WAVES), 1))

    # Multiply matrices left to right: M_total = M_1 @ M_2 @ ... @ M_n @ M_sub
    # Each step: newA = A*Aj + B*Cj, newB = A*Bj + B*Dj,
    #            newC = C*Aj + D*Cj, newD = C*Bj + D*Dj
    A_tot = A[:, 0:1]
    B_tot = B[:, 0:1]
    C_tot = C[:, 0:1]
    D_tot = D[:, 0:1]
    for j in range(1, n):
        Aj = A[:, j:j+1]; Bj = B[:, j:j+1]
        Cj = C[:, j:j+1]; Dj = D[:, j:j+1]
        A_tot = A_tot * Aj + B_tot * Cj
        B_tot = A_tot * Bj + B_tot * Dj
        C_tot = C_tot * Aj + D_tot * Cj
        D_tot = C_tot * Bj + D_tot * Dj

    # Now multiply by substrate
    A_tot = A_tot * A_sub + B_tot * C_sub
    B_tot = A_tot * B_sub + B_tot * D_sub
    C_tot = C_tot * A_sub + D_tot * C_sub
    D_tot = C_tot * B_sub + D_tot * D_sub

    # Effective admittance Y = C/A, reflectance at air interface
    Y = C_tot / A_tot
    r = (N_AMB - Y) / (N_AMB + Y)
    refl = r * np.conj(r)
    refl = np.real(refl)
    refl = np.clip(refl, 0.0, None)
    return float(np.mean(refl))


def _feasible(labels, thicknesses):
    if len(labels) > MAX_LAYERS:
        return False
    if len(labels) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0.0 or t > MAX_LAYER_NM + 1e-9:
            return False
    if sum(thicknesses) > MAX_TOTAL_NM + 1e-9:
        return False
    return True


def _make_objective(labels):
    n = len(labels)

    def objective(x):
        # clip to bounds
        tc = np.clip(x, 0.0, MAX_LAYER_NM)
        tot = tc.sum()
        if tot > MAX_TOTAL_NM:
            tc = tc * (MAX_TOTAL_NM / tot)
        r = _compute_reflectance(labels, tc)
        # small penalty for boundary violations (shouldn't happen after clip)
        return r
    return objective


def _quarterwave_thick(label, center=600.0):
    n = N_L if label == "L" else N_H
    return center / (4.0 * n)


def _candidate_patterns():
    """Yield a list of label patterns (diverse) to seed the search."""
    patterns = []
    # symmetric L/H quarter-wave stacks of increasing order
    for order in range(1, 9):
        # pattern: L H L H ... L  (starts and ends with L), length = 2*order-1
        seq = []
        for i in range(2 * order - 1):
            seq.append("L" if i % 2 == 0 else "H")
        patterns.append(seq)
    # alternate starting with H
    for order in range(1, 7):
        seq = []
        for i in range(2 * order - 1):
            seq.append("H" if i % 2 == 0 else "L")
        patterns.append(seq)
    # single layer
    patterns.append(["L"])
    patterns.append(["H"])
    # longer mixed patterns
    patterns.append(["L", "H", "L", "H", "L", "H", "L", "H", "L", "H",
                     "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"])
    return patterns


def design(problem):
    deadline = time.time() + 120.0   # safety margin before the 120s eval / 150s kill

    best_labels = None
    best_thick = None
    best_r = float("inf")

    def consider(labels, thicknesses):
        nonlocal best_labels, best_thick, best_r
        if not _feasible(labels, thicknesses):
            return
        r = _compute_reflectance(labels, thicknesses)
        if r < best_r:
            best_r = r
            best_labels = list(labels)
            best_thick = [float(t) for t in thicknesses]

    # Seed with a couple of quick quarter-wave designs
    for lab in (["L"], ["H"]):
        consider(lab, [_quarterwave_thick(lab[0])])

    for pattern in _candidate_patterns():
        if time.time() > deadline - 5.0:
            break
        n = len(pattern)
        if n > MAX_LAYERS:
            continue
        # initial guess: quarter-wave at 600nm, scaled so total <= MAX_TOTAL
        x0 = np.array([_quarterwave_thick(lab) for lab in pattern], dtype=float)
        if x0.sum() > MAX_TOTAL_NM:
            x0 = x0 * (MAX_TOTAL_NM / x0.sum())
        obj = _make_objective(pattern)
        try:
            res = minimize(
                obj, x0, method="Nelder-Mead",
                bounds=[(0.0, MAX_LAYER_NM)] * n,
                options={"maxiter": 400, "xatol": 0.5, "fatol": 1e-6,
                         "adaptive": True},
            )
        except Exception:
            continue
        consider(pattern, res.x)
        # also try a second, perturbed start for the best pattern
        if time.time() > deadline - 5.0:
            break
        x1 = np.clip(x0 * np.random.uniform(0.7, 1.3, size=n), 0.0, MAX_LAYER_NM)
        if x1.sum() > MAX_TOTAL_NM:
            x1 = x1 * (MAX_TOTAL_NM / x1.sum())
        try:
            res2 = minimize(
                obj, x1, method="Nelder-Mead",
                bounds=[(0.0, MAX_LAYER_NM)] * n,
                options={"maxiter": 300, "xatol": 0.5, "fatol": 1e-6,
                         "adaptive": True},
            )
        except Exception:
            continue
        consider(pattern, res2.x)

    # Fallback if nothing feasible found
    if best_labels is None:
        best_labels = ["L"]
        best_thick = [_quarterwave_thick("L")]

    return best_labels, best_thick
