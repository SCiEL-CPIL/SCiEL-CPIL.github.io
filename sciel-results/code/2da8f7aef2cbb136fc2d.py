"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE_SECONDS = 140.0
_START = time.perf_counter()

# Fixed structural design: 20 layers, symmetric, alternating L/H (L on air side).
# This approximates a graded index profile from air (1.0) toward the substrate.
_FIXED_LABELS = ["L", "H"] * 10          # L H L H ... L H (20 layers)
_NL = 20
_NMAT = {"L": 1.38, "H": 2.35}
_AMBIENT = 1.0
_SUBSTRATE = 1.52
_MAX_LAYER = 400.0
_MAX_TOTAL = 1000.0
_BAND = (400.0, 800.0)


def _elapsed():
    return time.perf_counter() - _START


def _deadline_hit():
    return _elapsed() >= _DEADLINE_SECONDS


def _reflectance_mean(materials, thicknesses):
    """Mean normal-incidence reflectance (fraction) over 401 wavelengths,
    computed exactly with the characteristic-matrix method."""
    w0, w1 = _BAND
    wl = np.linspace(w0, w1, 401)
    n0 = _AMBIENT
    ns = _SUBSTRATE
    n_layers = np.array([_NMAT[m] for m in materials], dtype=float)
    d = np.asarray(thicknesses, dtype=float)
    k = 2.0 * math.pi / wl
    # Build the 2x2 characteristic matrix product for all layers.
    # M_i = [[cos(d_i), (i/sin(d_i))], [i*sin(d_i), cos(d_i)]] * 1/n_i
    # We accumulate M_total = M_1 * M_2 * ... * M_N
    M = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=complex)
    for i in range(len(materials)):
        theta = k * d[i]
        c = np.cos(theta)
        s = np.sin(theta)
        n = n_layers[i]
        # M_i
        Mi = np.array([[c, 1j / s], [1j * s, c]], dtype=complex) / n
        M = M @ Mi
    # Effective admittance: Y = (M11 + M12) / (M21 + M22)  (normalized to n0)
    Y = (M[0, 0] + M[0, 1]) / (M[1, 0] + M[1, 1])
    # Reflectance
    r = (n0 * Y - ns) / (n0 * Y + ns)
    R = (r * np.conj(r)).real
    R = np.clip(R, 0.0, 1.0)
    return float(np.mean(R))


def _objective(materials, thicknesses):
    """Penalized objective (lower is better)."""
    d = np.asarray(thicknesses, dtype=float)
    if d.size != len(materials):
        return 1e6
    if np.any(d < -1e-9) or np.any(d > _MAX_LAYER + 1e-9):
        return 1e6
    if float(np.sum(d)) > _MAX_TOTAL + 1e-9:
        return 1e6
    return _reflectance_mean(materials, d)


def _clamp(d):
    d = np.clip(np.asarray(d, dtype=float), 0.0, _MAX_LAYER)
    if float(np.sum(d)) > _MAX_TOTAL:
        # scale down proportionally
        d = d * (_MAX_TOTAL / float(np.sum(d)))
    return d


def _nelder_mead(objective, x0, max_iter=400, tol=1e-9):
    """Simple Nelder-Mead on the raw objective (caller clamps)."""
    x = np.array(x0, dtype=float)
    n = x.size
    # Initial simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.2 if abs(xi[i]) < 1e-6 else 0.2 * abs(xi[i])
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(_clamp(s)) for s in simplex]
    for _ in range(max_iter):
        if _deadline_hit():
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(_clamp(xr))
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(_clamp(xe))
            if fe < fr:
                simplex[-1] = _clamp(xe)
                fvals[-1] = fe
            else:
                simplex[-1] = _clamp(xr)
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = _clamp(xr)
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(_clamp(xc))
            if fc < fvals[-1]:
                simplex[-1] = _clamp(xc)
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(_clamp(simplex[i]))
        if max(fvals) - min(fvals) < tol:
            break
    order = np.argsort(fvals)
    return simplex[order[0]].copy(), fvals[order[0]]


def _quarter_wave_init():
    """Analytical quarter-wave stack at band centre, made symmetric."""
    centre = 0.5 * (_BAND[0] + _BAND[1])
    lam = centre / _NMAT["L"]
    lam_h = centre / _NMAT["H"]
    # quarter-wave optical thickness: n*d = lambda/4
    d_l = lam / 4.0
    d_h = lam_h / 4.0
    d = np.empty(_NL)
    for i in range(_NL):
        d[i] = d_l if i % 2 == 0 else d_h
    return _clamp(d)


def design(problem):
    global _START
    _START = time.perf_counter()
    materials = list(_FIXED_LABELS)
    n_layers = len(materials)

    # Objective wrapper with deadline guard.
    def obj(d):
        if _deadline_hit():
            return 1e6
        return _objective(materials, d)

    best_x = _quarter_wave_init()
    best_f = obj(best_x)

    # Multi-start Nelder-Mead from several initializations.
    rng = np.random.default_rng(12345)
    # Start 1: quarter-wave symmetric.
    x, f = _nelder_mead(obj, best_x, max_iter=300)
    if f < best_f:
        best_f, best_x = f, x

    # Additional starts: perturb the quarter-wave solution.
    n_starts = 6
    for s in range(n_starts):
        if _deadline_hit():
            break
        x0 = best_x + rng.normal(0.0, 30.0, size=n_layers)
        x0 = _clamp(x0)
        x, f = _nelder_mead(obj, x0, max_iter=250)
        if f < best_f:
            best_f, best_x = f, x

    # Final polish from the best point.
    if not _deadline_hit():
        x, f = _nelder_mead(obj, best_x, max_iter=400)
        if f < best_f:
            best_f, best_x = f, x

    best_x = _clamp(best_x)
    # Ensure valid: if somehow invalid, fall back to quarter-wave.
    if obj(best_x) > 1e5:
        best_x = _quarter_wave_init()

    return materials, [float(v) for v in best_x]
# EVOLVE-BLOCK-END
