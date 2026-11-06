"""Broadband antireflection coating design with numerical optimization."""
import time
import math
import numpy as np

# Deadline guard: kill before 150s wall-clock, leave safety margin.
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_left():
    return _DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _reflectance_stack(n_layers, thicknesses, n_mats, n0, ns, wls):
    """Compute normal-incidence reflectance (%) for a stack via characteristic matrix.

    n_layers: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    n_mats: dict {'L': nL, 'H': nH}
    n0: ambient index, ns: substrate index
    wls: array of wavelengths in nm
    Returns: array of reflectance in percent, same shape as wls.
    """
    m = len(n_layers)
    # Precompute per-layer refractive indices
    n_stack = np.array([n_mats[lb] for lb in n_layers], dtype=float)
    d = np.asarray(thicknesses, dtype=float)

    # Optical phase delta = 2*pi*n*d/lambda
    # wls shape (W,)
    wls = np.asarray(wls, dtype=float)
    delta = (2.0 * np.pi * np.outer(n_stack, d)) / wls  # shape (m, W)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    # Characteristic matrix per layer: [[cos, i sin/(n0_layer)], [i n0_layer sin, cos]]
    # We'll accumulate M = M1 @ M2 @ ... @ Mm (operating on (B,C) column vector)
    # Represent matrix as 2x2 array of shape (m, 2, 2, W)
    M = np.empty((m, 2, 2, wls.size), dtype=complex)
    for i in range(m):
        n_i = n_stack[i]
        c = cos_d[i]
        s = sin_d[i]
        M[i, 0, 0] = c
        M[i, 0, 1] = 1j * s / n_i
        M[i, 1, 0] = 1j * n_i * s
        M[i, 1, 1] = c

    # Multiply sequentially: total = M[0] @ M[1] @ ...
    # Start with identity
    total = np.empty((2, 2, wls.size), dtype=complex)
    total[:, :, :] = 0
    total[0, 0, :] = 1.0
    total[1, 1, :] = 1.0

    for i in range(m):
        # total_new = total @ M[i]
        # total (2,2,W), M[i] (2,2,W)
        t00 = total[0,0]*M[i,0,0] + total[0,1]*M[i,1,0]
        t01 = total[0,0]*M[i,0,1] + total[0,1]*M[i,1,1]
        t10 = total[1,0]*M[i,0,0] + total[1,1]*M[i,1,0]
        t11 = total[1,0]*M[i,0,1] + total[1,1]*M[i,1,1]
        total[0,0] = t00
        total[0,1] = t01
        total[1,0] = t10
        total[1,1] = t11

    # Admittance Y = (C/B) where (B,C) = total @ (1, ns)
    # B = total[0,0]*1 + total[0,1]*ns
    # C = total[1,0]*1 + total[1,1]*ns
    B = total[0,0] + total[0,1]*ns
    C = total[1,0] + total[1,1]*ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    return 100.0 * R


def _objective(d, n_layers, n_mats, n0, ns, wls):
    """Mean reflectance over the band (in percent)."""
    R = _reflectance_stack(n_layers, d, n_mats, n0, ns, wls)
    return float(np.mean(R))


def _make_seed(problem):
    """Analytical quarter-wave stack: H L H ... alternating, centered on band center."""
    band = problem["band_nm"]
    lam_c = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    # Quarter-wave thicknesses at center wavelength
    tL = lam_c / (4.0 * nL)
    tH = lam_c / (4.0 * nH)

    # Build alternating stack starting with H (to step up from air 1.0 toward glass 1.52)
    # Actually for AR on glass, we want to reduce reflectance; a single L layer works.
    # Multi-layer: alternate H/L starting and ending appropriately.
    # Use a symmetric-ish stack: H L H L ... with even number, or odd.
    # Let's pick a number of layers that fits budget.
    # Each H layer ~ lam_c/(4*nH) ~ 600/(4*2.35) ~ 63.8 nm
    # Each L layer ~ 600/(4*1.38) ~ 108.7 nm
    # Pair H+L ~ 172.5 nm. With 1000 nm budget, up to ~5 pairs = 10 layers.
    # Let's try 8 layers: H L H L H L H L
    layers = []
    thick = []
    # Start with H, alternate
    for i in range(max_layers):
        if i % 2 == 0:
            layers.append('H')
            thick.append(tH)
        else:
            layers.append('L')
            thick.append(tL)
        # Check total
        if sum(thick) > max_total:
            thick.pop()
            layers.pop()
            break
        if len(thick) >= max_layers:
            break

    # Clamp each thickness to max_layer
    thick = [min(max(0.0, t), max_layer) for t in thick]
    return layers, thick


def design(problem):
    """Design a broadband AR coating; returns (materials, thicknesses)."""
    n0 = problem.get("ambient_index", 1.0)
    ns = problem.get("substrate_index", 1.52)
    n_mats = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    # Wavelength grid: 401 equally spaced points in [400, 800]
    wls = np.linspace(band[0], band[1], 401)

    # Start with analytical seed
    layers, thick = _make_seed(problem)
    if not layers:
        return ["L"], [band[0] / (4.0 * n_mats["L"])]

    d = np.asarray(thick, dtype=float)
    best_d = d.copy()
    best_val = _objective(d, layers, n_mats, n0, ns, wls)

    # Numerical optimization via L-BFGS-B
    try:
        from scipy.optimize import minimize
    except Exception:
        return layers, [float(x) for x in best_d]

    bounds = [(0.0, max_layer) for _ in d]
    # Also total thickness constraint handled by penalizing or by bound; we'll add a penalty.
    def obj(dv):
        # Penalty if total exceeds max_total
        total = np.sum(dv)
        if total > max_total:
            penalty = 10.0 * (total - max_total) / max_total
            return _objective(dv, layers, n_mats, n0, ns, wls) + penalty
        return _objective(dv, layers, n_mats, n0, ns, wls)

    # Time budget for optimization
    t0 = _elapsed()
    time_budget = min(_deadline_left(), 120.0)
    if time_budget < 5.0:
        return layers, [float(x) for x in best_d]

    # L-BFGS-B
    try:
        res = minimize(
            obj,
            d,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 500, 'disp': False,
                     'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_val:
            cand = np.clip(res.x, 0.0, max_layer)
            if np.sum(cand) <= max_total + 1e-6:
                val = _objective(cand, layers, n_mats, n0, ns, wls)
                if val < best_val:
                    best_val = val
                    best_d = cand.copy()
    except Exception:
        pass

    # Check deadline
    if _deadline_left() < 5.0:
        return layers, [float(x) for x in best_d]

    # Local refinement: try small perturbations / multi-start with Nelder-Mead on subsets
    # Use Nelder-Mead for a second pass to escape local minima
    try:
        from scipy.optimize import minimize as min2
        t1 = _elapsed()
        remaining = _deadline_left() - 5.0
        if remaining > 10.0:
            # Nelder-Mead on the current best
            res2 = min2(
                obj,
                best_d,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-10,
                         'maxfev': 400, 'adaptive': True}
            )
            if res2.fun < best_val:
                cand = np.clip(res2.x, 0.0, max_layer)
                if np.sum(cand) <= max_total + 1e-6:
                    val = _objective(cand, layers, n_mats, n0, ns, wls)
                    if val < best_val:
                        best_val = val
                        best_d = cand.copy()
    except Exception:
        pass

    if _deadline_left() < 5.0:
        return layers, [float(x) for x in best_d]

    # Try different layer counts: search over number of layers from 2 to max_layers
    # For each, build a quarter-wave seed and optimize.
    nL = n_mats["L"]
    nH = n_mats["nH"] if "nH" in n_mats else n_mats["H"]
    lam_c = 0.5 * (band[0] + band[1])
    tL_qw = lam_c / (4.0 * nL)
    tH_qw = lam_c / (4.0 * nH)

    for n_layers_try in range(2, max_layers + 1):
        if _deadline_left() < 10.0:
            break
        # Build seed for this layer count
        layers_t = []
        thick_t = []
        for i in range(n_layers_try):
            if i % 2 == 0:
                layers_t.append('H')
                thick_t.append(tH_qw)
            else:
                layers_t.append('L')
                thick_t.append(tL_qw)
        # Check total
        if sum(thick_t) > max_total:
            continue
        d_t = np.asarray(thick_t, dtype=float)
        d_t = np.clip(d_t, 0.0, max_layer)

        def obj_t(dv):
            total = np.sum(dv)
            if total > max_total:
                penalty = 10.0 * (total - max_total) / max_total
                return _objective(dv, layers_t, n_mats, n0, ns, wls) + penalty
            return _objective(dv, layers_t, n_mats, n0, ns, wls)

        try:
            res_t = minimize(
                obj_t,
                d_t,
                method='L-BFGS-B',
                bounds=[(0.0, max_layer)] * n_layers_try,
                options={'maxiter': 100, 'maxfun': 300, 'disp': False,
                         'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_t.fun < best_val:
                cand = np.clip(res_t.x, 0.0, max_layer)
                if np.sum(cand) <= max_total + 1e-6:
                    val = _objective(cand, layers_t, n_mats, n0, ns, wls)
                    if val < best_val:
                        best_val = val
                        best_d = cand.copy()
                        layers = layers_t  # update layers
        except Exception:
            pass

    # Final clamp and validity
    best_d = np.clip(best_d, 0.0, max_layer)
    if np.sum(best_d) > max_total:
        # Scale down proportionally
        scale = max_total / np.sum(best_d)
        best_d = best_d * scale
        best_d = np.clip(best_d, 0.0, max_layer)

    return layers, [float(x) for x in best_d]
