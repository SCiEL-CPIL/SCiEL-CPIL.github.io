"""Broadband antireflection coating design: fast characteristic-matrix evaluator + coordinate descent + Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """Vectorized reflectance (0..1) over all wavelengths."""
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    nd_wl = np.empty((len(materials), len(wls)), dtype=float)
    for i, m in enumerate(materials):
        nd_wl[i, :] = n_mat[m] * thicknesses[i] / wls

    beta = 2.0 * math.pi * nd_wl
    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(len(wls), dtype=complex)
    M12 = np.zeros(len(wls), dtype=complex)
    M21 = np.zeros(len(wls), dtype=complex)
    M22 = np.ones(len(wls), dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        A = c
        B = -1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """Mean reflectance in percent over the band. Invalid -> large penalty."""
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
    for t in thicknesses:
        if t < -1e-9 or t > problem["max_layer_nm"] + 1e-9:
            return 1e6
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return 1e6
    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0
    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _nelder_mead(objective, x0, bounds, deadline, max_iter=1000):
    """Nelder-Mead with clipping to bounds."""
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-9 else 10.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))

    f_vals = [objective(x) for x in simplex]

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < 1e-8:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + 2.0 * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
        x_c = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best = order[0]
    return np.clip(simplex[best], lo, hi), f_vals[best]

def _coordinate_descent(objective, x0, bounds, deadline, max_iter=200, step_init=10.0):
    """Coordinate descent with adaptive step size."""
    x = np.array(x0, dtype=float)
    lo, hi = bounds
    n = len(x)
    f_current = objective(x)
    steps = np.full(n, step_init)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try positive step
            x_try = x.copy()
            x_try[i] = min(x[i] + steps[i], hi[i])
            f_try = objective(x_try)
            if f_try < f_current:
                x = x_try
                f_current = f_try
                steps[i] *= 1.5
                improved = True
                continue
            else:
                steps[i] *= 0.5
            
            # Try negative step
            x_try = x.copy()
            x_try[i] = max(x[i] - steps[i], lo[i])
            f_try = objective(x_try)
            if f_try < f_current:
                x = x_try
                f_current = f_try
                steps[i] *= 1.5
                improved = True
                continue
            else:
                steps[i] *= 0.5
        
        if not improved:
            break
    
    return x, f_current

def design(problem):
    """Design a broadband antireflection coating (air -> ... -> glass)."""
    start = time.time()
    deadline = start + 145.0  # safety margin before the 150 s kill

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    wls = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_mat = problem["materials"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])

    def objective(th, mats):
        return _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)

    def scale_to_fit(mats, ths):
        s = sum(ths)
        if s > max_total:
            f = max_total / s
            ths = [t * f for t in ths]
        return mats, ths

    # Generate diverse initial candidates
    candidates = []
    
    # Layer counts to try
    layer_counts = [2, 4, 6, 8, 10, 12]
    
    for n_layers in layer_counts:
        if n_layers > max_layers:
            continue
            
        # Alternate L and H, starting with L
        mats = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
            else:
                mats.append('H')
        
        # Initial thicknesses: Quarter wave at center wavelength
        ths = []
        for m in mats:
            n = n_mat[m]
            q = centre / (4.0 * n)
            ths.append(q)
        
        # Ensure constraints
        mats, ths = scale_to_fit(mats, ths)
        ths = [min(max(t, 0.0), max_layer) for t in ths]
        if sum(ths) > max_total + 1e-6:
            f = max_total / sum(ths)
            ths = [t * f for t in ths]
        
        candidates.append((mats, ths))

    # Add H-start candidate
    mats_h_start = []
    ths_h_start = []
    for i in range(6):
        if i % 2 == 0:
            mats_h_start.append('H')
            ths_h_start.append(centre / (4.0 * n_H))
        else:
            mats_h_start.append('L')
            ths_h_start.append(centre / (4.0 * n_L))
    if len(mats_h_start) <= max_layers:
        candidates.append((mats_h_start, ths_h_start))

    # Add random candidates
    for _ in range(20):
        n_layers = random.randint(2, min(12, max_layers))
        mats = []
        ths = []
        for i in range(n_layers):
            m = random.choice(['L', 'H'])
            mats.append(m)
            ths.append(random.uniform(20, 200))
        
        mats, ths = scale_to_fit(mats, ths)
        ths = [min(max(t, 0.0), max_layer) for t in ths]
        if sum(ths) > max_total + 1e-6:
            f = max_total / sum(ths)
            ths = [t * f for t in ths]
        
        candidates.append((mats, ths))

    # Evaluate all candidates and keep the top few
    scored_candidates = []
    for mats, ths in candidates:
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        scored_candidates.append((val, mats, ths))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_val = float("inf")
    best_mats = None
    best_ths = None
    
    # Refine the top candidates
    num_to_refine = min(5, len(scored_candidates))
    
    for i in range(num_to_refine):
        if time.time() > deadline:
            break
            
        init_val, init_mats, init_ths = scored_candidates[i]
        
        if init_val < best_val:
            best_val = init_val
            best_mats = init_mats
            best_ths = init_ths
            
        materials = list(init_mats)
        lo = np.zeros(len(materials))
        hi = np.full(len(materials), max_layer)
        x0 = np.array(init_ths, dtype=float)
        
        def obj(x):
            return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)
        
        # First try coordinate descent (faster convergence for smooth problems)
        x_cd, f_cd = _coordinate_descent(obj, x0, (lo, hi), deadline, max_iter=100, step_init=10.0)
        
        # Then refine with Nelder-Mead
        x_best, f_best = _nelder_mead(obj, x_cd, (lo, hi), deadline, max_iter=300)
        
        x_best = np.clip(x_best, 0.0, max_layer)
        if sum(x_best) > max_total + 1e-6:
            f = max_total / sum(x_best)
            x_best = x_best * f
        x_best = np.clip(x_best, 0.0, max_layer)
        
        val = _mean_reflectance_percent(materials, list(x_best), problem, wls, n_mat)
        
        if val < best_val:
            best_val = val
            best_mats = materials
            best_ths = list(x_best)
            
    # Fallback
    if best_mats is None:
        best_mats = ["L"]
        best_ths = [centre / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)

    # Final safety
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]

    return list(best_mats), list(best_ths)
# EVOLVE-BLOCK-END
