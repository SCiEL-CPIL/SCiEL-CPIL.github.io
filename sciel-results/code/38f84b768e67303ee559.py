"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _mean_reflectance(labels, thicknesses_nm, n_L, n_H, ambient, substrate, wavelengths):
    """Exact characteristic-matrix reflectance, mean over wavelengths, in percent."""
    n_layers = len(labels)
    if n_layers == 0:
        # Bare substrate
        r = (ambient - substrate) / (ambient + substrate)
        return float((r * r) * 100.0)
    
    # Precompute per-layer indices
    layer_ns = np.array([n_L if lab == 'L' else n_H for lab in labels], dtype=float)
    d = np.array(thicknesses_nm, dtype=float)
    
    # delta_j(λ) = 2π n_j d_j / λ
    # Shape: (n_layers, n_wl)
    delta = 2.0 * np.pi * layer_ns[:, None] * d[:, None] / wavelengths[None, :]
    
    c = np.cos(delta)  # (n_layers, n_wl)
    s = np.sin(delta)
    
    # Per-layer matrix components (vectorized over layers and wavelengths)
    # M_j = [[c, i/(n_j s)], [i n_j s, c]]
    M00 = c
    M01 = 1j / (layer_ns[:, None] * s)
    M10 = 1j * layer_ns[:, None] * s
    M11 = c
    
    # Chain multiply: M = M_1 * M_2 * ... * M_n
    # Start with identity (2x2, per wavelength)
    acc00 = np.ones_like(wavelengths, dtype=complex)
    acc01 = np.zeros_like(wavelengths, dtype=complex)
    acc10 = np.zeros_like(wavelengths, dtype=complex)
    acc11 = np.ones_like(wavelengths, dtype=complex)
    
    for j in range(n_layers):
        new00 = acc00 * M00[j] + acc01 * M10[j]
        new01 = acc00 * M01[j] + acc01 * M11[j]
        new10 = acc10 * M00[j] + acc11 * M10[j]
        new11 = acc10 * M01[j] + acc11 * M11[j]
        acc00, acc01, acc10, acc11 = new00, new01, new10, new11
    
    Y_sub = substrate
    Y = (acc00 + acc01 * Y_sub) / (acc10 + acc11 * Y_sub)
    r = (Y - ambient) / (Y + ambient)
    R = np.abs(r) ** 2
    return float(np.mean(R) * 100.0)


def _constrain(x, max_layer, max_total):
    """Project x into feasible box + sum constraint."""
    x = np.clip(x, 0.0, max_layer)
    s = np.sum(x)
    if s > max_total:
        x = x * (max_total / s)
        x = np.clip(x, 0.0, max_layer)
    return x


def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    using global + local search against the exact characteristic-matrix objective."""
    t_start = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    n_L = materials["L"]
    n_H = materials["H"]
    band = problem.get("band_nm", [400, 800])
    lam_min, lam_max = band[0], band[1]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)
    center = 0.5 * (lam_min + lam_max)
    
    def make_symmetric_labels(n_layers):
        if n_layers % 2 != 0:
            n_layers -= 1
        if n_layers <= 0:
            return []
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full
    
    def objective(x, labels):
        x = _constrain(x, max_layer, max_total)
        return _mean_reflectance(labels, x, n_L, n_H, ambient, substrate, wavelengths)
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Try several layer counts, prioritizing those likely to give best broadband performance
    for n_layers in [12, 14, 10, 16, 8, 20]:
        if time.time() >= DEADLINE - 5:
            break
        labels = make_symmetric_labels(n_layers)
        if len(labels) == 0 or len(labels) > max_layers:
            continue
        n_var = len(labels)
        
        # Initial guess: quarter-wave at band center
        x0 = np.zeros(n_var)
        for i, lab in enumerate(labels):
            n_i = n_L if lab == 'L' else n_H
            x0[i] = center / (4.0 * n_i)
        x0 = _constrain(x0, max_layer, max_total)
        
        val0 = objective(x0, labels)
        if val0 < best_score:
            best_score = val0
            best_labels = list(labels)
            best_thick = list(x0)
        
        # Global search with differential evolution (time-boxed)
        bounds = [(0.0, max_layer)] * n_var
        t_de_start = time.time()
        de_deadline = min(time.time() + 10.0, DEADLINE - 15.0)
        
        def de_obj(x):
            if time.time() > de_deadline:
                return 100.0  # penalty to stop
            return objective(x, labels)
        
        try:
            de_res = differential_evolution(
                de_obj, bounds,
                maxiter=80, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0),
                recombination=0.7, seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < best_score:
                x_de = _constrain(de_res.x, max_layer, max_total)
                val_de = objective(x_de, labels)
                if val_de < best_score:
                    best_score = val_de
                    best_labels = list(labels)
                    best_thick = list(x_de)
        except Exception:
            pass
        
        # Local polish with L-BFGS-B
        if time.time() < DEADLINE - 10:
            # Start from best of initial and DE
            start_x = _constrain(x0, max_layer, max_total)
            try:
                res = minimize(
                    lambda x: objective(x, labels),
                    start_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    x_lbfgs = _constrain(res.x, max_layer, max_total)
                    val_lbfgs = objective(x_lbfgs, labels)
                    if val_lbfgs < best_score:
                        best_score = val_lbfgs
                        best_labels = list(labels)
                        best_thick = list(x_lbfgs)
            except Exception:
                pass
        
        # Nelder-Mead polish
        if time.time() < DEADLINE - 5:
            try:
                res2 = minimize(
                    lambda x: objective(x, labels),
                    np.array(best_thick),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
                )
                if res2.fun < best_score:
                    x_nm = _constrain(res2.x, max_layer, max_total)
                    val_nm = objective(x_nm, labels)
                    if val_nm < best_score:
                        best_score = val_nm
                        best_labels = list(labels)
                        best_thick = list(x_nm)
            except Exception:
                pass
    
    # Final validation
    if best_labels is None:
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
    
    # Ensure constraints
    best_thick = list(best_thick)
    for i in range(len(best_thick)):
        best_thick[i] = float(np.clip(best_thick[i], 0.0, max_layer))
    s = sum(best_thick)
    if s > max_total:
        scale = max_total / s
        best_thick = [t * scale for t in best_thick]
        for i in range(len(best_thick)):
            best_thick[i] = float(np.clip(best_thick[i], 0.0, max_layer))
    
    if len(best_labels) > max_layers:
        best_labels = best_labels[:max_layers]
        best_thick = best_thick[:max_layers]
    
    return best_labels, best_thick

# EVOLVE-BLOCK-END
