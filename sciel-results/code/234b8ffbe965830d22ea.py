"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _reflectance_spectrum(materials, thicknesses, wavelengths, n_sub):
    """
    Compute reflectance spectrum using the characteristic matrix method.
    materials: list of refractive indices
    thicknesses: list of physical thicknesses in nm
    wavelengths: array of wavelengths in nm
    n_sub: substrate index
    Returns: array of reflectances (0 to 1)
    """
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        r = (1.0 - n_sub) / (1.0 + n_sub)
        return np.full_like(wavelengths, r * r, dtype=float)

    # Stack matrices
    # M = product of layer matrices * substrate matrix
    # For normal incidence:
    # Layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # Substrate: [[1, 0], [n_sub, 0]]  (actually the substrate is handled as the final medium)
    
    # Actually, the standard characteristic matrix method:
    # For each layer j: M_j = [[cos(delta_j), (i/n_j)*sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Then: (B, C)^T = M * (1, n_sub)^T
    # Y = C/B  (optical admittance)
    # r = (1 - Y)/(1 + Y)  (for air side, n_air = 1)
    # R = |r|^2

    # Vectorized over wavelengths
    n_w = len(wavelengths)
    
    # Initialize M as identity for each wavelength
    M = np.zeros((n_w, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0

    for j in range(n_layers):
        n_j = materials[j]
        d_j = thicknesses[j]
        # delta_j = 2*pi*n_j*d_j / wavelength
        delta = 2.0 * np.pi * n_j * d_j / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        
        # Layer matrix for each wavelength
        # M_j = [[c, i*s/n_j], [i*n_j*s, c]]
        Mj = np.zeros((n_w, 2, 2), dtype=complex)
        Mj[:, 0, 0] = c
        Mj[:, 0, 1] = 1j * s / n_j
        Mj[:, 1, 0] = 1j * n_j * s
        Mj[:, 1, 1] = c
        
        # M = M * M_j
        M = np.einsum('wki,wkj->wij', M, Mj)

    # Multiply by substrate vector (1, n_sub)
    B = M[:, 0, 0] * 1.0 + M[:, 0, 1] * n_sub
    C = M[:, 1, 0] * 1.0 + M[:, 1, 1] * n_sub
    
    Y = C / B
    r = (1.0 - Y) / (1.0 + Y)
    R = np.abs(r) ** 2
    return R


def _mean_reflectance_pct(materials_list, thicknesses_list, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_sub = problem["substrate_index"]
    materials = [problem["materials"][m] for m in materials_list]
    wavelengths = np.linspace(band[0], band[1], 401)
    R = _reflectance_spectrum(materials, thicknesses_list, wavelengths, n_sub)
    return np.mean(R) * 100.0


def _is_valid(materials_list, thicknesses_list, problem):
    """Check if the design satisfies all constraints."""
    if len(materials_list) > problem["max_layers"]:
        return False
    if len(materials_list) != len(thicknesses_list):
        return False
    for i, t in enumerate(thicknesses_list):
        if t < 0 or t > problem["max_layer_nm"]:
            return False
        if materials_list[i] not in problem["materials"]:
            return False
    if sum(thicknesses_list) > problem["max_total_nm"]:
        return False
    return True


def _nelder_mead(f, x0, args=(), max_iter=500, step=0.1, timeout=None):
    """
    Simple Nelder-Mead optimization.
    f: objective function
    x0: initial point (list or array)
    args: extra args for f
    Returns: (best_x, best_f)
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += step if x0[i] == 0 else 0.05 * x0[i]
        simplex.append(xi)
    
    # Evaluate initial points
    fvals = [f(x, *args) for x in simplex]
    
    for it in range(max_iter):
        if timeout is not None and time.time() > timeout:
            break
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        best_f = fvals[0]
        worst_f = fvals[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr, *args)
        
        if fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < worst_f:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe, *args)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc, *args)
            if fc < worst_f:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i], *args)
    
    # Find best
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def design(problem):
    """Design an antireflection coating stack."""
    global _START
    _START = time.time()
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Central wavelength
    lambda_c = 0.5 * (band[0] + band[1])
    
    # Best result so far
    best_materials = ["L"]
    best_thicknesses = [lambda_c / (4.0 * n_L)]
    best_score = _mean_reflectance_pct(best_materials, best_thicknesses, problem)
    
    # Try various layer counts with Nelder-Mead optimization
    # Start with small layer counts and increase
    
    for n_layers in range(1, min(max_layers, 15) + 1):
        if _remaining() < 5.0:
            break
        
        # Initial guess: quarter-wave stack
        # For a quarter-wave stack, the optimal pattern alternates L and H
        # Starting with L on air side for AR coating
        init_materials = []
        init_thicknesses = []
        for i in range(n_layers):
            if i % 2 == 0:
                init_materials.append("L")
                init_thicknesses.append(lambda_c / (4.0 * n_L))
            else:
                init_materials.append("H")
                init_thicknesses.append(lambda_c / (4.0 * n_H))
        
        # Check if initial guess is valid
        if not _is_valid(init_materials, init_thicknesses, problem):
            # Scale down if too thick
            total = sum(init_thicknesses)
            if total > max_total_nm:
                scale = max_total_nm / total
                init_thicknesses = [t * scale for t in init_thicknesses]
            if not _is_valid(init_materials, init_thicknesses, problem):
                continue
        
        # Define objective function for Nelder-Mead
        def objective(thicknesses_arr, mats=init_materials):
            if any(t < 0 or t > max_layer_nm for t in thicknesses_arr):
                return 100.0
            if sum(thicknesses_arr) > max_total_nm:
                return 100.0
            return _mean_reflectance_pct(mats, list(thicknesses_arr), problem)
        
        x0 = np.array(init_thicknesses, dtype=float)
        
        # Run Nelder-Mead with a time limit
        timeout = time.time() + max(1.0, _remaining() - 2.0)
        try:
            best_x, best_f = _nelder_mead(objective, x0, max_iter=200, step=5.0, timeout=timeout)
        except Exception:
            continue
        
        if best_f < best_score:
            # Check validity
            if _is_valid(init_materials, list(best_x), problem):
                best_materials = init_materials[:]
                best_thicknesses = [float(t) for t in best_x]
                best_score = best_f
    
    # Also try a few different initial patterns
    patterns = [
        ["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
        ["L", "L", "H", "H", "L", "L", "H", "H", "L", "L"],
        ["H", "H", "L", "L", "H", "H", "L", "L", "H", "H"],
    ]
    
    for pattern in patterns:
        if _remaining() < 5.0:
            break
        
        for n_layers in [4, 6, 8, 10, 12]:
            if n_layers > max_layers:
                continue
            if _remaining() < 5.0:
                break
            
            mats = pattern[:n_layers]
            thicknesses = []
            for m in mats:
                n_m = problem["materials"][m]
                thicknesses.append(lambda_c / (4.0 * n_m))
            
            total = sum(thicknesses)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicknesses = [t * scale for t in thicknesses]
            
            if not _is_valid(mats, thicknesses, problem):
                continue
            
            def objective(thicknesses_arr, mats=mats):
                if any(t < 0 or t > max_layer_nm for t in thicknesses_arr):
                    return 100.0
                if sum(thicknesses_arr) > max_total_nm:
                    return 100.0
                return _mean_reflectance_pct(mats, list(thicknesses_arr), problem)
            
            x0 = np.array(thicknesses, dtype=float)
            timeout = time.time() + max(1.0, _remaining() - 2.0)
            try:
                best_x, best_f = _nelder_mead(objective, x0, max_iter=150, step=5.0, timeout=timeout)
            except Exception:
                continue
            
            if best_f < best_score and _is_valid(mats, list(best_x), problem):
                best_materials = mats[:]
                best_thicknesses = [float(t) for t in best_x]
                best_score = best_f
    
    # Final validity check
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to simple quarter-wave
        best_materials = ["L"]
        best_thicknesses = [lambda_c / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
