"""Design broadband antireflection coating via differential evolution + polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s wall-clock kill


def _characteristic_matrix(n, d, lam):
    """Return the 2x2 characteristic matrix for a single layer."""
    beta = 2.0 * math.pi * n * d / lam
    cb = math.cos(beta)
    sb = math.sin(beta)
    # For normal incidence, B = n (admittance = n)
    # M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    m00 = cb
    m01 = 1j * sb / n
    m10 = 1j * n * sb
    m11 = cb
    return np.array([[m00, m01], [m10, m11]], dtype=complex)


def _reflectance_stack(materials, thicknesses, n_air, n_sub, lam):
    """Compute reflectance (0..1) for a single wavelength."""
    # Build total matrix from air side to substrate
    # M_total = M1 * M2 * ... * M_N
    # Then Y = M[1,0]/M[0,0] (input admittance)
    # R = |(Y - n_air)/(Y + n_air)|^2
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for i in range(len(materials)):
        n = materials[i]
        d = thicknesses[i]
        Mi = _characteristic_matrix(n, d, lam)
        M = M @ Mi
    Y = M[1, 0] / M[0, 0]
    r = (Y - n_air) / (Y + n_air)
    R = abs(r) ** 2
    return R


def _mean_reflectance(materials, thicknesses, problem, n_samples=401):
    """Compute mean reflectance in percent over the band."""
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], n_samples)
    total_R = 0.0
    for lam in wavelengths:
        R = _reflectance_stack(materials, thicknesses, n_air, n_sub, lam)
        total_R += R
    return 100.0 * total_R / n_samples


def _validate(materials, thicknesses, problem):
    """Check constraints. Returns True if valid."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-6:
        return False
    return True


def _fix_thicknesses(thicknesses, max_layer_nm, max_total_nm):
    """Clip thicknesses to valid range and scale down if total exceeds limit."""
    thicknesses = [max(0.0, min(t, max_layer_nm)) for t in thicknesses]
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    return thicknesses


def _quarter_wave_stack(n_air, n_sub, n_L, n_H, n_layers, center_wl):
    """Generate a quarter-wave stack starting from a reasonable guess."""
    materials = []
    thicknesses = []
    # Start with L (low index) at air interface
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append(n_L)
            thicknesses.append(center_wl / (4.0 * n_L))
        else:
            materials.append(n_H)
            thicknesses.append(center_wl / (4.0 * n_H))
    return materials, thicknesses


def _random_stack(n_layers, n_L, n_H, max_layer_nm, max_total_nm):
    """Generate a random valid stack."""
    materials = []
    thicknesses = []
    remaining = max_total_nm
    for i in range(n_layers):
        if i % 2 == 0:
            n = n_L
        else:
            n = n_H
        materials.append(n)
        # Random thickness between 50 and min(400, remaining)
        max_t = min(max_layer_nm, remaining)
        if max_t < 50:
            t = max(0, remaining)
        else:
            t = 50.0 + np.random.random() * (max_t - 50.0)
        thicknesses.append(t)
        remaining -= t
    # Scale if needed
    thicknesses = _fix_thicknesses(thicknesses, max_layer_nm, max_total_nm)
    return materials, thicknesses


def _diff_evaluate(x, problem, n_L, n_H):
    """Evaluate a candidate: x encodes n_layers and thicknesses."""
    n_layers = int(round(x[0]))
    n_layers = max(1, min(n_layers, problem.get("max_layers", 20)))
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    thicknesses = list(x[1:1 + n_layers])
    # Pad or trim to n_layers
    while len(thicknesses) < n_layers:
        thicknesses.append(0.0)
    thicknesses = thicknesses[:n_layers]
    thicknesses = _fix_thicknesses(thicknesses, max_layer_nm, max_total_nm)
    
    # Build materials: alternate L/H starting with L
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append(n_L)
        else:
            materials.append(n_H)
    
    if not _validate(materials, thicknesses, problem):
        return 100.0  # penalty
    
    return _mean_reflectance(materials, thicknesses, problem)


def _local_polish(materials, thicknesses, problem, n_L, n_H, max_iters=5000):
    """Local refinement using Nelder-Mead style perturbation."""
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    n_layers = len(materials)
    
    best_thick = list(thicknesses)
    best_score = _mean_reflectance(materials, best_thick, problem)
    
    # Use scipy if available, otherwise manual hill climbing
    try:
        from scipy.optimize import minimize
        def objective(th):
            th = _fix_thicknesses(th, max_layer_nm, max_total_nm)
            if not _validate(materials, th, problem):
                return 100.0
            return _mean_reflectance(materials, th, problem)
        
        result = minimize(objective, best_thick, method='Nelder-Mead',
                         options={'maxiter': max_iters, 'xatol': 0.1, 'fatol': 1e-5})
        if result.fun < best_score:
            best_thick = _fix_thicknesses(result.x, max_layer_nm, max_total_nm)
            best_score = result.fun
    except Exception:
        # Manual local search
        step = 10.0
        for _ in range(max_iters):
            if time.time() > _DEADLINE - 5.0:
                break
            improved = False
            for i in range(n_layers):
                for delta in [step, -step, step/2, -step/2]:
                    new_thick = list(best_thick)
                    new_thick[i] = max(0, min(new_thick[i] + delta, max_layer_nm))
                    new_thick = _fix_thicknesses(new_thick, max_layer_nm, max_total_nm)
                    if _validate(materials, new_thick, problem):
                        s = _mean_reflectance(materials, new_thick, problem)
                        if s < best_score - 1e-6:
                            best_thick = new_thick
                            best_score = s
                            improved = True
                            break
            if not improved:
                step *= 0.7
                if step < 0.5:
                    break
    
    return best_thick, best_score


def design(problem):
    """Design a broadband AR coating using differential evolution + local polish."""
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    center_wl = 0.5 * (band[0] + band[1])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Start with a good initial guess: quarter-wave stack with 9 layers
    n_init = 9
    init_materials, init_thicknesses = _quarter_wave_stack(n_air, n_sub, n_L, n_H, n_init, center_wl)
    init_score = _mean_reflectance(init_materials, init_thicknesses, problem)
    
    best_materials = list(init_materials)
    best_thicknesses = list(init_thicknesses)
    best_score = init_score
    
    # Differential evolution over thicknesses for a fixed layer count
    # We'll try several layer counts and use DE to optimize thicknesses
    
    from scipy.optimize import differential_evolution
    
    def _run_de(n_layers, popsize, maxiter, tol):
        nonlocal best_materials, best_thicknesses, best_score
        
        materials = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials.append(n_L)
            else:
                materials.append(n_H)
        
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        def objective(th):
            th = _fix_thicknesses(th, max_layer_nm, max_total_nm)
            if not _validate(materials, th, problem):
                return 100.0
            return _mean_reflectance(materials, th, problem)
        
        try:
            result = differential_evolution(
                objective, bounds,
                popsize=popsize,
                maxiter=maxiter,
                tol=tol,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if result.fun < best_score:
                best_thicknesses = _fix_thicknesses(result.x, max_layer_nm, max_total_nm)
                best_materials = list(materials)
                best_score = result.fun
        except Exception:
            pass
    
    # Phase 1: DE for 9 layers (known good starting point)
    _run_de(9, popsize=15, maxiter=80, tol=1e-4)
    
    # Phase 2: DE for other layer counts if time permits
    for n_try in [11, 13, 7]:
        if time.time() > _DEADLINE - 30.0:
            break
        _run_de(n_try, popsize=12, maxiter=50, tol=1e-4)
    
    # Phase 3: Local polish on the best solution
    if time.time() < _DEADLINE - 10.0:
        polished_thick, polished_score = _local_polish(
            best_materials, best_thicknesses, problem, n_L, n_H, max_iters=3000
        )
        if polished_score < best_score:
            best_thicknesses = polished_thick
            best_score = polished_score
    
    # Final validation
    best_thicknesses = _fix_thicknesses(best_thicknesses, max_layer_nm, max_total_nm)
    if not _validate(best_materials, best_thicknesses, problem):
        # Fall back to initial
        best_materials = list(init_materials)
        best_thicknesses = list(init_thicknesses)
    
    # Convert to 'L'/'H' labels
    labels = []
    for m in best_materials:
        if m == n_L:
            labels.append('L')
        else:
            labels.append('H')
    
    return labels, best_thicknesses
# EVOLVE-BLOCK-END
