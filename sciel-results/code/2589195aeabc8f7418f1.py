import numpy as np
import time
from scipy.optimize import minimize

def _characteristic_matrix(n, d, wl, n0, ns):
    """Compute reflectance for a stack using characteristic matrix method.
    n: array of refractive indices of layers (in order from air to substrate)
    d: array of thicknesses in nm
    wl: array of wavelengths in nm
    n0: ambient index
    ns: substrate index
    Returns: reflectance (0 to 1) array
    """
    # For each wavelength, build the total characteristic matrix
    # M = M1 * M2 * ... * MN * Ms
    # where Mi = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # Ms = [1, 0; 0, 1/n_s] (for the substrate interface)
    
    reflectances = np.zeros(len(wl))
    for idx, w in enumerate(wl):
        # Total matrix starts as identity
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        for i in range(len(n)):
            delta = 2.0 * np.pi * n[i] * d[i] / w
            ci = np.cos(delta)
            si = np.sin(delta)
            Mi = np.array([[ci, 1j*si/n[i]], [1j*n[i]*si, ci]])
            M = M @ Mi
        # Substrate matrix
        Ms = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0/n_s+0j]])
        M = M @ Ms
        
        # Y = M12/M11, B = M22/M21... actually:
        # Y = (M11 + M12*ns) / (M21 + M22*ns) ... let me use the standard formula
        # Actually, the admittance Y = M21/M22 where M is the total matrix
        # Wait, let me be careful.
        
        # Standard: Y = (M11 + M12 * Ys) / (M21 + M22 * Ys) where Ys = 1/ns for the substrate
        # But actually for normal incidence, Ys = ns (the admittance of the substrate)
        # Let me use: Y = M21 / M22
        # No, the correct formula is:
        # The total characteristic matrix M = [M11 M12; M21 M22]
        # Y = (M11 + M12 * Ys) / (M21 + M22 * Ys)
        # where Ys = ns (for the semi-infinite substrate)
        
        Ys = n_s  # substrate admittance
        Y = (M[0,0] + M[0,1] * Ys) / (M[1,0] + M[1,1] * Ys)
        
        # Reflectance
        r = (n0 - Y) / (n0 + Y)
        reflectances[idx] = np.abs(r)**2
    
    return reflectances

def _mean_reflectance(labels, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    materials = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    
    # Convert labels to indices
    n_layers = np.array([materials[l] for l in labels])
    d_layers = np.array(thicknesses, dtype=float)
    
    # 401 equally spaced wavelengths
    wls = np.linspace(band[0], band[1], 401)
    
    refl = _characteristic_matrix(n_layers, d_layers, wls, n0, ns)
    return np.mean(refl) * 100.0

def design(problem):
    """Design a broadband AR coating."""
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre_wl = 0.5 * (band[0] + band[1])
    
    # Best result tracking
    best_score = float('inf')
    best_labels = None
    best_thick = None
    
    # Generate candidate layer sequences
    # Strategy: try various numbers of layers with L-heavy bias
    # Good AR structures often have odd number of layers, L-heavy
    
    def try_optimize(labels, init_thick, time_left_check=None):
        nonlocal best_score, best_labels, best_thick
        
        n = len(labels)
        if n == 0:
            return
        
        # Objective function
        def objective(thick_vec):
            # Ensure thicknesses are in valid range
            t = np.clip(thick_vec, 0.1, max_layer_nm)
            total = np.sum(t)
            if total > max_total_nm:
                # Penalize
                return 100.0 + total - max_total_nm
            return _mean_reflectance(labels, t, problem)
        
        # Initial guess
        x0 = np.array(init_thick, dtype=float)
        x0 = np.clip(x0, 0.1, max_layer_nm)
        
        # Bounds
        bounds = [(0.1, max_layer_nm)] * n
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          bounds=bounds,
                          options={'maxiter': 500, 'xatol': 0.5, 'fatol': 0.01,
                                   'adaptive': True})
            if res.success or res.fun < best_score:
                t_opt = np.clip(res.x, 0.1, max_layer_nm)
                if np.sum(t_opt) <= max_total_nm:
                    score = _mean_reflectance(labels, t_opt, problem)
                    if score < best_score:
                        best_score = score
                        best_labels = list(labels)
                        best_thick = [float(x) for x in t_opt]
        except Exception:
            pass
        
        # Check time
        if time_left_check:
            if time.time() > time_left_check:
                return True
        return False
    
    # Try various layer configurations
    # Start with simple ones and build up
    
    # 1-layer: just L
    labels = ['L']
    init = [centre_wl / (4.0 * nL)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 2-layer: L-H
    labels = ['L', 'H']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 3-layer: L-H-L
    labels = ['L', 'H', 'L']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH), centre_wl / (4.0 * nL)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 3-layer: H-L-H
    labels = ['H', 'L', 'H']
    init = [centre_wl / (4.0 * nH), centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 4-layer: L-H-L-H
    labels = ['L', 'H', 'L', 'H']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 2
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 4-layer: H-L-H-L
    labels = ['H', 'L', 'H', 'L']
    init = [centre_wl / (4.0 * nH), centre_wl / (4.0 * nL)] * 2
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 5-layer: L-H-L-H-L
    labels = ['L', 'H', 'L', 'H', 'L']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH), centre_wl / (4.0 * nL),
            centre_wl / (4.0 * nH), centre_wl / (4.0 * nL)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 5-layer: H-L-H-L-H
    labels = ['H', 'L', 'H', 'L', 'H']
    init = [centre_wl / (4.0 * nH), centre_wl / (4.0 * nL), centre_wl / (4.0 * nH),
            centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 6-layer: L-H-L-H-L-H
    labels = ['L', 'H', 'L', 'H', 'L', 'H']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 3
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 6-layer: H-L-H-L-H-L
    labels = ['H', 'L', 'H', 'L', 'H', 'L']
    init = [centre_wl / (4.0 * nH), centre_wl / (4.0 * nL)] * 3
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 7-layer: L-H-L-H-L-H-L
    labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH), centre_wl / (4.0 * nL),
            centre_wl / (4.0 * nH), centre_wl / (4.0 * nL), centre_wl / (4.0 * nH),
            centre_wl / (4.0 * nL)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 8-layer: L-H-L-H-L-H-L-H
    labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 4
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 9-layer: L-H-L-H-L-H-L-H-L
    labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH), centre_wl / (4.0 * nL),
            centre_wl / (4.0 * nH), centre_wl / (4.0 * nL), centre_wl / (4.0 * nH),
            centre_wl / (4.0 * nL), centre_wl / (4.0 * nH), centre_wl / (4.0 * nL)]
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # 10-layer: L-H-L-H-L-H-L-H-L-H
    labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
    init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 5
    if not try_optimize(labels, init, time.time() + deadline - 5):
        pass
    
    # If we have time, try more complex structures
    if time.time() < start_time + deadline - 10:
        # 12-layer
        labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
        init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 6
        if not try_optimize(labels, init, time.time() + deadline - 5):
            pass
        
        # 14-layer
        labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
        init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 7
        if not try_optimize(labels, init, time.time() + deadline - 5):
            pass
        
        # 16-layer
        labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
        init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 8
        if not try_optimize(labels, init, time.time() + deadline - 5):
            pass
        
        # 18-layer
        labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
        init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 9
        if not try_optimize(labels, init, time.time() + deadline - 5):
            pass
        
        # 20-layer
        labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
        init = [centre_wl / (4.0 * nL), centre_wl / (4.0 * nH)] * 10
        if not try_optimize(labels, init, time.time() + deadline - 5):
            pass
    
    # If no valid result found, return a safe default
    if best_labels is None:
        # Return a single L layer
        return ["L"], [centre_wl / (4.0 * nL)]
    
    return best_labels, best_thick
