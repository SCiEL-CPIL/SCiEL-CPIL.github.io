"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """Compute reflectance (fraction) at each wavelength using characteristic matrix method."""
    nL = materials['L']
    nH = materials['H']
    n0 = ambient_index
    ns = substrate_index
    N = len(thicknesses)
    # We need the layer indices in order from air to substrate
    # But thicknesses alone don't tell us which material. We need labels too.
    # Actually the function signature should include labels.
    raise NotImplementedError

def _reflectance_spectrum_with_labels(labels, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """Compute reflectance (fraction) at each wavelength using characteristic matrix method."""
    nL = materials['L']
    nH = materials['H']
    n0 = ambient_index
    ns = substrate_index
    N = len(labels)
    
    # Build layer index array
    n_layers = np.zeros(N)
    for i, lab in enumerate(labels):
        n_layers[i] = nL if lab == 'L' else nH
    
    # Wavelengths in nm
    # k = 2*pi*n*d / lambda
    # For each wavelength, compute M matrix product
    
    refl = np.zeros(len(wavelengths))
    for wi, lam in enumerate(wavelengths):
        # Build M for each layer
        # M_i = [[cos(delta_i), (i/sin(delta_i))/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # where delta_i = 2*pi*n_i*d_i / lambda
        # For normal incidence, s-polarization doesn't matter, use TE-like form
        # Actually for normal incidence, the characteristic matrix is:
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        
        M_total = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        for i in range(N):
            delta = 2.0 * math.pi * n_layers[i] * thicknesses[i] / lam
            c = math.cos(delta)
            s = math.sin(delta)
            Mi = np.array([[c, 1j*s/n_layers[i]], [1j*n_layers[i]*s, c]], dtype=complex)
            M_total = M_total @ Mi
        
        # Reflectance: R = |(B - A*ns)/(B + A*ns)|^2 where [A,B]^T = M_total @ [1, 0]^T? 
        # Actually: Y = (M11 + M12*ns) / (M21 + M22*ns) is the admittance
        # R = |(Y - n0)/(Y + n0)|^2
        M11, M12 = M_total[0,0], M_total[0,1]
        M21, M22 = M_total[1,0], M_total[1,1]
        Y = (M11 + M12 * ns) / (M21 + M22 * ns)
        R = np.abs((Y - n0) / (Y + n0))**2
        refl[wi] = R
    return refl

def _mean_reflectance_percent(labels, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    materials = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band = problem['band_nm']
    wavelengths = np.linspace(band[0], band[1], 401)
    refl = _reflectance_spectrum_with_labels(labels, thicknesses, materials, ambient_index, substrate_index, wavelengths)
    return 100.0 * np.mean(refl)

def _is_valid(labels, thicknesses, problem):
    """Check constraints."""
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    if len(labels) > max_layers:
        return False
    if len(labels) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _quarter_wave_stack(n_layers, center_wl, materials, ambient_index, substrate_index, band):
    """Create a quarter-wave stack design."""
    nL = materials['L']
    nH = materials['H']
    # For a quarter-wave stack at center wavelength, thickness = center_wl / (4 * n)
    # We alternate L and H starting from air side
    # Actually, for AR coating, we want to match impedance. 
    # A common approach: start with L (low index) close to air, then H, etc.
    labels = []
    thicknesses = []
    for i in range(n_layers):
        if i % 2 == 0:
            labels.append('L')
            thicknesses.append(center_wl / (4.0 * nL))
        else:
            labels.append('H')
            thicknesses.append(center_wl / (4.0 * nH))
    return labels, thicknesses

def _nelder_mead(objective, x0, bounds, max_iter=200, xatol=0.1, fatol=1e-4, deadline=None):
    """Simple Nelder-Mead implementation with bounds and deadline check."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        step = 0.05 * abs(xi[i]) if xi[i] != 0 else 0.05
        xi[i] += step
        # Clip to bounds
        xi = np.clip(xi, bounds[0], bounds[1])
        simplex.append(xi)
    
    # Evaluate initial simplex
    values = [objective(x) for x in simplex]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Sort simplex by value
        idx = np.argsort(values)
        simplex = [simplex[i] for i in idx]
        values = [values[i] for i in idx]
        
        # Check convergence
        if abs(values[0] - values[-1]) < fatol:
            break
        if np.max(np.abs(simplex[0] - simplex[1])) < xatol:
            break
            
        # Calculate centroid of all but the worst point
        centroid = np.sum(simplex[:-1], axis=0) / n
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, bounds[0], bounds[1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[0], bounds[1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    values[i] = objective(simplex[i])
                    
    # Return best point
    idx = np.argmin(values)
    return simplex[idx], values[idx]

def _optimize_one(labels_init, thicknesses_init, problem, deadline, best_state):
    """Optimize a single design using Nelder-Mead, updating best_state if improved."""
    materials = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band = problem['band_nm']
    max_layer_nm = problem['max_layer_nm']
    max_total_nm = problem['max_total_nm']
    max_layers = problem['max_layers']
    
    n_layers = len(labels_init)
    
    def objective(x):
        # x contains thicknesses
        if time.time() > deadline:
            return 1e6
        # Check constraints
        for t in x:
            if t < 0 or t > max_layer_nm:
                return 1e6
        if sum(x) > max_total_nm:
            return 1e6
        labels = labels_init
        thicknesses = list(x)
        return _mean_reflectance_percent(labels, thicknesses, problem)
    
    x0 = np.array(thicknesses_init, dtype=float)
    bounds = (np.zeros(n_layers), np.full(n_layers, max_layer_nm))
    
    try:
        x_best, f_best = _nelder_mead(objective, x0, bounds, max_iter=200, xatol=0.1, fatol=1e-4, deadline=deadline)
        if time.time() > deadline:
            return
        if f_best < best_state['score']:
            best_state['score'] = f_best
            best_state['labels'] = list(labels_init)
            best_state['thicknesses'] = list(x_best)
    except Exception:
        pass

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    materials = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    center_wl = 0.5 * (band[0] + band[1])
    
    best_state = {
        'score': 1e6,
        'labels': ['L'],
        'thicknesses': [center_wl / (4.0 * materials['L'])]
    }
    
    # Generate initial designs
    initial_designs = []
    
    # 1. Single quarter-wave L layer
    labels, thicknesses = _quarter_wave_stack(1, center_wl, materials, ambient_index, substrate_index, band)
    initial_designs.append((labels, thicknesses))
    
    # 2. Single quarter-wave H layer
    labels, thicknesses = _quarter_wave_stack(1, center_wl, materials, ambient_index, substrate_index, band)
    labels = ['H']
    thicknesses = [center_wl / (4.0 * materials['H'])]
    initial_designs.append((labels, thicknesses))
    
    # 3. Two-layer quarter-wave stacks at different center wavelengths
    for frac in [0.4, 0.5, 0.6]:
        wl = band[0] + frac * (band[1] - band[0])
        for n in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            labels, thicknesses = _quarter_wave_stack(n, wl, materials, ambient_index, substrate_index, band)
            if sum(thicknesses) <= max_total_nm:
                initial_designs.append((labels, thicknesses))
    
    # 4. Some random designs
    np.random.seed(42)
    for _ in range(20):
        n = np.random.randint(2, max_layers + 1)
        labels = ['L' if np.random.random() < 0.5 else 'H' for _ in range(n)]
        thicknesses = [np.random.uniform(0, max_layer_nm) for _ in range(n)]
        if sum(thicknesses) <= max_total_nm:
            initial_designs.append((labels, thicknesses))
    
    # Evaluate all initial designs and keep top few for optimization
    scored_designs = []
    for labels, thicknesses in initial_designs:
        if not _is_valid(labels, thicknesses, problem):
            continue
        score = _mean_reflectance_percent(labels, thicknesses, problem)
        scored_designs.append((score, labels, thicknesses))
    
    scored_designs.sort(key=lambda x: x[0])
    
    # Update best with best initial design
    if scored_designs:
        best_score, best_labels, best_thicknesses = scored_designs[0]
        best_state['score'] = best_score
        best_state['labels'] = list(best_labels)
        best_state['thicknesses'] = list(best_thicknesses)
    
    # Optimize top designs
    max_optimize = 10
    for i, (score, labels, thicknesses) in enumerate(scored_designs[:max_optimize]):
        if time.time() > deadline:
            break
        _optimize_one(labels, thicknesses, problem, deadline, best_state)
    
    # Final check
    if not _is_valid(best_state['labels'], best_state['thicknesses'], problem):
        # Fallback to simple design
        best_state['labels'] = ['L']
        best_state['thicknesses'] = [center_wl / (4.0 * materials['L'])]
    
    return best_state['labels'], best_state['thicknesses']
# EVOLVE-BLOCK-END
