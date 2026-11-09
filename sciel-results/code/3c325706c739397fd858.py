"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _reflectance_spectrum_full(labels, thicknesses, materials, substrate_index, band_nm, n_wl=401):
    """Compute mean reflectance (%) using characteristic matrix method."""
    nL = materials['L']
    nH = materials['H']
    n_sub = substrate_index
    n_air = 1.0
    
    wls = np.linspace(band_nm[0], band_nm[1], n_wl)
    k = 2 * np.pi / wls  # 2pi/lambda in nm^-1
    
    N = len(thicknesses)
    if N == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        R = np.abs(r) ** 2
        return np.mean(R) * 100.0
    
    # Determine n_i for each layer
    n_layers = np.array([nL if lab == 'L' else nH for lab in labels])
    d_layers = np.array(thicknesses, dtype=float)
    
    # delta_i = k * n_i * d_i, shape (n_wl, N)
    delta = k[:, None] * n_layers[None, :] * d_layers[None, :]
    
    # Build characteristic matrix for each wavelength
    M_acc = np.zeros((n_wl, 2, 2), dtype=complex)
    M_acc[:, 0, 0] = 1.0
    M_acc[:, 1, 1] = 1.0
    
    for i in range(N):
        delta_i = delta[:, i]
        n_i = n_layers[i]
        
        c = np.cos(delta_i)
        s = np.sin(delta_i)
        
        M_i = np.zeros((n_wl, 2, 2), dtype=complex)
        M_i[:, 0, 0] = c
        M_i[:, 0, 1] = 1j * s / n_i
        M_i[:, 1, 0] = 1j * n_i * s
        M_i[:, 1, 1] = c
        
        M_acc = np.einsum('wij,wjk->wik', M_acc, M_i)
    
    m11 = M_acc[:, 0, 0]
    m12 = M_acc[:, 0, 1]
    m21 = M_acc[:, 1, 0]
    m22 = M_acc[:, 1, 1]
    
    Y = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    r = (n_air - Y) / (n_air + Y)
    R = np.abs(r) ** 2
    
    return np.mean(R) * 100.0


def design(problem):
    """Design an antireflection coating."""
    t_start = time.time()
    deadline = t_start + 140.0
    
    nL = problem['materials']['L']
    nH = problem['materials']['H']
    n_sub = problem['substrate_index']
    band_nm = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thick = None
    best_score = np.inf
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Design 0: Quarter-wave stack, alternating L/H starting with L
    labels0 = []
    thick0 = []
    for i in range(20):
        if i % 2 == 0:
            labels0.append('L')
            thick0.append(center_wl / (4 * nL))
        else:
            labels0.append('H')
            thick0.append(center_wl / (4 * nH))
    initial_designs.append((labels0, thick0))
    
    # Design 1: Quarter-wave stack starting with H
    labels1 = []
    thick1 = []
    for i in range(20):
        if i % 2 == 0:
            labels1.append('H')
            thick1.append(center_wl / (4 * nH))
        else:
            labels1.append('L')
            thick1.append(center_wl / (4 * nL))
    initial_designs.append((labels1, thick1))
    
    # Design 2: Graded thickness profile
    labels2 = []
    thick2 = []
    for i in range(20):
        frac = 0.25 + 0.25 * np.sin(np.pi * i / 19.0)
        if i % 2 == 0:
            labels2.append('L')
            thick2.append(center_wl * frac / nL)
        else:
            labels2.append('H')
            thick2.append(center_wl * frac / nH)
    initial_designs.append((labels2, thick2))
    
    # Designs 3-7: Random with structure
    np.random.seed(42)
    for r in range(5):
        labels_r = []
        thick_r = []
        for i in range(20):
            if np.random.random() < 0.5:
                labels_r.append('L')
                thick_r.append(np.random.uniform(40, 180))
            else:
                labels_r.append('H')
                thick_r.append(np.random.uniform(25, 120))
        initial_designs.append((labels_r, thick_r))
    
    for idx, (labels, thick) in enumerate(initial_designs):
        if time.time() > deadline:
            break
        
        thicknesses = np.array(thick, dtype=float)
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        if np.sum(thicknesses) > max_total_nm:
            thicknesses *= max_total_nm / np.sum(thicknesses)
        
        def objective(x, _labels=labels):
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 100.0
            if np.sum(x) > max_total_nm:
                return 100.0
            return _reflectance_spectrum_full(_labels, x, problem['materials'], n_sub, band_nm)
        
        try:
            # Use a larger initial simplex for better exploration
            simplex = 0.3
            result = minimize(objective, thicknesses, method='Nelder-Mead',
                            options={'maxiter': 800, 'xatol': 0.05, 'fatol': 1e-5,
                                    'initial_simplex': simplex})
            
            if result.fun < best_score:
                best_score = result.fun
                best_labels = labels
                best_thick = result.x.copy()
                
        except Exception:
            pass
    
    if best_labels is None:
        return ['L'], [center_wl / (4 * nL)]
    
    best_thick = np.round(best_thick, 2)
    best_thick = np.clip(best_thick, 0, max_layer_nm)
    
    return best_labels, list(best_thick)

# EVOLVE-BLOCK-END
