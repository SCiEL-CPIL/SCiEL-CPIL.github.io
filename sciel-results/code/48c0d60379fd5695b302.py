"""Antireflection coating design via simulated annealing on layer thicknesses."""
import time
import numpy as np

def _reflectance_spectrum(thicknesses, labels, n_amb, n_sub, n_L, n_H, wavelengths):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the
    characteristic-matrix method.
    """
    n = len(thicknesses)
    if n == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wavelengths, r * r, dtype=float)
    
    # Precompute refractive indices for each layer
    n_layers = np.array([n_L if lab == 'L' else n_H for lab in labels], dtype=float)
    
    reflectances = np.zeros_like(wavelengths, dtype=float)
    
    for i, lam in enumerate(wavelengths):
        # Optical thickness in radians: 2*pi*n*d/lambda
        beta = 2.0 * np.pi * n_layers * np.array(thicknesses) / lam
        
        # Build characteristic matrix
        # For each layer, the matrix is:
        # [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
        # We multiply all layer matrices, then multiply by substrate admittance
        
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Start with identity
        M = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=complex)
        
        for j in range(n):
            layer_M = np.array([
                [c[j], 1j * s[j] / n_layers[j]],
                [1j * n_layers[j] * s[j], c[j]]
            ], dtype=complex)
            M = M @ layer_M
        
        # Total admittance: Y = M[1,0]/M[0,0] (in units of n_sub)
        # Actually, the characteristic matrix relates:
        # [E, H] at input = M @ [E, H] at substrate
        # For normal incidence, the reflectance is:
        # r = (n_amb * M[0,0] + M[1,0] - n_amb * M[0,1] - M[1,1]) /
        #     (n_amb * M[0,0] + M[1,0] + n_amb * M[0,1] + M[1,1])
        # Wait, let me use the standard formula.
        
        # The standard characteristic matrix method:
        # The total matrix M_total = M_1 @ M_2 @ ... @ M_N
        # where each M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #                    [i*n_j*sin(delta_j), cos(delta_j)]]
        # 
        # Then the reflectance at normal incidence is:
        # r = (n_0 * M_total[0,0] + M_total[1,0] - n_0 * M_total[0,1] - M_total[1,1]) /
        #     (n_0 * M_total[0,0] + M_total[1,0] + n_0 * M_total[0,1] + M_total[1,1])
        # where n_0 is the ambient index.
        
        M00, M01 = M[0, 0], M[0, 1]
        M10, M11 = M[1, 0], M[1, 1]
        
        numerator = n_amb * M00 + M10 - n_amb * M01 - M11
        denominator = n_amb * M00 + M10 + n_amb * M01 + M11
        
        r = numerator / denominator
        reflectances[i] = np.abs(r) ** 2
    
    return reflectances

def design(problem):
    """
    Design an antireflection coating stack using simulated annealing.
    Returns (materials, thicknesses) where materials is a list of 'L'/'H'
    and thicknesses is a list of nm values.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_amb = problem.get("ambient_index", 1.0)
    n_sub = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {'L': 1.38, 'H': 2.35})
    n_L = materials['L']
    n_H = materials['H']
    band = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Objective: mean reflectance in percent
    def objective(labels, thicknesses):
        """Return mean reflectance in percent."""
        n = len(thicknesses)
        if n == 0:
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * r * r
        refl = _reflectance_spectrum(thicknesses, labels, n_amb, n_sub, n_L, n_H, wavelengths)
        return 100.0 * np.mean(refl)
    
    # Initial solution: quarter-wave stack alternating L/H centered at 600nm
    centre_wl = 0.5 * (band[0] + band[1])
    # Start with a reasonable number of layers
    n_init = 6
    labels = ['L', 'H'] * (n_init // 2)
    if len(labels) > n_init:
        labels = labels[:n_init]
    thicknesses = [centre_wl / (4.0 * (n_L if lab == 'L' else n_H)) for lab in labels]
    
    # Ensure constraints
    if sum(thicknesses) > max_total_nm:
        scale = max_total_nm / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
    thicknesses = [min(t, max_layer_nm) for t in thicknesses]
    
    best_labels = list(labels)
    best_thicknesses = list(thicknesses)
    best_score = objective(best_labels, best_thicknesses)
    
    # Simulated annealing
    current_labels = list(labels)
    current_thicknesses = list(thicknesses)
    current_score = best_score
    
    T = 1.0  # Initial temperature (in percent reflectance units)
    T_min = 0.001
    cooling = 0.999
    max_iter = 50000
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb: randomly change one layer's thickness or swap two labels
        op = rng.random()
        
        if op < 0.7:
            # Perturb a random thickness
            if len(current_thicknesses) == 0:
                continue
            idx = rng.integers(0, len(current_thicknesses))
            delta = rng.normal(0, 20.0)
            new_t = current_thicknesses[idx] + delta
            new_t = max(0.0, min(new_t, max_layer_nm))
            new_thicknesses = list(current_thicknesses)
            new_thicknesses[idx] = new_t
            
            # Check total thickness constraint
            if sum(new_thicknesses) > max_total_nm:
                scale = max_total_nm / sum(new_thicknesses)
                new_thicknesses = [t * scale for t in new_thicknesses]
                new_thicknesses = [min(t, max_layer_nm) for t in new_thicknesses]
            
            new_labels = list(current_labels)
            new_score = objective(new_labels, new_thicknesses)
        else:
            # Swap two layer labels
            if len(current_labels) < 2:
                continue
            i, j = rng.choice(len(current_labels), size=2, replace=False)
            new_labels = list(current_labels)
            new_labels[i], new_labels[j] = new_labels[j], new_labels[i]
            new_thicknesses = list(current_thicknesses)
            new_score = objective(new_labels, new_thicknesses)
        
        # Accept/reject
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < np.exp(-delta_score / max(T, 1e-10)):
            current_labels = new_labels
            current_thicknesses = new_thicknesses
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_labels = list(current_labels)
                best_thicknesses = list(current_thicknesses)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_labels, best_thicknesses
