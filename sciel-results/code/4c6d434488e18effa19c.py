"""Improved antireflection coating design using gradient-free optimization."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _compute_reflectance(thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    
    Parameters:
    - thicknesses: array of physical thicknesses in nm
    - materials: dict mapping 'L'/'H' to refractive indices
    - ambient_index: refractive index of ambient (air)
    - substrate_index: refractive index of substrate (glass)
    - wavelengths: array of wavelengths in nm
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wavelengths, r**2, dtype=float)
    
    # Precompute
    k0 = 2 * np.pi / 1e9  # 2*pi / lambda in nm -> rad/nm
    
    # Build characteristic matrix for each wavelength
    # M = product of layer matrices * substrate interface
    # For normal incidence, each layer matrix is:
    # [cos(delta), i*sin(delta)/n]
    # [i*n*sin(delta), cos(delta)]
    # where delta = 2*pi*n*d/lambda
    
    reflectances = np.zeros(len(wavelengths), dtype=float)
    
    for i, lam in enumerate(wavelengths):
        # Start with substrate interface matrix
        # The characteristic matrix for the stack is:
        # M_total = M_1 * M_2 * ... * M_N
        # Then the admittance Y = M21 / M11 (normalized)
        # Actually, let's use the standard formulation.
        
        # Initialize the running product matrix
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            
            c = np.cos(delta)
            s = np.sin(delta)
            
            # Layer matrix
            lm11 = c
            lm12 = 1j * s / n_j
            lm21 = 1j * n_j * s
            lm22 = c
            
            # Multiply: M_new = M_old * M_layer
            nm11 = m11 * lm11 + m12 * lm21
            nm12 = m11 * lm12 + m12 * lm22
            nm21 = m21 * lm11 + m22 * lm21
            nm22 = m21 * lm12 + m22 * lm22
            
            m11, m12, m21, m22 = nm11, nm12, nm21, nm22
        
        # Now, the total field at the interface with ambient:
        # The characteristic matrix relates the fields at the air side to the substrate side.
        # Actually, the standard approach:
        # Let M be the product of all layer matrices.
        # Then the admittance at the air side is:
        # Y = (n_sub * m11 + i * m21) / (n_sub * m12 + i * m11)  ... this isn't quite right.
        
        # Let me use the correct formula:
        # The characteristic matrix M satisfies:
        # [E]   = [m11  m12] [E_sub]
        # [H]     [m21  m22] [H_sub]
        # where E and H are the electric and magnetic field amplitudes.
        
        # At the substrate, we have a plane wave in the substrate:
        # E_sub / H_sub = n_sub (for the forward wave)
        
        # So: E = m11 * E_sub + m12 * H_sub = (m11 + m12 * n_sub) * E_sub
        #     H = m21 * E_sub + m22 * H_sub = (m21 + m22 * n_sub) * E_sub
        
        # The admittance at the air side is:
        # Y = E / H = (m11 + m12 * n_sub) / (m21 + m22 * n_sub)
        
        # The reflectance is:
        # r = (n_amb - Y) / (n_amb + Y)
        # R = |r|^2
        
        Y = (m11 + m12 * substrate_index) / (m21 + m22 * substrate_index)
        r = (ambient_index - Y) / (ambient_index + Y)
        reflectances[i] = np.abs(r)**2
    
    return reflectances


def _score_design(labels, thicknesses, problem):
    """Compute the mean reflectance in percent for a given design."""
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band = problem["band_nm"]
    
    # Map labels to indices
    n_layers = len(labels)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    materials_arr = np.array([materials[l] for l in labels], dtype=float)
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Compute reflectance
    refl = _compute_reflectance(thicknesses_arr, materials_arr, ambient_index, substrate_index, wavelengths)
    
    return np.mean(refl) * 100.0  # in percent


def _is_valid(labels, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
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


def design(problem):
    """Design an antireflection coating using Nelder-Mead optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    materials = problem["materials"]
    band = problem["band_nm"]
    
    # Initial guess: a few quarter-wave layers
    centre = 0.5 * (band[0] + band[1])
    
    # Start with a simple 3-layer design: L-H-L
    initial_labels = ["L", "H", "L"]
    initial_thicknesses = [
        centre / (4.0 * materials["L"]),
        centre / (4.0 * materials["H"]),
        centre / (4.0 * materials["L"])
    ]
    
    # Ensure validity
    while not _is_valid(initial_labels, initial_thicknesses, problem):
        # Scale down if needed
        scale = max_total_nm / sum(initial_thicknesses)
        initial_thicknesses = [t * scale for t in initial_thicknesses]
        if not _is_valid(initial_labels, initial_thicknesses, problem):
            break
    
    best_labels = list(initial_labels)
    best_thicknesses = list(initial_thicknesses)
    best_score = _score_design(best_labels, best_thicknesses, problem)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    def objective(x):
        """Objective function for optimization."""
        # x contains: first n_layers are thicknesses, then one value per layer for label (0=L, 1=H)
        n_l = len(best_labels)
        thicknesses = x[:n_l]
        label_vals = x[n_l:]
        
        # Clip thicknesses
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        # Convert label values to labels
        labels = []
        for v in label_vals:
            labels.append("L" if v < 0.5 else "H")
        
        # Check total thickness
        if np.sum(thicknesses) > max_total_nm:
            return 100.0  # Penalty for invalid
        
        return _score_design(labels, thicknesses, problem)
    
    # Initial point
    x0 = np.array(initial_thicknesses + [0.0, 1.0, 0.0])
    
    # Bounds
    bounds = [(0, max_layer_nm)] * 3 + [(0, 1)] * 3
    
    # Options for Nelder-Mead
    options = {
        'maxiter': 500,
        'xatol': 1e-4,
        'fatol': 1e-5,
        'adaptive': True
    }
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options=options)
        
        if result.success or result.fun < best_score:
            x_opt = result.x
            n_l = len(best_labels)
            opt_thicknesses = np.clip(x_opt[:n_l], 0, max_layer_nm)
            opt_labels = ["L" if x_opt[n_l+i] < 0.5 else "H" for i in range(n_l)]
            
            if _is_valid(opt_labels, opt_thicknesses.tolist(), problem):
                opt_score = result.fun
                if opt_score < best_score:
                    best_labels = opt_labels
                    best_thicknesses = opt_thicknesses.tolist()
                    best_score = opt_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_labels, best_thicknesses
    
    # Try with more layers if time permits
    if time.time() < deadline - 30:
        for n_l in [5, 7]:
            if time.time() > deadline - 20:
                break
            
            # Initialize with alternating L/H
            labels = []
            thicknesses = []
            for i in range(n_l):
                if i % 2 == 0:
                    labels.append("L")
                    thicknesses.append(centre / (4.0 * materials["L"]))
                else:
                    labels.append("H")
                    thicknesses.append(centre / (4.0 * materials["H"]))
            
            # Scale to fit within max_total_nm
            total = sum(thicknesses)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicknesses = [t * scale for t in thicknesses]
            
            if not _is_valid(labels, thicknesses, problem):
                continue
            
            current_score = _score_design(labels, thicknesses, problem)
            
            # Optimize
            def objective_n(x, n_l=n_l):
                thicknesses = x[:n_l]
                label_vals = x[n_l:]
                
                thicknesses = np.clip(thicknesses, 0, max_layer_nm)
                labels = ["L" if v < 0.5 else "H" for v in label_vals]
                
                if np.sum(thicknesses) > max_total_nm:
                    return 100.0
                
                return _score_design(labels, thicknesses, problem)
            
            x0_n = np.array(thicknesses + [0.0 if i % 2 == 0 else 1.0 for i in range(n_l)])
            bounds_n = [(0, max_layer_nm)] * n_l + [(0, 1)] * n_l
            
            try:
                result_n = minimize(objective_n, x0_n, method='Nelder-Mead', bounds=bounds_n, options=options)
                
                if result_n.fun < best_score:
                    x_opt_n = result_n.x
                    opt_thicknesses = np.clip(x_opt_n[:n_l], 0, max_layer_nm)
                    opt_labels = ["L" if x_opt_n[n_l+i] < 0.5 else "H" for i in range(n_l)]
                    
                    if _is_valid(opt_labels, opt_thicknesses.tolist(), problem):
                        best_labels = opt_labels
                        best_thicknesses = opt_thicknesses.tolist()
                        best_score = result_n.fun
            except Exception:
                pass
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
