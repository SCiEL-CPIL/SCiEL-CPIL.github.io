"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating using differential evolution."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    ambient_index = problem["ambient_index"]  # 1.0
    substrate_index = problem["substrate_index"]  # 1.52
    materials = problem["materials"]  # {'L': 1.38, 'H': 2.35}
    band_nm = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400

    n_wavelengths = 401
    wavelengths = np.linspace(band_nm[0], band_nm[1], n_wavelengths)

    n_L = materials["L"]
    n_H = materials["H"]

    def compute_reflectance(thicknesses, num_layers):
        """Compute mean reflectance using characteristic matrix method."""
        # thicknesses: list of (material_index, thickness_nm)
        # material_index: 0 for L, 1 for H
        
        # Build the characteristic matrix for each layer
        # M_j = [[cos(delta_j), (i/sin(delta_j)) * (1/n_j)],
        #        [i * n_j * sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*n_j*t_j / lambda
        
        total_M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for idx in range(num_layers):
            mat_idx, t_nm = thicknesses[idx]
            n_j = n_L if mat_idx == 0 else n_H
            
            # Optical thickness in radians
            delta = 2.0 * np.pi * n_j * t_nm / wavelengths  # shape (n_wavelengths,)
            
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Handle sin_d close to zero to avoid division by zero
            # When sin_d is very small, use limit: 1/sin_d -> infinity, but physically
            # the layer is very thin. We'll use a small epsilon.
            sin_d_safe = np.where(np.abs(sin_d) < 1e-12, 1e-12, sin_d)
            
            # Matrix elements
            B11 = cos_d
            B12 = 1j / (n_j * sin_d_safe)
            B21 = 1j * n_j * sin_d_safe
            B22 = cos_d
            
            # M_j for each wavelength
            M_j = np.stack([
                np.stack([B11, B12], axis=-1),
                np.stack([B21, B22], axis=-1)
            ], axis=-2)  # shape (n_wavelengths, 2, 2)
            
            # Multiply into total: total_M = total_M @ M_j
            # total_M is (2,2) but we need to handle per-wavelength
            # Let's restructure: total_M as (n_wavelengths, 2, 2)
            pass
        
        # Actually, let me restructure to handle per-wavelength matrices properly
        total_M = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex), (n_wavelengths, 1, 1))
        
        for idx in range(num_layers):
            mat_idx, t_nm = thicknesses[idx]
            n_j = n_L if mat_idx == 0 else n_H
            
            delta = 2.0 * np.pi * n_j * t_nm / wavelengths
            
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            sin_d_safe = np.where(np.abs(sin_d) < 1e-12, 1e-12, sin_d)
            
            B11 = cos_d
            B12 = 1j / (n_j * sin_d_safe)
            B21 = 1j * n_j * sin_d_safe
            B22 = cos_d
            
            M_j = np.empty((n_wavelengths, 2, 2), dtype=complex)
            M_j[:, 0, 0] = B11
            M_j[:, 0, 1] = B12
            M_j[:, 1, 0] = B21
            M_j[:, 1, 1] = B22
            
            # total_M = total_M @ M_j
            total_M = np.einsum('wij,wjk->wik', total_M, M_j)
        
        # Now compute reflectance
        # The characteristic matrix M = [[M11, M12], [M21, M22]]
        # For normal incidence:
        # Y = (M21 + M22 * n_s) / (M11 + M12 * n_s)  where n_s is substrate index
        # Wait, the standard formula:
        # The admittance Y = (M21 + M22 * n_0) / (M11 + M12 * n_0) where n_0 is the incident medium
        # Actually, let me recall the standard characteristic matrix method:
        
        # For a stack on substrate with incident medium n_0:
        # M_total = product of individual layer matrices
        # Y = (M21 + M22 * n_s) / (M11 + M12 * n_s)  -- this is the effective admittance at the front
        # R = |(n_0 - Y) / (n_0 + Y)|^2
        
        # Wait, I need to be careful. The standard formulation:
        # Each layer j has matrix:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        # (for non-magnetic, normal incidence)
        
        # Total matrix M = M_1 * M_2 * ... * M_N
        # Then the admittance seen from the incident side:
        # Y = (M[1,0] + M[1,1] * n_s) / (M[0,0] + M[0,1] * n_s)
        # where n_s is the substrate refractive index
        
        # Reflectance:
        # R = |(n_0 - Y)/(n_0 + Y)|^2
        
        M11 = total_M[:, 0, 0]
        M12 = total_M[:, 0, 1]
        M21 = total_M[:, 1, 0]
        M22 = total_M[:, 1, 1]
        
        Y = (M21 + M22 * substrate_index) / (M11 + M12 * substrate_index)
        
        R = np.abs((ambient_index - Y) / (ambient_index + Y))**2
        R = np.clip(R, 0.0, 1.0)
        
        return np.mean(R) * 100.0  # in percent

    def objective(params):
        """Objective function for optimization."""
        num_layers = int(params[0])
        if num_layers < 1:
            num_layers = 1
        num_layers = min(num_layers, max_layers)
        
        # Extract material indices and thicknesses
        thicknesses = []
        for i in range(num_layers):
            mat_idx = int(params[1 + 2*i])
            t = params[2 + 2*i]
            mat_idx = 0 if mat_idx < 0.5 else 1
            t = max(0.0, min(t, max_layer_nm))
            thicknesses.append((mat_idx, t))
        
        # Check total thickness constraint
        total_t = sum(t for _, t in thicknesses)
        if total_t > max_total_nm:
            return max_total_nm  # penalty
        
        return compute_reflectance(thicknesses, num_layers)

    # Use differential evolution
    # Parameters: [num_layers, mat1, t1, mat2, t2, ..., matN, tN]
    # But we need to handle variable number of layers
    
    # Strategy: start with a few layers and increase
    best_result = None
    best_score = float('inf')
    
    # Try different numbers of layers
    for n_layers in [4, 6, 8, 10, 12]:
        if time.time() > deadline:
            break
        
        # Parameters: for each layer, material (0 or 1) and thickness (0 to max_layer_nm)
        # But we also need to ensure total thickness <= max_total_nm
        # Let's use a simpler approach: optimize thicknesses directly with material pattern fixed or alternating
        
        # For simplicity, let's try alternating patterns and optimize thicknesses
        # Pattern: L H L H ... or H L H L ...
        
        for start_mat in [0, 1]:  # 0=L, 1=H
            if time.time() > deadline:
                break
            
            # Create alternating pattern
            pattern = [(start_mat + i) % 2 for i in range(n_layers)]
            
            # Optimize thicknesses only
            def obj_thickness(thick):
                thicknesses = [(pattern[i], thick[i]) for i in range(n_layers)]
                total_t = sum(t for _, t in thicknesses)
                if total_t > max_total_nm:
                    return max_total_nm
                return compute_reflectance(thicknesses, n_layers)
            
            bounds = [(0.0, max_layer_nm)] * n_layers
            
            try:
                result = differential_evolution(
                    obj_thickness,
                    bounds,
                    seed=42,
                    maxiter=100,
                    tol=1e-6,
                    polish=True,
                    workers=1
                )
                if result.success:
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_result = [(pattern[i], result.x[i]) for i in range(n_layers)]
            except Exception:
                pass
    
    # If no valid result, return a simple design
    if best_result is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    # Convert to output format
    materials_list = []
    thicknesses_list = []
    for mat_idx, t in best_result:
        materials_list.append('L' if mat_idx == 0 else 'H')
        thicknesses_list.append(t)
    
    return materials_list, thicknesses_list
# EVOLVE-BLOCK-END
