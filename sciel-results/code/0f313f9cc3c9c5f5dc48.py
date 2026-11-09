import numpy as np
from scipy.optimize import minimize
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit

    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Define a symmetric 14-layer stack
    # L H L H L H L L H L H L H L
    # This is symmetric around the center
    layer_types = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    n_layers = len(layer_types)
    
    # Precompute indices for the stack
    n_stack = np.array([n_L if t == 'L' else n_H for t in layer_types])
    
    # Wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def reflectance_spectrum(thicknesses):
        """
        Calculate reflectance spectrum for a given set of thicknesses.
        thicknesses: array of physical thicknesses in nm.
        Returns: array of reflectance values (0 to 1).
        """
        # Ensure thicknesses are within bounds
        t = np.clip(thicknesses, 0, max_layer_nm)
        
        # Check total thickness constraint
        if np.sum(t) > max_total_nm:
            # Scale down if necessary? Or just penalize in objective.
            # For spectrum calculation, we just compute what we have.
            pass
            
        # Characteristic matrix method
        # M = product of matrices for each layer
        # M_i = [[cos(delta_i), i/sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # delta_i = 2 * pi * n_i * t_i / lambda
        
        # We compute for all wavelengths at once using vectorization
        # t shape: (N_layers,)
        # wavelengths shape: (N_w,)
        
        # delta shape: (N_w, N_layers)
        delta = 2 * np.pi * n_stack[None, :] * t[None, :] / wavelengths[:, None]
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Avoid division by zero in sin_d
        sin_d_safe = np.where(np.abs(sin_d) < 1e-12, 1e-12, sin_d)
        
        # Matrix elements
        # M_i = [[cos_d, 1j * sin_d / n_i], [1j * n_i * sin_d, cos_d]]
        # Note: Standard characteristic matrix for normal incidence:
        # B_i = [[cos(delta_i), i * sin(delta_i) / n_i], [i * n_i * sin(delta_i), cos(delta_i)]]
        
        # We need to multiply these matrices.
        # Since we have many wavelengths, we can do this iteratively or use a loop.
        # N_layers is 14, N_w is 401. Looping over layers is fine.
        
        # Initialize M as identity for all wavelengths
        M = np.ones((len(wavelengths), 2, 2), dtype=complex)
        
        for i in range(n_layers):
            # M = M_i @ M
            # M_i is a 2x2 matrix for each wavelength
            # We can construct M_i for all wavelengths at once
            
            # cos_d_i shape: (N_w,)
            # sin_d_i shape: (N_w,)
            # n_i is scalar
            
            # M_i[0,0] = cos_d_i
            # M_i[0,1] = 1j * sin_d_i / n_i
            # M_i[1,0] = 1j * n_i * sin_d_i
            # M_i[1,1] = cos_d_i
            
            # M_new = M_i @ M
            # M_new[0,0] = M_i[0,0]*M[0,0] + M_i[0,1]*M[1,0]
            # M_new[0,1] = M_i[0,0]*M[0,1] + M_i[0,1]*M[1,1]
            # M_new[1,0] = M_i[1,0]*M[0,0] + M_i[1,1]*M[1,0]
            # M_new[1,1] = M_i[1,0]*M[0,1] + M_i[1,1]*M[1,1]
            
            cos_i = cos_d[:, i]
            sin_i = sin_d[:, i]
            n_i = n_stack[i]
            
            # Precompute terms
            m00_i = cos_i
            m01_i = 1j * sin_i / n_i
            m10_i = 1j * n_i * sin_i
            m11_i = cos_i
            
            # M is (N_w, 2, 2)
            m00 = M[:, 0, 0]
            m01 = M[:, 0, 1]
            m10 = M[:, 1, 0]
            m11 = M[:, 1, 1]
            
            new_m00 = m00_i * m00 + m01_i * m10
            new_m01 = m00_i * m01 + m01_i * m11
            new_m10 = m10_i * m00 + m11_i * m10
            new_m11 = m10_i * m01 + m11_i * m11
            
            M[:, 0, 0] = new_m00
            M[:, 0, 1] = new_m01
            M[:, 1, 0] = new_m10
            M[:, 1, 1] = new_m11
            
        # Now M is the total characteristic matrix for the stack.
        # We need to include the substrate.
        # The admittance of the substrate is n_s.
        # The reflectance is given by:
        # r = (n_0 * M11 + n_0 * n_s * M21 - M12 - n_s * M22) / (n_0 * M11 + n_0 * n_s * M21 + M12 + n_s * M22)
        # Wait, the formula depends on the definition of M.
        # Standard:
        # [E, H] at air side = M @ [E, H] at substrate side
        # At substrate side, H = n_s * E (for forward wave)
        # So [E_s, n_s E_s]
        # Then [E_0, H_0] = M @ [1, n_s] E_s
        # E_0 = (M11 + M12 n_s) E_s
        # H_0 = (M21 + M22 n_s) E_s
        # Admittance at air side Y_0 = H_0 / E_0 = (M21 + M22 n_s) / (M11 + M12 n_s)
        # Reflection coefficient r = (n_0 - Y_0) / (n_0 + Y_0)
        # r = (n_0 (M11 + M12 n_s) - (M21 + M22 n_s)) / (n_0 (M11 + M12 n_s) + (M21 + M22 n_s))
        
        M11 = M[:, 0, 0]
        M12 = M[:, 0, 1]
        M21 = M[:, 1, 0]
        M22 = M[:, 1, 1]
        
        n0 = ambient_index
        ns = substrate_index
        
        numerator = n0 * (M11 + M12 * ns) - (M21 + M22 * ns)
        denominator = n0 * (M11 + M12 * ns) + (M21 + M22 * ns)
        
        r = numerator / denominator
        R = np.abs(r) ** 2
        
        return R

    def objective(thicknesses):
        """
        Objective function to minimize.
        """
        t = np.clip(thicknesses, 0, max_layer_nm)
        
        # Penalty for total thickness
        total_t = np.sum(t)
        penalty = 0.0
        if total_t > max_total_nm:
            penalty = 10.0 * (total_t - max_total_nm)
        
        R = reflectance_spectrum(t)
        mean_R = np.mean(R)
        
        return mean_R + penalty

    # Initial guess: Quarter-wave at center wavelength
    lambda_c = np.sqrt(band_nm[0] * band_nm[1])
    t_init = np.array([lambda_c / (4.0 * n) for n in n_stack])
    
    # Ensure initial guess is valid
    t_init = np.clip(t_init, 0, max_layer_nm)
    if np.sum(t_init) > max_total_nm:
        t_init = t_init * (max_total_nm / np.sum(t_init))
        
    best_t = t_init
    best_score = objective(best_t)
    
    # Run Nelder-Mead
    # We might need to try a few times or use a better method.
    # Let's try one run with Nelder-Mead first.
    
    try:
        res = minimize(objective, t_init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success and res.fun < best_score:
            best_t = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline:
            pass # Skip further optimization
            
        # If we have time, try a second start with slightly perturbed initial guess
        if time.time() < deadline - 10:
            # Perturb initial guess
            np.random.seed(42)
            t_init2 = t_init * (1 + 0.1 * np.random.randn(n_layers))
            t_init2 = np.clip(t_init2, 0, max_layer_nm)
            if np.sum(t_init2) > max_total_nm:
                t_init2 = t_init2 * (max_total_nm / np.sum(t_init2))
                
            res2 = minimize(objective, t_init2, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res2.success and res2.fun < best_score:
                best_t = res2.x
                best_score = res2.fun
                
    except Exception as e:
        # If optimization fails, return initial guess
        pass
        
    # Final check on constraints for the best solution
    best_t = np.clip(best_t, 0, max_layer_nm)
    if np.sum(best_t) > max_total_nm:
        best_t = best_t * (max_total_nm / np.sum(best_t))
        
    # Return layers and thicknesses
    return layer_types, list(best_t)

