import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """
    Design a stack of layers to minimize mean reflectance.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin
    
    # Extract parameters
    n_amb = problem.get("ambient_index", 1.0)
    n_sub = problem.get("substrate_index", 1.52)
    materials = problem["materials"]
    n_L = materials["L"]
    n_H = materials["H"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Wavelengths for evaluation (401 points)
    wl_min, wl_max = band_nm
    wavelengths = np.linspace(wl_min, wl_max, 401)
    
    def calc_mean_reflectance(thicknesses, sequence):
        """
        Calculate mean reflectance in percent for a given set of thicknesses and material sequence.
        thicknesses: list of floats (nm)
        sequence: list of 'L' or 'H'
        """
        N = len(sequence)
        if N == 0:
            # Bare substrate
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * abs(r)**2
        
        # Precompute delta and Y for each layer at each wavelength
        # deltas: shape (N, len(wavelengths))
        # Ys: shape (N,)
        
        # Check constraints
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return 1e6
        if sum(thicknesses) > max_total_nm:
            return 1e6
            
        d = np.array(thicknesses)
        ns = np.array([n_L if s == 'L' else n_H for s in sequence])
        
        # Vectorized calculation for all wavelengths
        # delta_j = 2 * pi * n_j * d_j / lambda
        # lambda shape (W,), d shape (N,) -> broadcast to (N, W)
        deltas = 2.0 * np.pi * np.outer(ns, d) / wavelengths[:, np.newaxis]
        
        # Initialize total matrix M
        # M starts as identity
        M = np.ones((N, 2, 2, len(wavelengths)), dtype=complex)
        
        for i in range(N):
            # M_i = [[cos(d_i), i*sin(d_i)/n_i], [i*n_i*sin(d_i), cos(d_i)]]
            cos_d = np.cos(deltas[i])
            sin_d = np.sin(deltas[i])
            n_i = ns[i]
            
            M_i = np.zeros((2, 2, len(wavelengths)), dtype=complex)
            M_i[0, 0] = cos_d
            M_i[0, 1] = 1j * sin_d / n_i
            M_i[1, 0] = 1j * n_i * sin_d
            M_i[1, 1] = cos_d
            
            # M = M_i @ M_prev
            # Matrix multiplication for each wavelength
            # M[..., 0, 0] = M_i[0,0]*M[0,0] + M_i[0,1]*M[1,0]
            # etc.
            # Using np.einsum might be cleaner, but explicit loops over 2x2 elements are fast enough for W=401, N=20
            M_new = np.zeros_like(M)
            M_new[..., 0, 0] = M_i[0, 0] * M[..., 0, 0] + M_i[0, 1] * M[..., 1, 0]
            M_new[..., 0, 1] = M_i[0, 0] * M[..., 0, 1] + M_i[0, 1] * M[..., 1, 1]
            M_new[..., 1, 0] = M_i[1, 0] * M[..., 0, 0] + M_i[1, 1] * M[..., 1, 0]
            M_new[..., 1, 1] = M_i[1, 0] * M[..., 0, 1] + M_i[1, 1] * M[..., 1, 1]
            M = M_new
            
        m11 = M[..., 0, 0]
        m12 = M[..., 0, 1]
        m21 = M[..., 1, 0]
        m22 = M[..., 1, 1]
        
        # Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
        Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
        
        # R = | (Y_in - n_amb) / (Y_in + n_amb) |^2
        r = (Y_in - n_amb) / (Y_in + n_amb)
        R = np.abs(r)**2
        mean_R = np.mean(R)
        return 100.0 * mean_R

    # Candidate sequences
    # We will try a few different numbers of layers and starting materials
    # Quarter-wave thicknesses for center wavelength 600nm
    lam_c = 0.5 * (wl_min + wl_max)
    d_L_qw = lam_c / (4.0 * n_L)
    d_H_qw = lam_c / (4.0 * n_H)
    
    best_score = 1e6
    best_result = None
    
    # Define a set of candidate structures
    # (sequence, initial_thicknesses)
    candidates = []
    
    # 2 layers
    candidates.append((['L', 'H'], [d_L_qw, d_H_qw]))
    candidates.append((['H', 'L'], [d_H_qw, d_L_qw]))
    
    # 4 layers
    candidates.append((['L', 'H', 'L', 'H'], [d_L_qw, d_H_qw, d_L_qw, d_H_qw]))
    candidates.append((['H', 'L', 'H', 'L'], [d_H_qw, d_L_qw, d_H_qw, d_L_qw]))
    
    # 6 layers
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H'], [d_L_qw, d_H_qw, d_L_qw, d_H_qw, d_L_qw, d_H_qw]))
    candidates.append((['H', 'L', 'H', 'L', 'H', 'L'], [d_H_qw, d_L_qw, d_H_qw, d_L_qw, d_H_qw, d_L_qw]))
    
    # 8 layers
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'], [d_L_qw, d_H_qw, d_L_qw, d_H_qw, d_L_qw, d_H_qw, d_L_qw, d_H_qw]))
    candidates.append((['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'], [d_H_qw, d_L_qw, d_H_qw, d_L_qw, d_H_qw, d_L_qw, d_H_qw, d_L_qw]))

    for seq, init_thick in candidates:
        if time.time() > deadline:
            break
            
        # Define objective function for optimizer
        def objective(x):
            # x is array of thicknesses
            # Ensure bounds are respected during optimization? 
            # Nelder-Mead doesn't handle bounds well. 
            # We can penalize violations in the objective.
            # Or use 'L-BFGS-B' if we had gradients, but we don't.
            # Let's just penalize.
            if any(x < 0):
                return 1e6
            if any(x > max_layer_nm):
                return 1e6
            if np.sum(x) > max_total_nm:
                return 1e6
            return calc_mean_reflectance(x, seq)
        
        # Initial guess
        x0 = np.array(init_thick)
        
        # Run optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success or res.fun < best_score:
                # Check final validity
                x_opt = res.x
                if all(0 <= t <= max_layer_nm for t in x_opt) and sum(x_opt) <= max_total_nm:
                    score = res.fun
                    if score < best_score:
                        best_score = score
                        best_result = (seq, x_opt)
        except Exception:
            pass

    if best_result is not None:
        seq, thick = best_result
        # Convert to lists of strings and floats
        return seq, list(thick)
    else:
        # Fallback: single quarter wave L layer
        return ["L"], [d_L_qw]

