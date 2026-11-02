"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial, sqrt, pi

def construct():
    start_time = time.time()
    deadline = 210.0
    
    # Degrees present in the sum
    degrees = [0, 4, 8, 12]
    N = len(degrees)
    
    # Precompute norms ||psi_k||^2 = sqrt(pi) * 2^k * k!
    norms_sq = np.zeros(N)
    for idx, k in enumerate(degrees):
        norms_sq[idx] = sqrt(pi) * (2 ** k) * factorial(k)
        
    # Precompute matrix M where M[i,j] = <psi_i, x^2 psi_j>
    # x^2 psi_k = A_k A_{k-1} psi_{k-2} + (A_k B_{k-1} + B_k A_{k+1}) psi_k + B_k B_{k+1} psi_{k+2}
    # where A_k = sqrt(k/2), B_k = sqrt((k+1)/2)
    # Note: A_0 = 0.
    
    # We need <psi_i, psi_j> = norms_sq[j] * delta_ij
    # So <psi_i, x^2 psi_j> is non-zero only if i is in {j-2, j, j+2}
    # and equals the coefficient of psi_i in the expansion of x^2 psi_j multiplied by norms_sq[i].
    
    M = np.zeros((N, N))
    
    # Helper to get index of degree k in degrees list, or -1 if not found
    deg_to_idx = {d: i for i, d in enumerate(degrees)}
    
    for j_idx, k in enumerate(degrees):
        # Coeffs for x^2 psi_k
        A_k = sqrt(k / 2.0)
        B_k = sqrt((k + 1) / 2.0)
        
        # Term psi_{k-2}
        if k >= 2:
            A_km1 = sqrt((k - 1) / 2.0) # Wait, formula was A_k A_{k-1}
            # A_{k-1} = sqrt((k-1)/2)
            coeff_km2 = A_k * A_km1
            target_deg = k - 2
            if target_deg in deg_to_idx:
                i_idx = deg_to_idx[target_deg]
                M[i_idx, j_idx] = coeff_km2 * norms_sq[i_idx]
                M[j_idx, i_idx] = M[i_idx, j_idx] # Symmetric
                
        # Term psi_k
        # Coeff is A_k B_{k-1} + B_k A_{k+1}
        # A_{k-1} = sqrt((k-1)/2)
        # A_{k+1} = sqrt((k+1)/2) = B_k
        # B_{k-1} = sqrt(k/2) = A_k
        # So coeff = A_k * A_k + B_k * B_k = k/2 + (k+1)/2 = k + 0.5
        coeff_k = k + 0.5
        M[j_idx, j_idx] = coeff_k * norms_sq[j_idx]
        
        # Term psi_{k+2}
        B_kp1 = sqrt((k + 2) / 2.0) # B_{k+1}
        coeff_kp2 = B_k * B_kp1
        target_deg = k + 2
        if target_deg in deg_to_idx:
            i_idx = deg_to_idx[target_deg]
            M[i_idx, j_idx] = coeff_kp2 * norms_sq[i_idx]
            M[j_idx, i_idx] = M[i_idx, j_idx]
            
    # H_n(0) values
    def H_n_zero(n):
        if n % 2 != 0:
            return 0.0
        k = n // 2
        return ((-1) ** k) * (factorial(n) / factorial(k))
        
    H_vals = np.array([H_n_zero(d) for d in degrees])
    
    # Constraint: sum c_i H_i(0) = 0
    # c0*H0 + c4*H4 + c8*H8 + c12*H12 = 0
    # c12 = -(c0*H0 + c4*H4 + c8*H8) / H12
    
    H0_0, H4_0, H8_0, H12_0 = H_vals
    
    def get_full_coeffs(c0, c4, c8):
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        return np.array([c0, c4, c8, c12])
        
    def objective(params):
        c0, c4, c8 = params
        c = get_full_coeffs(c0, c4, c8)
        
        N0 = np.dot(c, norms_sq * c)
        if N0 <= 1e-10:
            return 1e10
        N2 = np.dot(c, M.dot(c))
        
        return N2 / N0

    # Fix c0 = 1 to remove scaling invariance
    def objective_2d(params):
        c4, c8 = params
        c0 = 1.0
        return objective([c0, c4, c8])

    # Seed values
    seed_c0 = -113 / 100
    seed_c4 = 1 / 25
    seed_c8 = 1 / 3240
    
    # Normalize seed to c0=1
    x0 = np.array([seed_c4 / seed_c0, seed_c8 / seed_c0])
    
    best_score = objective_2d(x0)
    best_params = x0
    
    # Optimize
    try:
        res = minimize(objective_2d, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
        
    if time.time() < deadline - 10:
        # Try a second run from a different point
        x0_alt = np.array([0.0, 0.0])
        try:
            res2 = minimize(objective_2d, x0_alt, method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    c4_opt, c8_opt = best_params
    c0_opt = 1.0
    
    return np.array([c0_opt, c4_opt, c8_opt])

# EVOLVE-BLOCK-END
