import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import math
import time

def _compute_matrices():
    """Pre-compute the necessary moment matrices for the Hermite-Gaussian basis."""
    # Degrees of Hermite polynomials
    degs = [0, 4, 8, 12]
    n = len(degs)
    
    # N_k = integral H_k^2 exp(-x^2) dx = sqrt(pi) * 2^k * k!
    N = np.zeros(n)
    for i, k in enumerate(degs):
        N[i] = math.sqrt(math.pi) * (2 ** k) * math.factorial(k)
        
    # M_spatial: diagonal matrix of <x^2> for each H_k
    # <x^2>_{H_k} = (k + 0.5) * N_k
    M_spatial = np.zeros((n, n))
    for i, k in enumerate(degs):
        M_spatial[i, i] = (k + 0.5) * N[i]
        
    # M_fourier: matrix of integrals <xi^2 H_k H_j> for the Fourier side
    # We need to compute I_kj = int xi^2 H_k(xi) H_j(xi) exp(-xi^2) dxi
    # This is non-zero for j = k, k+2, k-2
    M_fourier = np.zeros((n, n))
    
    # Use numerical integration for robustness with specific degrees
    # The functions are smooth and decay fast
    for i, k in enumerate(degs):
        for j, l in enumerate(degs):
            # Only non-zero if abs(k-l) is 0 or 2
            if abs(k - l) in [0, 2]:
                # Define integrand
                def integrand(x):
                    # eval_hermite is the physicists' Hermite polynomial
                    hk = eval_hermite(k, x)
                    hl = eval_hermite(l, x)
                    return x**2 * hk * hl * np.exp(-x**2)
                
                # Integrate from -inf to inf
                # Using quad
                from scipy.integrate import quad
                val, err = quad(integrand, -np.inf, np.inf, limit=50)
                M_fourier[i, j] = val
            else:
                M_fourier[i, j] = 0.0
                
    return degs, N, M_spatial, M_fourier

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Pre-compute matrices
    degs, N, M_spatial, M_fourier = _compute_matrices()
    
    # Constraint: g(0) = 0
    # H_0(0)=1, H_4(0)=-3, H_8(0)=105, H_12(0)=-10395
    # c0*1 + c4*(-3) + c8*105 + c12*(-10395) = 0
    # c12 = (c0 - 3c4 + 105c8) / 10395
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = (c0 - 3.0 * c4 + 105.0 * c8) / 10395.0
        c = np.array([c0, c4, c8, c12])
        
        # Energy E = c^T N c
        # Since N is diagonal, E = sum c_i^2 N_i
        E = np.dot(c, c * N)
        
        if E < 1e-10:
            return 1e10 # Avoid division by zero
            
        # Spatial variance
        # E_x = c^T M_spatial c
        E_x = np.dot(c, M_spatial.dot(c))
        sigma_x_sq = E_x / E
        
        # Fourier side
        # Q(xi) = sum c_k i^k H_k(xi)
        # |Q|^2 = sum_{k,j} c_k c_j i^{j-k} H_k H_j
        # E_ft = sum_{k,j} c_k c_j i^{j-k} M_fourier[k,j]
        
        # Construct the complex matrix M_eff where M_eff[k,j] = i^{j-k} M_fourier[k,j]
        # Then E_ft = c^T M_eff c ?
        # Note: c is real.
        # sum_{k,j} c_k c_j (i^{j-k}) M_fourier[k,j]
        
        E_ft = 0.0
        for k_idx, k_deg in enumerate(degs):
            for j_idx, j_deg in enumerate(degs):
                # i^(j-k)
                # j-k can be -8, -4, 0, 4, 8 etc.
                # i^0 = 1
                # i^4 = 1
                # i^8 = 1
                # i^-4 = 1
                # i^-8 = 1
                # In this specific case, all degrees are multiples of 4.
                # So i^(j-k) is always 1!
                # Let's verify: degs are 0, 4, 8, 12.
                # Differences are multiples of 4.
                # i^4 = 1.
                # So the phase factor is always 1.
                
                E_ft += c[k_idx] * c[j_idx] * M_fourier[k_idx, j_idx]
                
        sigma_xi_sq = E_ft / E
        
        # Uncertainty product
        unc = np.sqrt(sigma_x_sq * sigma_xi_sq)
        return unc

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.000308])
    
    # Minimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        if time.time() > deadline - 10:
             # If close to deadline, just return best found so far or seed
             # But minimize usually returns quickly
             pass
             
        if res.success:
            return np.array(res.x)
        else:
            # If optimization fails, return seed or best point found
            # res.x might still be better than x0
            return np.array(res.x)
            
    except Exception as e:
        # Fallback to seed values
        return np.array([-1.13, 0.04, 0.000308])
