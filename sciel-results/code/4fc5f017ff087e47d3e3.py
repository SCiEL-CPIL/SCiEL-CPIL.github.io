import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simpson

def construct():
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H12 = 4096 * x**12 - 204800 * x**10 + 3225600 * x**8 - 18662400 * x**6 + 40840320 * x**4 - 39916800 * x**2 + 10395000
    
    # Derivatives of Hermite polynomials
    # H_n'(x) = 2n H_{n-1}(x)
    # H4'(x) = 8 H3(x) = 8(8x^3 - 12x) = 64x^3 - 96x
    # H8'(x) = 16 H7(x) = 16(128x^7 - 1344x^5 + 3360x^3 - 1680x) = 2048x^7 - 21504x^5 + 53760x^3 - 26880x
    # H12'(x) = 24 H11(x)
    # H11(x) = 2048x^11 - 77824x^9 + 896000x^7 - 3326400x^5 + 4804800x^3 - 2217600x
    # H12'(x) = 24(2048x^11 - ...) = 49152x^11 - 1867776x^9 + 21504000x^7 - 79833600x^5 + 115315200x^3 - 53222400x
    
    H4_prime = 64 * x**3 - 96 * x
    H8_prime = 2048 * x**7 - 21504 * x**5 + 53760 * x**3 - 26880 * x
    H12_prime = 49152 * x**11 - 1867776 * x**9 + 21504000 * x**7 - 79833600 * x**5 + 115315200 * x**3 - 53222400 * x
    
    gauss = np.exp(-x**2 / 2)
    
    # Constants for constraint P(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 10395000.0
    
    def objective(params):
        a0, a4, a8 = params
        # Correct constraint: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        g = P * gauss
        
        # Analytical derivative
        P_prime = a0 * 0 + a4 * H4_prime + a8 * H8_prime + a12 * H12_prime
        g_prime = (P_prime - x * P) * gauss
        
        g2 = g * g
        gp2 = g_prime * g_prime
        
        norm2 = simpson(g2, x=x)
        if norm2 < 1e-15:
            return 1e10
            
        x2g2 = simpson(x * x * g2, x=x)
        xg2 = simpson(x * g2, x=x)
        
        mean_x = xg2 / norm2
        var_x = x2g2 / norm2 - mean_x**2
        
        gp2_sum = simpson(gp2, x=x)
        
        if var_x < 0 or gp2_sum < 0:
            return 1e10
            
        return np.sqrt(var_x * gp2_sum / norm2)
    
    # Initial guess
    x0 = np.array([1.0, 0.1, 0.01])
    
    # Use Powell method for robustness
    result = minimize(objective, x0, method='Powell', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    if result.fun < objective(x0):
        return np.array(result.x)
    return x0
