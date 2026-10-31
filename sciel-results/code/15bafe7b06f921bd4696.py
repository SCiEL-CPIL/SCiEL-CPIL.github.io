"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad
import math

def construct():
    # Precompute constants
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Degrees involved
    degrees = [0, 4, 8, 12]
    
    # Helper to compute L2 norm squared analytically
    # Integral of (sum c_k H_k(x))^2 e^{-x^2} dx
    # = sum c_k^2 sqrt(pi) 2^k k!
    def compute_norm2_sq(c):
        # c is list of 4 coeffs [c0, c4, c8, c12]
        norm_sq = 0.0
        for i, k in enumerate(degrees):
            norm_sq += c[i]**2 * math.sqrt(math.pi) * (2**k) * math.factorial(k)
        return norm_sq

    # Helper to compute L4 norm numerically
    # Integral of |(sum c_k H_k(x)) e^{-x^2/2}|^4 dx
    # = Integral (sum c_k H_k(x))^4 e^{-2x^2} dx
    def compute_norm4(c):
        def integrand(x):
            val = 0.0
            for i, k in enumerate(degrees):
                val += c[i] * eval_hermite(k, x)
            return (val**4) * np.exp(-2*x**2)
        
        # Integrate over a range where the function is negligible
        # e^{-2x^2} decays fast. 10-15 sigma is enough.
        res, err = quad(integrand, -15, 15, limit=100)
        return res

    # Helper to compute FT L4 norm numerically
    # hat_g(omega) = sqrt(2*pi) * sum c_k (-i)^k H_k(omega) e^{-omega^2/2}
    # |hat_g(omega)|^4 = (2*pi)^2 * |sum c_k (-i)^k H_k(omega) e^{-omega^2/2}|^4
    def compute_norm4_hat(c):
        def integrand(w):
            val = 0.0
            for i, k in enumerate(degrees):
                # (-i)^k
                if k % 4 == 0:
                    factor = 1.0
                elif k % 4 == 1:
                    factor = -1j
                elif k % 4 == 2:
                    factor = -1.0
                else:
                    factor = 1j
                
                val += c[i] * factor * eval_hermite(k, w)
            
            # val is sum c_k (-i)^k H_k(w)
            # hat_g(w) = sqrt(2*pi) * val * exp(-w^2/2)
            # |hat_g(w)|^4 = (2*pi)^2 * |val|^4 * exp(-2w^2)
            return (2*math.pi)**2 * (abs(val)**4) * np.exp(-2*w**2)
        
        res, err = quad(integrand, -15, 15, limit=100)
        return res

    def objective(params):
        try:
            s0, m0, s4, m4, s8, m8 = params
            c0 = s0 * np.exp(m0)
            c4 = s4 * np.exp(m4)
            c8 = s8 * np.exp(m8)
            
            # Enforce g(0) = 0
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            c = [c0, c4, c8, c12]
            
            norm2_sq = compute_norm2_sq(c)
            if norm2_sq < 1e-10:
                return 1e10
            
            norm4_val = compute_norm4(c)
            norm4_val = norm4_val ** 0.25
            
            # Parseval: ||hat_g||_2^2 = 2*pi * ||g||_2^2
            norm2_hat_sq = 2 * math.pi * norm2_sq
            
            norm4_hat_val = compute_norm4_hat(c)
            norm4_hat_val = norm4_hat_val ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10

    # Multi-start optimization
    best_score = 1e10
    best_params = None
    
    # Seed initial guess
    c0_seed = -1.13
    c4_seed = 0.04
    c8_seed = 1.0 / 3240.0
    
    starts = []
    
    # Add seed
    starts.append([np.sign(c0_seed), np.log(np.abs(c0_seed)), 
                   np.sign(c4_seed), np.log(np.abs(c4_seed)), 
                   np.sign(c8_seed), np.log(np.abs(c8_seed))])
    
    # Add random restarts
    np.random.seed(42)
    for _ in range(20):
        # Random log-magnitudes in range [-5, 5]
        m0 = np.random.uniform(-5, 5)
        m4 = np.random.uniform(-5, 5)
        m8 = np.random.uniform(-5, 5)
        
        # Random signs
        s0 = np.random.choice([-1, 1])
        s4 = np.random.choice([-1, 1])
        s8 = np.random.choice([-1, 1])
        
        starts.append([s0, m0, s4, m4, s8, m8])
    
    for x0 in starts:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    if best_params is not None:
        s0, m0, s4, m4, s8, m8 = best_params
        return np.array([s0 * np.exp(m0), s4 * np.exp(m4), s8 * np.exp(m8)])
    else:
        return np.array([c0_seed, c4_seed, c8_seed])

# EVOLVE-BLOCK-END
