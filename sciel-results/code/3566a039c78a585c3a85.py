import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

# Hermite Polynomials (Physicists')
# H_n(x)
def H0(x):
    return 1.0

def H4(x):
    return 16*x**4 - 48*x**2 + 12

def H8(x):
    return 256*x**8 - 3072*x**6 + 11520*x**4 - 11520*x**2 + 1680

def H12(x):
    return 4096*x**12 - 73728*x**10 + 483840*x**8 - 1307520*x**6 + 1548480*x**4 - 665280*x**2 + 103968

# Values at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 103968.0

def get_c12(c0, c4, c8):
    return -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0

def g_func(x, c0, c4, c8):
    c12 = get_c12(c0, c4, c8)
    return (c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)) * np.exp(-x**2 / 2.0)

def objective(params):
    c0, c4, c8 = params
    # Integrate |g(x)| from -inf to inf
    # Since g is even, integrate 0 to inf and multiply by 2
    # Use quad with limits [0, 20] which is effectively infinity for exp(-x^2/2)
    try:
        val, err = quad(lambda x: abs(g_func(x, c0, c4, c8)), 0, 20, limit=100)
        return 2 * val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin
    
    # Seed values
    c0_seed = -113 / 100
    c4_seed = 1 / 25
    c8_seed = 1 / 3240
    x0 = np.array([c0_seed, c4_seed, c8_seed])
    
    best_score = objective(x0)
    best_params = x0
    
    # Try local optimization
    # Nelder-Mead is robust for small dimensions
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000, 'maxfev': 10000})
        if res.success and res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline:
            return np.array(best_params)
            
        # Try BFGS for further refinement if time permits
        # We need a gradient for BFGS. 
        # grad_i = 2 * int_0^inf sgn(g(x)) H_i(x) exp(-x^2/2) dx  (for i in 0,4,8)
        # Note: c12 depends on c0,c4,c8, so the derivative includes the chain rule for c12.
        # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12(c) H12) e^{-x^2/2}
        # d/dc_i g(x) = (H_i(x) + H12(x) * dc12/dc_i) e^{-x^2/2}
        
        def gradient(params):
            c0, c4, c8 = params
            c12 = get_c12(c0, c4, c8)
            
            # dc12/dc0 = -1/H12_0
            # dc12/dc4 = -H4_0/H12_0
            # dc12/dc8 = -H8_0/H12_0
            dc12_dc0 = -1.0 / H12_0
            dc12_dc4 = -H4_0 / H12_0
            dc12_dc8 = -H8_0 / H12_0
            
            def integrand(x):
                val = g_func(x, c0, c4, c8)
                sgn = np.sign(val)
                if sgn == 0:
                    sgn = 1.0 # Arbitrary, measure zero
                base = sgn * np.exp(-x**2 / 2.0)
                d_c0 = (H0(x) + H12(x) * dc12_dc0) * base
                d_c4 = (H4(x) + H12(x) * dc12_dc4) * base
                d_c8 = (H8(x) + H12(x) * dc12_dc8) * base
                return [d_c0, d_c4, d_c8]
            
            # Integrate each component
            try:
                g0, _ = quad(lambda x: integrand(x)[0], 0, 20, limit=100)
                g4, _ = quad(lambda x: integrand(x)[1], 0, 20, limit=100)
                g8, _ = quad(lambda x: integrand(x)[2], 0, 20, limit=100)
                return np.array([2*g0, 2*g4, 2*g8])
            except Exception:
                return None

        if time.time() < deadline - 10:
            res2 = minimize(objective, best_params, jac=gradient, method='BFGS', options={'gtol': 1e-10, 'maxiter': 1000})
            if res2.success and res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
                
    except Exception as e:
        pass

    return np.array(best_params)
