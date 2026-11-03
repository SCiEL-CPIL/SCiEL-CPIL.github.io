"""
Optimized program using Basin-Hopping on a scale-invariant 2D parameterization.
The problem has 3 coefficients (c0, c4, c8) but the objective is scale-invariant.
We parameterize by ratios (u = c4/c0, v = c8/c0) and optimize on the 2D plane.
We use Basin-Hopping to navigate the non-convex landscape, combining global
moves with local L-BFGS-B refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    """
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    q = np.convolve(p, p)
    
    # Integrate x^{2k} e^{-x^2}
    factors = np.zeros(13)
    for k in range(13):
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
        
    x2 = I[1] / I[0]
    
    # Calculate R(x) for Fourier side
    # R(x) is derived from the Fourier transform of g(x) = P(x)e^{-x^2/2}
    # The code below calculates the coefficients for the numerator of the Fourier side variance
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        if j >= 1 and j <= 7:
            val += p[j-1]
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    s = np.convolve(r, r)
    
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    product = math.sqrt(x2 * xi2)
    
    return product

def _objective_2d(z):
    """
    Objective function for 2D optimization.
    z = [u, v] where u = c4/c0, v = c8/c0.
    We set c0 = 1.
    """
    u, v = z
    try:
        # We need to handle the case where the resulting polynomial is invalid or norm is 0
        val = _compute_bound(1.0, u, v)
        if val > 1e9:
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin for 180s limit
    
    # Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # Initial guesses for 2D optimization (u, v)
    # Based on seed: c0 ~ -1.13, c4 ~ 0.04, c8 ~ 0.0003
    # If we scale by c0 (negative), u = c4/c0 is negative, v = c8/c0 is negative.
    # Let's try a few starting points.
    
    start_points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [seed_coeffs[1]/seed_coeffs[0], seed_coeffs[2]/seed_coeffs[0]]
    ]
    
    # Filter out invalid or high-value starts
    valid_starts = []
    for sp in start_points:
        val = _objective_2d(sp)
        if val < 1e9:
            valid_starts.append(np.array(sp))
            
    if not valid_starts:
        valid_starts = [np.array([0.0, 0.0])]
        
    # Run Basin-Hopping from the best starting point
    # We pick the best starting point among the candidates
    best_start = None
    best_start_val = 1e10
    for sp in valid_starts:
        val = _objective_2d(sp)
        if val < best_start_val:
            best_start_val = val
            best_start = sp
            
    if best_start_val < best_val:
        best_val = best_start_val
        best_coeffs = np.array([1.0, best_start[0], best_start[1]])
        
    try:
        # Basin-Hopping parameters
        # min_steps: number of local minimization steps
        # stepsize: size of random perturbation
        # niter: number of basin-hopping iterations
        
        res = scipy.optimize.basinhopping(
            _objective_2d, 
            x0=best_start, 
            stepsize=0.5,
            minsteps=5,
            maxfun=2000,
            niter=50,
            seed=42,
            take_step=True,
            accept_test=None
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = np.array([1.0, res.x[0], res.x[1]])
            
    except Exception:
        pass

    # Check time and try a second run with different parameters if time permits
    if time.time() < deadline - 30.0:
        try:
            # Try a larger step size and more iterations
            res2 = scipy.optimize.basinhopping(
                _objective_2d, 
                x0=best_start, 
                stepsize=1.0,
                minsteps=3,
                maxfun=1000,
                niter=30,
                seed=123
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_coeffs = np.array([1.0, res2.x[0], res2.x[1]])
        except Exception:
            pass
            
    # Final check: if we found a better value, ensure it's valid
    # We should also check if the seed is still better (in case optimization failed)
    seed_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    if seed_val < best_val:
        return seed_coeffs
    else:
        return best_coeffs

# EVOLVE-BLOCK-END
