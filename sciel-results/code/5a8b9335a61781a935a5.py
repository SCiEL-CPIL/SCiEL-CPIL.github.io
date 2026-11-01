"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
import itertools

def construct():
    """
    Construct three coefficients (c0, c4, c8) for the Hermite combination.
    The degree-12 weight is fixed so that g(0) = 0.
    We want to minimize the upper bound on C4 = A(f)*A(Fourier(f)).
    
    Strategy:
    1. Define the test function g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2).
       H_n are physicists' Hermite polynomials? Or probabilists'?
       The task mentions "Fourier-invariant". Usually, for Fourier transform properties,
       probabilists' Hermite polynomials (He_n) or normalized physicists' (H_n) are used.
       Let's assume standard physicists' Hermite polynomials H_n(x) defined by:
       H_0 = 1
       H_1 = 2x
       H_2 = 4x^2 - 2
       H_3 = 8x^3 - 12x
       H_4 = 16x^4 - 48x^2 + 12
       H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
       H_12 = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 14457600x^4 - 14515200x^2 + 465120
        
       Wait, the task says "Fourier-invariant". The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}.
       So the magnitude is invariant.
       
       The quantity A(f) is likely the amplitude or L-infinity norm? Or maybe related to uncertainty principle?
       "Uncertainty inequality ... upper bound on A(f)*A(Fourier(f))"
       Usually uncertainty principle is ||x f|| * ||k F|| >= C.
       But here it says A(f)*A(Fourier(f)).
       Let's look at the "Prior state of the art" coefficients:
       c0 = -1.13, c4 = 0.04, c8 = 0.000308...
       
       Let's derive the objective function.
       If A(f) is the L^2 norm (energy), then by Parseval ||f||_2 = ||F||_2, so the product is constant.
       So A(f) must be something else.
       Perhaps A(f) = ||f||_infinity? Or max |f(x)|?
       Or maybe it's related to the "uncertainty" in a specific sense.
       
       Let's re-read: "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce the upper bound on A(f)*A(Fourier(f))."
       
       In many contexts, A(f) might denote the amplitude, i.e., sup_x |f(x)|.
       Let's assume A(f) = ||f||_infinity.
       Then we want to minimize ||g||_infinity * ||G||_infinity.
       Since g is real and even (Hermite polys of even degree are even, Gaussian is even), g is even.
       G(k) = Fourier transform. For real even functions, Fourier transform is real and even.
       Also, since g is a linear combination of H_n e^{-x^2/2}, G is (-i)^n times the same combination (with signs).
       Specifically, if g = sum c_n H_n e^{-x^2/2}, then G = sum c_n (-i)^n H_n e^{-k^2/2}.
       Since n=0,4,8,12 are even, (-i)^n = 1.
       So G(k) = g(k).
       Thus ||g||_infinity = ||G||_infinity.
       So we want to minimize ||g||_infinity^2.
       This means we want to make the peak of g as small as possible.
       Constraint: g(0) = 0.
       Also, we probably want to normalize the energy? Or is the scale fixed?
       The coefficients are free. If we scale all coefficients by lambda, ||g||_inf scales by lambda, so the product scales by lambda^2.
       We can make it arbitrarily small by scaling down.
       So there must be a normalization constraint.
       Usually, uncertainty principles involve ||x f|| or ||f||_2.
       If the product is A(f)*A(F), and A is sup norm, we need a constraint like ||f||_2 = 1.
       
       Let's assume: Minimize ||g||_infinity subject to ||g||_2 = 1 and g(0)=0.
       Then the score is ||g||_infinity (since product is ||g||_inf^2, minimizing the product is equivalent to minimizing the norm, or maybe the score IS the product? "score is the upper bound on C4". C4 might be the constant in the inequality?
       "reduce the upper bound on A(f)*A(Fourier(f))"
       If the inequality is A(f)A(F) <= C4 * (some fixed quantity), then we want to minimize the ratio.
       But if we can scale f, the ratio is scale-invariant.
       
       Let's look at the seed values again.
       c0 = -1.13, c4 = 0.04, c8 = 1/3240 ~ 0.0003.
       Let's compute g(0).
       H0(0)=1, H4(0)=12, H8(0)=3360, H12(0)=465120.
       g(0) = (c0*1 + c4*12 + c8*3360 + c12*465120) * 1.
       We need g(0)=0.
       So c12 = -(c0 + 12c4 + 3360c8) / 465120.
       
       Let's check the seed:
       c0 = -1.13
       c4 = 0.04
       c8 = 0.00030864
       c0 + 12c4 + 3360c8 = -1.13 + 0.48 + 1.037 = -0.613
       c12 = -(-0.613)/465120 = 0.613/465120 ~ 1.3e-6.
       
       Now, what is the objective?
       If we minimize ||g||_inf subject to ||g||_2=1, we are looking for the "flattest" function with unit energy, zero at origin.
       
       Let's implement this optimization.
       Variables: c0, c4, c8.
       c12 is determined by c0, c4, c8.
       g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^{-x^2/2}.
       
       We need to compute ||g||_2^2 and ||g||_inf.
       ||g||_2^2 = integral |g(x)|^2 dx.
       Since H_n are orthogonal with weight e^{-x^2}, 
       integral H_n H_m e^{-x^2} dx = sqrt(pi) 2^n n! delta_nm.
       Here we have e^{-x^2} in the integral (from |e^{-x^2/2}|^2 = e^{-x^2}).
       So ||g||_2^2 = sqrt(pi) * (c0^2 0! + c4^2 4! 2^4 + c8^2 8! 2^8 + c12^2 12! 2^12).
       Let's call the coefficients of the norm N0, N4, N8, N12.
       N0 = 1
       N4 = 24 * 16 = 384
       N8 = 40320 * 256 = 10,321,920
       N12 = 479001600 * 4096 = 1,962,000,000,000 approx.
       
       So ||g||_2^2 = sqrt(pi) * (c0^2 + 384 c4^2 + 10321920 c8^2 + 1.96e12 c12^2).
       
       We want to minimize ||g||_inf.
       ||g||_inf = max_x |g(x)|.
       Since g is even and smooth, the max occurs at some x >= 0.
       We can use scipy.optimize.minimize to minimize ||g||_inf subject to ||g||_2^2 = 1.
       
       However, ||g||_inf is not differentiable. We can use a smooth approximation or just evaluate on a grid and use the max.
       Alternatively, we can minimize the max on a dense grid.
       
       Let's try a direct optimization approach:
       1. Define g(x; c) using c0, c4, c8.
       2. Compute norm2(c).
       3. Normalize c so that norm2 = 1.
       4. Compute max |g(x)| on a grid.
       5. Minimize this max value.
       
       Since the problem is scale-invariant for the ratio, fixing norm2=1 is a good constraint.
       
       Let's use scipy.optimize.minimize with a grid-based objective.
       
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Keep a safety margin
    
    # Hermite polynomials (physicists')
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
    # H12 = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 14457600x^4 - 14515200x^2 + 465120
    
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 3360
    def H12(x): return 4096*x**12 - 98304*x**10 + 1032192*x**8 - 5806080*x**6 + 14457600*x**4 - 14515200*x**2 + 465120
    
    def get_c12(c0, c4, c8):
        # g(0) = c0*1 + c4*12 + c8*3360 + c12*465120 = 0
        return -(c0 + 12*c4 + 3360*c8) / 465120.0
    
    # Grid for evaluating max
    # The function decays as e^{-x^2/2}. Significant support is x in [-5, 5].
    # Since even, x in [0, 5].
    x_grid = np.linspace(0, 5, 1000)
    gauss = np.exp(-x_grid**2 / 2)
    H0_grid = H0(x_grid)
    H4_grid = H4(x_grid)
    H8_grid = H8(x_grid)
    H12_grid = H12(x_grid)
    
    # Norm coefficients
    N0 = 1.0
    N4 = 384.0
    N8 = 10321920.0
    N12 = 1.9620000000000000e12  # 12! * 2^12 = 479001600 * 4096
    
    def norm2(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        return np.sqrt(np.pi) * (N0*c0**2 + N4*c4**2 + N8*c8**2 + N12*c12**2)
    
    def objective(params):
        c0, c4, c8 = params
        n2 = norm2(c0, c4, c8)
        if n2 == 0:
            return 1e10
        scale = 1.0 / np.sqrt(n2)
        c0_n, c4_n, c8_n = c0*scale, c4*scale, c8*scale
        c12_n = get_c12(c0_n, c4_n, c8_n)
        
        g_vals = (c0_n*H0_grid + c4_n*H4_grid + c8_n*H8_grid + c12_n*H12_grid) * gauss
        return np.max(np.abs(g_vals))
    
    # Initial guess: use the seed values
    c0_init, c4_init, c8_init = -1.13, 0.04, 1/3240.0
    x0 = np.array([c0_init, c4_init, c8_init])
    
    best_params = x0
    best_obj = objective(x0)
    
    # Try a few random starts and local optimization
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts:
        if time.time() > deadline - 10:
            break
            
        if attempts == 0:
            x_start = x0
        else:
            # Random perturbation
            x_start = x0 + rng.normal(0, 0.1, size=3)
            
        res = minimize(objective, x_start, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
            
        attempts += 1
        
    # Return the best coefficients found
    # Ensure they are floats
    return np.array([best_params[0], best_params[1], best_params[2]])

# EVOLVE-BLOCK-END
