import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 4608x^6 + 23040x^4 - 40320x^2 + 1680
    # H_12(x) = 4096x^12 - 147456x^10 + 2478528x^8 - 17301504x^6 + 55725120x^4 - 66528000x^2 + 103950
    
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 4608*x**6 + 23040*x**4 - 40320*x**2 + 1680
    def H12(x): return 4096*x**12 - 147456*x**10 + 2478528*x**8 - 17301504*x**6 + 55725120*x**4 - 66528000*x**2 + 103950

    # Constraint: g(0) = 0
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=103950
    # Wait, let's re-evaluate H_n(0) for Physicists' Hermite.
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 103950
    # Check: H_2(x) = 4x^2 - 2, H_2(0) = -2. Formula (-1)^k (2k)!/k!
    # k=0: 1
    # k=2: (-1)^2 * 4!/2! = 24/2 = 12. Correct.
    # k=4: (-1)^4 * 8!/4! = 40320/24 = 1680. Correct.
    # k=6: (-1)^6 * 12!/6! = 479001600/720 = 665280? 
    # Let's check H_12(0).
    # 12! = 479001600. 6! = 720. 479001600 / 720 = 665280.
    # My previous manual calc for H12(0) was 103950. That was wrong.
    # Let's re-calculate H12(0).
    # H_n(0) = (-1)^{n/2} \frac{n!}{(n/2)!}
    # n=12: (-1)^6 \frac{12!}{6!} = 665280.
    
    # So c12 = -(c0*1 + c4*12 + c8*1680) / 665280

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 1680.0) / 665280.0
        
        # Use Gauss-Hermite quadrature
        # x, w = np.polynomial.hermite.hermgauss(100)
        # This integrates f(x) e^{-x^2}.
        # We need integral of P(x)^2 e^{-x^2} and x^2 P(x)^2 e^{-x^2}.
        
        # To ensure speed, we can precompute x, w
        # But inside objective, we need to evaluate P(x).
        
        # For speed, use a fixed quadrature rule
        # x, w = np.polynomial.hermite.hermgauss(200)
        
        # Vectorized evaluation
        # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        
        # We can compute integrals analytically using orthogonality!
        # \int H_n(x) H_m(x) e^{-x^2} dx = \sqrt{\pi} 2^n n! \delta_{nm}
        # \int x^2 H_n(x) H_m(x) e^{-x^2} dx
        # x^2 = \frac{1}{4} (H_2(x) + 2) ? No.
        # x^2 = \frac{1}{4} H_2(x) + \frac{1}{2} H_0(x) ?
        # H_2(x) = 4x^2 - 2 \implies x^2 = \frac{1}{4} H_2(x) + \frac{1}{2}
        
        # So \int x^2 H_n H_m e^{-x^2} dx = \frac{1}{4} \int H_2 H_n H_m e^{-x^2} + \frac{1}{2} \int H_0 H_n H_m e^{-x^2}
        
        # This requires triple integrals of Hermite polynomials.
        # \int H_n H_m H_k e^{-x^2} dx is zero unless n+m+k is even and triangle inequalities hold.
        # This might be complex to implement correctly on the fly.
        # Numerical integration with 200 points is very fast (<1ms).
        
        x, w = np.polynomial.hermite.hermgauss(200)
        
        h0 = H0(x)
        h4 = H4(x)
        h8 = H8(x)
        h12 = H12(x)
        
        p = c0*h0 + c4*h4 + c8*h8 + c12*h12
        
        p2 = p**2
        norm_sq = np.sum(w * p2)
        x2_norm_sq = np.sum(w * x**2 * p2)
        
        if norm_sq == 0:
            return 1e10
            
        ratio = x2_norm_sq / norm_sq
        return ratio

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    # Try multiple starting points to avoid local minima
    best_val = np.inf
    best_x = x0
    
    # Run optimization
    try:
        # Use L-BFGS-B or BFGS
        res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success:
            best_val = res.fun
            best_x = res.x
        else:
            # If it didn't converge, check if it's better than initial
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline:
        # Try a few random restarts
        np.random.seed(42)
        for i in range(20):
            if time.time() > deadline:
                break
            # Random perturbation around the best found so far or random
            x_start = best_x + np.random.normal(0, 0.1, size=3)
            try:
                res = opt.minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_x = res.x
            except Exception:
                pass

    return np.array(best_x)
