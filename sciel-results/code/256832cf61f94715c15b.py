import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return poly(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    def compute_uncertainty_product_exact(c0, c4, c8):
        """
        Compute A(f) * A(Fourier(f)) exactly using polynomial expansion.
        
        f(x) = P(x) * exp(-x^2/2) where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        
        We need:
        - I0 = ∫|f(x)|^2 dx
        - I2 = ∫x^2|f(x)|^2 dx  
        - J0 = ∫|F̂f(ξ)|^2 dξ = I0 (Parseval, with our convention)
        - J2 = ∫ξ^2|F̂f(ξ)|^2 dξ
        
        For J2, we use: ∫ξ^2|F̂f(ξ)|^2 dξ = (1/(2π)^2) ∫|f''(x)|^2 dx
        
        So A(f)*A(F̂f) = sqrt(I2/I0 * (1/(2π)^2) * ∫|f''|^2dx / I0)
                        = (1/(2π)) * sqrt(I2 * ∫|f''|^2dx / I0^2)
        """
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # We need ∫P(x)^2 * e^{-x^2} dx, ∫x^2*P(x)^2*e^{-x^2}dx, ∫(P''(x) + P'(x))^2 * e^{-x^2} dx
        # Wait, f(x) = P(x)*e^{-x^2/2}, so f'(x) = P'(x)*e^{-x^2/2} - x*P(x)*e^{-x^2/2}
        # f''(x) = P''(x)*e^{-x^2/2} - 2x*P'(x)*e^{-x^2/2} + (x^2-1)*P(x)*e^{-x^2/2}
        # So |f''(x)|^2 = [P''(x) - 2x*P'(x) + (x^2-1)*P(x)]^2 * e^{-x^2}
        
        # Let's compute P(x), P'(x), P''(x) as polynomials
        # H0 = 1, H0' = 0, H0'' = 0
        # H4 = x^4 - 6x^2 + 3, H4' = 4x^3 - 12x, H4'' = 12x^2 - 12
        # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        # H8' = 8x^7 - 168x^5 + 840x^3 - 840x
        # H8'' = 56x^6 - 840x^4 + 2520x^2 - 840
        # H12 = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
        # H12' = 12x^11 - 660x^9 + 10080x^7 - 55440x^5 + 92400x^3 - 55440x
        # H12'' = 132x^10 - 5940x^8 + 70560x^6 - 277200x^4 + 277200x^2 - 55440
        
        # P(x) = c0*1 + c4*(x^4 - 6x^2 + 3) + c8*(x^8 - 28x^6 + 210x^4 - 420x^2 + 105) + c12*(x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395)
        
        # Let's represent P(x) as a polynomial in x^2 (since all terms are even)
        # P(x) = a0 + a2*x^2 + a4*x^4 + a6*x^6 + a8*x^8 + a10*x^10 + a12*x^12
        
        a0 = c0 + 3*c4 + 105*c8 + 10395*c12
        a2 = -6*c4 - 420*c8 - 27720*c12
        a4 = c4 + 210*c8 + 23100*c12
        a6 = -28*c8 - 9240*c12
        a8 = c8 + 1260*c12
        a10 = -66*c12
        a12 = c12
        
        # P(x) = sum_{k=0}^{6} a_{2k} * x^{2k}
        
        # We need integrals of form ∫x^{2m} * P(x)^2 * e^{-x^2} dx
        # P(x)^2 = (sum a_i x^i)^2 = sum_{i,j} a_i a_j x^{i+j}
        # So ∫x^{2m} P(x)^2 e^{-x^2} dx = sum_{i,j} a_i a_j ∫x^{2m+i+j} e^{-x^2} dx
        # where i,j are even: i=2k, j=2l
        # ∫x^{2n} e^{-x^2} dx from -inf to inf = (2n-1)!! * sqrt(pi) / 2^n = (2n)! * sqrt(pi) / (4^n * n!)
        
        def even_moment(n):
            """∫x^{2n} e^{-x^2} dx from -inf to inf"""
            if n == 0:
                return np.sqrt(np.pi)
            # (2n)! * sqrt(pi) / (4^n * n!)
            from math import factorial
            return factorial(2*n) * np.sqrt(np.pi) / (4.0**n * factorial(n))
        
        # Compute I0 = ∫P(x)^2 e^{-x^2} dx
        # P(x)^2 = sum_{k=0}^{6} sum_{l=0}^{6} a_{2k} a_{2l} x^{2(k+l)}
        # I0 = sum_{k,l} a_{2k} a_{2l} even_moment(k+l)
        
        a = np.array([a0, a2, a4, a6, a8, a10, a12])  # a[0]=a0, a[1]=a2, ..., a[6]=a12
        
        # I0 = sum_{k=0}^{6} sum_{l=0}^{6} a[k]*a[l] * even_moment(k+l)
        I0 = 0.0
        for k in range(7):
            for l in range(7):
                I0 += a[k] * a[l] * even_moment(k + l)
        
        # I2 = ∫x^2 P(x)^2 e^{-x^2} dx = sum_{k,l} a[k]*a[l] * even_moment(k+l+1)
        I2 = 0.0
        for k in range(7):
            for l in range(7):
                I2 += a[k] * a[l] * even_moment(k + l + 1)
        
        # For f''(x): f(x) = P(x) e^{-x^2/2}
        # f'(x) = P'(x) e^{-x^2/2} - x P(x) e^{-x^2/2}
        # f''(x) = P''(x) e^{-x^2/2} - P'(x) x e^{-x^2/2} - P'(x) x e^{-x^2/2} - P(x) e^{-x^2/2} + x^2 P(x) e^{-x^2/2}
        #        = [P''(x) - 2x P'(x) + (x^2 - 1) P(x)] e^{-x^2/2}
        # Let Q(x) = P''(x) - 2x P'(x) + (x^2 - 1) P(x)
        # Then |f''(x)|^2 = Q(x)^2 e^{-x^2}
        
        # P(x) = sum a_k x^{2k}
        # P'(x) = sum 2k a_k x^{2k-1}
        # P''(x) = sum 2k(2k-1) a_k x^{2k-2}
        
        # Q(x) = P''(x) - 2x P'(x) + (x^2 - 1) P(x)
        # Let's expand:
        # P''(x) = sum_{k=1}^{6} 2k(2k-1) a_k x^{2k-2}
        # -2x P'(x) = -2 sum_{k=1}^{6} 2k a_k x^{2k}
        # (x^2 - 1) P(x) = sum_{k=0}^{6} a_k x^{2k+2} - sum_{k=0}^{6} a_k x^{2k}
        
        # Collect by powers of x:
        # x^0: from P'': 2*1*1*a_1 = 2a_1; from -2xP': 0; from (x^2-1)P: -a_0
        #   -> coeff of x^0: 2*a[1] - a[0]
        # x^2: from P'': 2*2*3*a_2 = 12a_2; from -2xP': -4a_1; from (x^2-1)P: a_0 - a_1
        #   -> coeff of x^2: 12*a[2] - 4*a[1] + a[0] - a[1] = 12*a[2] - 5*a[1] + a[0]
        # x^4: from P'': 2*3*5*a_3 = 30a_3; from -2xP': -12a_2; from (x^2-1)P: a_1 - a_2
        #   -> coeff of x^4: 30*a[3] - 12*a[2] + a[1] - a[2] = 30*a[3] - 13*a[2] + a[1]
        # x^6: from P'': 2*4*7*a_4 = 56a_4; from -2xP': -24a_3; from (x^2-1)P: a_2 - a_3
        #   -> coeff of x^6: 56*a[4] - 24*a[3] + a[2] - a[3] = 56*a[4] - 25*a[3] + a[2]
        # x^8: from P'': 2*5*9*a_5 = 90a_5; from -2xP': -40a_4; from (x^2-1)P: a_3 - a_4
        #   -> coeff of x^8: 90*a[5] - 40*a[4] + a[3] - a_4] = 90*a[5] - 41*a[4] + a[3]
        # x^10: from P'': 2*6*11*a_6 = 132a_6; from -2xP': -60a_5; from (x^2-1)P: a_4 - a_5
        #   -> coeff of x^10: 132*a[6] - 60*a[5] + a[4] - a[5] = 132*a[6] - 61*a[5] + a[4]
        # x^12: from P'': 0; from -2xP': 0; from (x^2-1)P: a_5 - a_6... wait
        #   Actually (x^2-1)P(x) gives x^{2k+2} terms: a_k x^{2k+2}
        #   So x^12 comes from a_5 x^{12} (k=5, 2k+2=12) and -a_6 x^{12} (k=6, 2k=12)
        #   -> coeff of x^12: a[5] - a[6]
        # x^14: from (x^2-1)P: a_6 x^{14}
        #   -> coeff of x^14: a[6]
        
        q0 = 2*a[1] - a[0]
        q2 = 12*a[2] - 5*a[1] + a[0]
        q4 = 30*a[3] - 13*a[2] + a[1]
        q6 = 56*a[4] - 25*a[3] + a[2]
        q8 = 90*a[5] - 41*a[4] + a[3]
        q10 = 132*a[6] - 61*a[5] + a[4]
        q12 = a[5] - a[6]
        q14 = a[6]
        
        q = np.array([q0, q2, q4, q6, q8, q10, q12, q14])  # q[0]=q0, q[1]=q2, ..., q[7]=q14
        
        # ∫Q(x)^2 e^{-x^2} dx = sum_{i=0}^{7} sum_{j=0}^{7} q[i]*q[j] * even_moment(i+j)
        # Note: Q(x) = sum_{i=0}^{7} q_i x^{2i}, so Q(x)^2 = sum_{i,j} q_i q_j x^{2(i+j)}
        # ∫x^{2(i+j)} e^{-x^2} dx = even_moment(i+j)
        
        I_dd = 0.0
        for i in range(8):
            for j in range(8):
                I_dd += q[i] * q[j] * even_moment(i + j)
        
        if I0 <= 0 or not np.isfinite(I0) or not np.isfinite(I2) or not np.isfinite(I_dd):
            return np.inf
        
        # A(f) = sqrt(I2/I0)
        # A(F̂f) = sqrt(J2/J0) where J0 = I0 (Parseval) and J2 = (1/(2π)^2) * I_dd
        # Wait, let me recheck the FT convention.
        # F̂(f)(ξ) = ∫f(x) e^{-2πixξ} dx
        # Parseval: ∫|F̂f(ξ)|² dξ = ∫|f(x)|² dx = I0
        # ∫ξ²|F̂f(ξ)|² dξ = (1/(2π)²) ∫|f''(x)|² dx = (1/(4π²)) I_dd
        
        # So A(f)*A(F̂f) = sqrt(I2/I0) * sqrt(I_dd/(4π²I0)) = sqrt(I2 * I_dd / (4π² I0²))
        #                = (1/(2π)) * sqrt(I2 * I_dd / I0²)
        
        if I2 < 0 or I_dd < 0:
            return np.inf
        
        product = (1.0 / (2.0 * np.pi)) * np.sqrt(I2 * I_dd / (I0 * I0))
        
        return product
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            val = compute_uncertainty_product_exact(c0, c4, c8)
            if not np.isfinite(val):
                return np.inf
            return val
        except Exception:
            return np.inf
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    # Try differential evolution for global search
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential evolution
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = opt.differential_evolution(
                objective, bounds, 
                maxiter=200, popsize=15,
                tol=1e-8, mutation=(0.5, 1.5),
                recombination=0.9, seed=42,
                polish=True, init='latinhypercube'
            )
            if np.isfinite(de_result.fun) and de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starting points
    if time.time() < deadline - 15:
        # Start from best and some random perturbations
        start_points = [best_coeffs.copy()]
        for i in range(20):
            if time.time() > deadline - 15:
                break
            scale = 0.5 * (1 + i/5.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
            start_points.append(x_start)
        
        for x_start in start_points:
            if time.time() > deadline - 10:
                break
            try:
                res = opt.minimize(objective, x_start, method='Nelder-Mead',
                                   options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500,
                                          'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead',
                               options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000,
                                      'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
