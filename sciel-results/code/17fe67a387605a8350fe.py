import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Physicists' Hermite Polynomials
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280

    # Values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Constraint: c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280

    # Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24
    # I[n] = (2n-1)/2 * I[n-1], I[0] = sqrt(pi)
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]

    # Hermite polynomial coefficients in standard basis x^(2k)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12 = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280

    # Coefficient matrix: rows are H0, H4, H8, H12; cols are x^0, x^2, ..., x^12 (indices 0..6)
    # H0: [1, 0, 0, 0, 0, 0, 0]
    # H4: [12, -48, 16, 0, 0, 0, 0]
    # H8: [1680, -48384, 26880, -4480, 256, 0, 0]
    # H12: [665280, -15974400, 16636928, -7654656, 1548288, -129024, 4096]
    
    # We want P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Let's express in terms of c0, c4, c8, with c12 derived.
    # p_k = c0*h0_k + c4*h4_k + c8*h8_k + c12*h12_k
    # where h_n_k is the coefficient of x^(2k) in H_n.

    # Let's define the coefficients of each Hermite poly
    h0 = np.array([1.0, 0, 0, 0, 0, 0, 0])
    h4 = np.array([12.0, -48, 16, 0, 0, 0, 0])
    h8 = np.array([1680.0, -48384, 26880, -4480, 256, 0, 0])
    h12 = np.array([665280.0, -15974400, 16636928, -7654656, 1548288, -129024, 4096])

    def get_coeffs(c0, c4, c8):
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280.0
        p = c0*h0 + c4*h4 + c8*h8 + c12*h12
        return p

    def compute_bound(c0, c4, c8):
        p = get_coeffs(c0, c4, c8)
        
        # Q(x) is the Fourier transform related polynomial
        # For even Hermite polynomials, the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
        # Wait, the Fourier transform convention matters.
        # Usually F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx or similar.
        # The problem statement says "Fourier-invariant".
        # For g(x) = P(x) e^{-x^2/2}, if P is a sum of even Hermite polynomials H_{4k},
        # then the Fourier transform is also a sum of even Hermite polynomials with same coefficients?
        # Let's check.
        # F(e^{-x^2/2}) = e^{-xi^2/2} (with appropriate normalization).
        # F(H_n(x) e^{-x^2/2}) = (-i)^n H_n(x) e^{-xi^2/2} ?
        # If n is a multiple of 4, (-i)^n = 1.
        # So if P(x) = sum c_k H_{4k}(x), then F(g)(xi) = sum c_k H_{4k}(xi) e^{-xi^2/2} = g(xi).
        # So g is its own Fourier transform (up to normalization).
        # The variance of g is Var(X) where X ~ |g|^2.
        # The variance of F(g) is Var(Xi) where Xi ~ |F(g)|^2.
        # Since |F(g)|^2 = |g|^2, the variances are identical?
        # No, the density is |g(x)|^2 / ||g||_2^2.
        # The variance is <x^2> - <x>^2. Since g is even, <x> = 0.
        # So Var = <x^2> = integral x^2 |g(x)|^2 dx / integral |g(x)|^2 dx.
        # The uncertainty product is sqrt(Var_x * Var_xi).
        # If g is self-Fourier, Var_x = Var_xi, so product is Var_x.
        
        # Let's verify the "Fourier-invariant" claim.
        # If g(x) = P(x) e^{-x^2/2} with P even, then g is even.
        # F(g)(xi) = integral P(x) e^{-x^2/2} e^{-i x xi} dx.
        # Using the property that Hermite functions are eigenfunctions of the Fourier transform.
        # h_n(x) = H_n(x) e^{-x^2/2} / (2^n n! sqrt(pi))^{1/2} are Hermite functions.
        # F(h_n)(xi) = (-i)^n h_n(xi).
        # So F(H_n(x) e^{-x^2/2}) = (-i)^n H_n(x) e^{-xi^2/2} * (normalization factor cancels).
        # For n=0, 4, 8, 12, (-i)^n = 1.
        # So yes, g is its own Fourier transform.
        # Thus Var_x = Var_xi.
        # The product is Var_x.
        
        # Var_x = (integral x^2 |P(x)|^2 e^{-x^2} dx) / (integral |P(x)|^2 e^{-x^2} dx)
        # Note: |g|^2 = |P|^2 e^{-x^2}.
        
        # Let m0 = integral |P|^2 e^{-x^2} dx
        # Let m2 = integral x^2 |P|^2 e^{-x^2} dx
        # Var = m2 / m0
        
        # We need to compute m0 and m2.
        # |P(x)|^2 = P(x)^2 since P is real.
        # P(x) = sum_{k=0}^6 p_k x^{2k}
        # P(x)^2 = sum_{i,j} p_i p_j x^{2(i+j)}
        # m0 = sum_{i,j} p_i p_j I[i+j]
        # m2 = sum_{i,j} p_i p_j I[i+j+1]
        
        # Note: I[n] is integral x^(2n) e^{-x^2} dx.
        # So x^{2(i+j)} corresponds to n=i+j.
        # x^2 * x^{2(i+j)} = x^{2(i+j+1)} corresponds to n=i+j+1.
        
        m0 = 0.0
        m2 = 0.0
        for i in range(7):
            for j in range(7):
                pij = p[i] * p[j]
                if pij == 0:
                    continue
                idx0 = i + j
                idx2 = i + j + 1
                if idx0 < 25:
                    m0 += pij * I[idx0]
                if idx2 < 25:
                    m2 += pij * I[idx2]
                
        if m0 <= 1e-15 or not np.isfinite(m0):
            return 1e18
        if m2 <= 0:
            return 0.0 # Should not happen for non-zero P
            
        var = m2 / m0
        return var

    # The score is the upper bound on C4.
    # The problem says "score is the upper bound on C4".
    # And "evaluator measures the raw value (lower is better)".
    # So we want to minimize the variance.
    
    def objective(c):
        return compute_bound(c[0], c[1], c[2])

    # Initial guess from current program
    c_init = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_c = c_init.copy()
    best_val = objective(c_init)
    
    # Phase 1: Nelder-Mead from initial guess
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(objective, c_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
        
    # Phase 2: Multi-start Nelder-Mead
    # Generate some random starts
    rng = np.random.default_rng(42)
    starts = [best_c.copy()]
    
    # Around seed
    for _ in range(20):
        s = c_init + rng.normal(0, 0.1, 3)
        starts.append(s)
        
    # Wider exploration
    for _ in range(30):
        s = np.array([rng.uniform(-3.0, 2.0), rng.uniform(-0.5, 0.5), rng.uniform(-0.01, 0.01)])
        starts.append(s)
        
    # Evaluate starts
    scored_starts = []
    for s in starts:
        if time.time() > deadline - 50:
            break
        try:
            val = objective(s)
            if np.isfinite(val):
                scored_starts.append((val, s))
        except Exception:
            pass
            
    scored_starts.sort(key=lambda x: x[0])
    
    # Take top 10 starts for local optimization
    top_starts = [s for _, s in scored_starts[:10]]
    if best_c not in top_starts:
        top_starts.append(best_c)
        
    for s in top_starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
            
    return best_c
