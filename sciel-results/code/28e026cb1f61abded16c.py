import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

# Deadline guard: evaluator kills at 210s. Use 195s as our internal deadline.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0

def _check_deadline():
    return time.time() < _DEADLINE

# Precompute moments I[n] = integral_{-inf}^{inf} x^{2n} e^{-x^2} dx = sqrt(pi) * (2n-1)!! / 2^n
# Actually, let's use the recurrence for the integral over [0, inf] or [-inf, inf].
# Let J[n] = integral_{-inf}^{inf} x^{2n} e^{-x^2} dx.
# J[0] = sqrt(pi)
# J[n] = (2n-1)/2 * J[n-1]
# We need J[n] for n up to 12 (since degree of P^2 is 12, times x^2 for variance is 14? No, variance is <x^2>/<x^0> so we need up to 12 for P^2, and for F we need Q^2 which is also degree 12).
# Wait, g(x) = P(x) e^{-x^2/2}.
# ||g||_2^2 = integral P(x)^2 e^{-x^2} dx.
# Variance of g: <x^2> = integral x^2 P(x)^2 e^{-x^2} dx / integral P(x)^2 e^{-x^2} dx.
# So we need integrals of x^{2k} P(x)^2 e^{-x^2} for k=0,1.
# P(x) is degree 12. P(x)^2 is degree 24. x^2 P(x)^2 is degree 26.
# So we need J[n] for n up to 13 (since 2n = 26 => n=13).

def _precompute_moments(max_n=13):
    J = np.zeros(max_n + 1)
    J[0] = np.sqrt(np.pi)
    for n in range(1, max_n + 1):
        J[n] = (2.0 * n - 1.0) / 2.0 * J[n - 1]
    return J

_J = _precompute_moments(13)

# Hermite polynomials (Physicists') coefficients
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
# H_12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280

# We parameterize g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * e^{-x^2/2}
# Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
# c12 = -(c0 + 12*c4 + 1680*c8) / 665280

# To use the "log-magnitude" idea from inspiration 1, we can optimize over (log|c0|, log|c4|, log|c8|, sign0, sign4, sign8)
# But c12 is derived.
# The uncertainty product is A(f) A(F).
# For g(x) = P(x) e^{-x^2/2}, the Fourier transform is (2pi)^{-1/2} (i)^n P(-x) e^{-x^2/2} if P is a Hermite polynomial?
# No, the Fourier transform of H_n(x) e^{-x^2/2} is (i)^n H_n(x) e^{-x^2/2}.
# Since we are summing H_n with n=0,4,8,12, and all are multiples of 4, (i)^n = 1.
# So g_hat(x) = (2pi)^{-1/2} P(x) e^{-x^2/2} = (2pi)^{-1/2} g(x).
# Wait, the normalization of Fourier transform matters.
# Usually A(f) = (integral |x f(x)|^2 / integral |f(x)|^2)^{1/2}.
# If g_hat = C g, then A(g_hat) = A(g).
# So A(g) A(g_hat) = A(g)^2.
# A(g)^2 = (integral x^2 g(x)^2 / integral g(x)^2) * (integral x^2 g_hat(x)^2 / integral g_hat(x)^2)
# Since g_hat is just a constant times g, the ratio is the same.
# So the product is (integral x^2 g^2 / integral g^2) * (integral x^2 g^2 / integral g^2) = (integral x^2 g^2 / integral g^2)^2.
# Wait, is g_hat(x) = P(x) e^{-x^2/2}?
# FT of H_n(x) e^{-x^2/2} is (i)^n H_n(x) e^{-x^2/2}.
# Yes.
# So the problem reduces to minimizing the variance of the probability density |g(x)|^2.
# Let rho(x) = |g(x)|^2 / ||g||_2^2.
# We want to minimize (integral x^2 rho(x) dx)^2.
# This is equivalent to minimizing the variance of x under the measure rho.
# Since g is real, rho(x) = P(x)^2 e^{-x^2} / (integral P^2 e^{-x^2}).
# We want to minimize (integral x^2 P(x)^2 e^{-x^2} dx) / (integral P(x)^2 e^{-x^2} dx).

def _compute_variance(c):
    c0, c4, c8 = c
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 665280.0
    
    # Construct P(x) coefficients
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # We need integral x^{2k} P(x)^2 e^{-x^2} dx for k=0,1.
    # P(x) is a polynomial of degree 12.
    # Let's represent P(x) by its coefficients p[0..12] where P(x) = sum p[i] x^i.
    p = np.zeros(13)
    # H0
    p[0] += c0
    # H4: 16x^4 - 48x^2 + 12
    p[0] += 12.0 * c4
    p[2] += -48.0 * c4
    p[4] += 16.0 * c4
    # H8: 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    p[0] += 1680.0 * c8
    p[2] += -48384.0 * c8
    p[4] += 26880.0 * c8
    p[6] += -4480.0 * c8
    p[8] += 256.0 * c8
    # H12: 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280
    p[0] += 665280.0 * c12
    p[2] += -15974400.0 * c12
    p[4] += 16636928.0 * c12
    p[6] += -7654656.0 * c12
    p[8] += 1548288.0 * c12
    p[10] += -129024.0 * c12
    p[12] += 4096.0 * c12
    
    # Compute m0 = integral P^2 e^{-x^2} and m2 = integral x^2 P^2 e^{-x^2}
    # m_k = sum_{i,j} p[i] p[j] integral x^{i+j+2k} e^{-x^2} dx
    # integral x^{2n} e^{-x^2} dx = J[n] if i+j+2k is even, else 0.
    # Since P is even, p[odd] = 0. So i+j is even.
    m0 = 0.0
    m2 = 0.0
    for i in range(0, 13, 2):
        if p[i] == 0: continue
        for j in range(0, 13, 2):
            if p[j] == 0: continue
            n0 = (i + j) // 2
            n2 = (i + j + 2) // 2
            if n0 <= 13:
                m0 += p[i] * p[j] * _J[n0]
            if n2 <= 13:
                m2 += p[i] * p[j] * _J[n2]
                
    if m0 <= 0:
        return 1e18
    return m2 / m0

def _objective(c):
    return _compute_variance(c)

def construct():
    start_time = time.time()
    
    # Initial guess from seed
    c_init = np.array([-1.13, 0.04, 1.0/3240.0])
    best_c = c_init.copy()
    best_val = _objective(c_init)
    
    # Phase 1: Local optimization with Nelder-Mead
    try:
        res = minimize(_objective, c_init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass
    
    # Phase 2: Differential Evolution for global search
    # Use a wider range for c0, c4, c8
    # c0 is around -1, c4 around 0.04, c8 around 0.0003
    # Let's try bounds that allow for some exploration
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    if time.time() - start_time < _DEADLINE - 30:
        try:
            de_res = differential_evolution(_objective, bounds, seed=42, maxiter=100, 
                                           tol=1e-10, polish=True, 
                                           callback=lambda x, convergence: (time.time() - start_time) < _DEADLINE - 10)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_c = de_res.x
        except Exception:
            pass
    
    # Phase 3: Fine-tune with Nelder-Mead from best found
    if time.time() - start_time < _DEADLINE - 10:
        try:
            res2 = minimize(_objective, best_c, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x
        except Exception:
            pass
            
    return best_c
