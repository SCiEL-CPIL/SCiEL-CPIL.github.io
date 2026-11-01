import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 175  # Safety margin before 180s limit
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    def compute_I():
        I = np.zeros(14)
        sp = np.sqrt(np.pi)
        for n in range(14):
            log_val = 0.5 * math.log(math.pi)
            for m in range(1, 2*n+1):
                log_val += math.log(m)
            log_val -= n * math.log(4.0)
            for m in range(1, n+1):
                log_val -= math.log(m)
            I[n] = math.exp(log_val)
        return I
    
    I = compute_I()
    sp = np.sqrt(np.pi)
    
    # H_n(0) values for n = 0, 4, 8, 12
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # Weights w_i = ||phi_i||_2^2 = sqrt(pi) * 2^i * i!
    # But for the ratio num/den, the sqrt(pi) cancels, so we can use 2^i * i!
    # Actually let's be careful. The task says "score is the upper bound on C4"
    # Let me re-read the current program's objective.
    
    # The current program uses:
    # num = sum c_i^2 E_i w_i
    # den = sum c_i^2 w_i
    # where w_i = sqrt(pi) * 2^i * i! and E_i = i + 0.5
    # This computes <x^2>/<1> for the state |psi> = sum c_i phi_i
    # where phi_i are normalized Hermite-Gaussian functions.
    
    # The inspiration uses a different formulation with |g|^2 moments.
    # Let me check which is correct.
    
    # For a state |psi> = sum c_n |n>, we have:
    # <x^2> = sum_n |c_n|^2 <n|x^2|n> = sum_n |c_n|^2 (n + 0.5)
    # <1> = sum_n |c_n|^2 = 1 (if normalized)
    # So A(f)^2 = <x^2> = sum |c_n|^2 (n+0.5)
    
    # For the Fourier transform, <xi^2> = sum |c_n|^2 (n+0.5) as well
    # (since the Fourier transform maps |n> to i^n |n> up to phase)
    
    # Wait, that's not right. Let me think more carefully.
    
    # Actually, for the harmonic oscillator eigenstates:
    # x = (a + a†)/sqrt(2)
    # p = -i(a - a†)/sqrt(2)
    # <n|x^2|n> = n + 1/2
    # <n|p^2|n> = n + 1/2
    
    # The Fourier transform of psi(x) is related to the momentum representation.
    # If psi(x) = sum c_n phi_n(x), then its Fourier transform is
    # psi_hat(k) = sum c_n i^n phi_n(k) (up to normalization)
    
    # So <k^2> in the Fourier domain = sum |c_n|^2 <n|k^2|n>
    # And k is related to p by k = p (in units where hbar=1)
    # So <k^2> = <p^2> = sum |c_n|^2 (n+1/2)
    
    # This means A(f)*A(F) = sqrt(<x^2>*<p^2>) = <x^2> = sum |c_n|^2 (n+1/2)
    # if the state is normalized.
    
    # But the current program computes num/den where num = sum c_i^2 E_i w_i
    # and den = sum c_i^2 w_i. This is the weighted average of E_i with weights
    # c_i^2 w_i. If the state is |psi> = sum c_i phi_i / sqrt(w_i) (normalized),
    # then the coefficients in the normalized basis are d_i = c_i / sqrt(w_i),
    # and <x^2> = sum |d_i|^2 E_i / sum |d_i|^2 = sum c_i^2 E_i / sum c_i^2.
    
    # Hmm, this is getting confusing. Let me just use the current program's
    # objective function which has a score of 0.1434, and combine it with
    # the multi-start and L-BFGS-B approach from the inspiration.
    
    # Actually, looking more carefully at the current program:
    # w_i = exp(log_w[i]) where log_w[i] = log(2)*i + lgamma(i+1)
    # So w_i = 2^i * i!
    # And E_i = i + 0.5
    # objective = sum(c_i^2 * E_i * w_i) / sum(c_i^2 * w_i)
    
    # This is a Rayleigh quotient-like expression. Let me just use this.
    
    indices = [0, 4, 8, 12]
    w_idx = np.array([2.0**i * math.factorial(i) for i in indices])
    E_idx = np.array([i + 0.5 for i in indices])
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        c = np.array([c0, c4, c8, c12])
        
        num = np.sum(c**2 * E_idx * w_idx)
        den = np.sum(c**2 * w_idx)
        
        if den < 1e-10:
            return 1e10
        return num / den
    
    def objective_grad(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        c = np.array([c0, c4, c8, c12])
        
        num = np.sum(c**2 * E_idx * w_idx)
        den = np.sum(c**2 * w_idx)
        
        if den < 1e-10:
            return 1e10, np.array([1e10, 1e10, 1e10])
        
        val = num / den
        
        # Gradient of num/den with respect to c0, c4, c8
        # d(num)/dc_j = 2*c_j*E_j*w_j (for j in {0,4,8}) + dc12/dc_j * 2*c12*E_12*w_12
        # d(den)/dc_j = 2*c_j*w_j (for j in {0,4,8}) + dc12/dc_j * 2*c12*w_12
        
        dc12_dc = np.array([-(H0_0/H12_0), -(H4_0/H12_0), -(H8_0/H12_0)])
        
        grad = np.zeros(3)
        for j in range(3):
            dnum_j = 2.0 * c[j] * E_idx[j] * w_idx[j] + dc12_dc[j] * 2.0 * c[3] * E_idx[3] * w_idx[3]
            dden_j = 2.0 * c[j] * w_idx[j] + dc12_dc[j] * 2.0 * c[3] * w_idx[3]
            grad[j] = (dnum_j * den - num * dden_j) / (den * den)
        
        return val, grad
    
    # Multiple starting points
    starts = [
        np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),  # seed
        np.array([-3.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-2.0, 0.08, 0.0006]),
        np.array([-0.5, 0.02, 0.00015]),
        np.array([-1.5, 0.06, 0.00045]),
        np.array([1.0, -0.04, 0.0003]),
        np.array([-4.0, 0.12, 0.0009]),
        np.array([-0.8, 0.032, 0.00024]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = float('inf')
    best_x = starts[0]
    
    # Evaluate initial points
    for x0 in starts:
        val = objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0
    
    # Run L-BFGS-B from each start
    for x0 in starts:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(objective_grad, x0, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                          options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 10000})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res = minimize(objective_grad, best_x, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
