"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def _compute_autocorr(x):
        """Compute circular autocorrelation via FFT."""
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(x, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        return acf

    def _max_sidelobe(x):
        """Maximum absolute autocorrelation at non-zero lags."""
        acf = _compute_autocorr(x)
        return np.max(np.abs(acf[1:N]))

    def _soft_max_sidelobe(x, beta=50.0):
        """Soft-max surrogate for the max sidelobe, smooth and differentiable."""
        acf = _compute_autocorr(x)
        sidelobes = np.abs(acf[1:N])
        # Log-sum-exp trick for numerical stability
        m = np.max(sidelobes)
        if m == 0:
            return 0.0
        log_sum = m + np.log(np.sum(np.exp(beta * (sidelobes - m))))
        return log_sum / beta

    def _objective_and_gradient(x):
        """Objective (soft-max sidelobe / N) and its gradient via numerical differentiation."""
        beta = 50.0
        acf = _compute_autocorr(x)
        sidelobes = np.abs(acf[1:N])
        m = np.max(sidelobes)
        if m == 0:
            return 0.0, np.zeros_like(x)
        
        # Soft-max value
        exp_terms = np.exp(beta * (sidelobes - m))
        sum_exp = np.sum(exp_terms)
        obj_val = (m + np.log(sum_exp)) / beta / N
        
        # Gradient via FFT-based derivative
        # d/dx of |acf[k]| for k in 1..N-1
        # acf = ifft(|fft(x)|^2), so gradient involves fft and ifft
        F = np.fft.fft(x, n=acf.size)
        # grad of |F|^2 w.r.t x: 2*Re(F * conj(fft(x))) -> actually for real x, 
        # d/dx_k sum |F_j|^2 = 2*Re( sum_j F_j * conj(dF_j/dx_k) )
        # dF_j/dx_k = i*j*exp(-2*pi*i*j*k/n) for circular
        # This is getting complex; use numerical gradient for reliability
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            xneg = x.copy()
            xneg[i] -= eps
            sp = _soft_max_sidelobe(xp, beta)
            sn = _soft_max_sidelobe(xneg, beta)
            grad[i] = (sp - sn) / (2 * eps)
        
        return obj_val, grad

    def _local_search(h, time_limit=5.0):
        """Discrete local search: flip bits to reduce max sidelobe."""
        best_h = h.copy()
        best_score = _max_sidelobe(best_h)
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            improved = False
            for i in range(N):
                if time.time() - t_start > time_limit:
                    break
                h[i] *= -1
                score = _max_sidelobe(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] *= -1
            if not improved:
                break
        
        return best_h, best_score

    best_h = np.ones(N)
    best_score = _max_sidelobe(best_h)

    rng = np.random.RandomState(42)
    
    # Try multiple continuous relaxations with different initializations
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Different initialization strategies
        if r == 0:
            # Start from ones
            x0 = np.ones(N)
        elif r == 1:
            # Start from random
            x0 = rng.choice([1.0, -1.0], N)
        elif r == 2:
            # Start from a structured pattern (alternating)
            x0 = np.ones(N)
            x0[::2] = -1.0
        elif r == 3:
            # Start from a pseudo-random structured pattern
            x0 = np.cos(2 * np.pi * rng.uniform(0, 1) * np.arange(N))
            x0 = np.sign(x0)
            x0[x0 == 0] = 1.0
        elif r == 4:
            # Start from a low-frequency pattern
            freq = rng.uniform(0.1, 0.5)
            x0 = np.cos(2 * np.pi * freq * np.arange(N) / N)
            x0 = np.sign(x0)
            x0[x0 == 0] = 1.0
        elif r == 5:
            # Start from another random
            x0 = rng.choice([1.0, -1.0], N)
        elif r == 6:
            # Start from a pattern based on Legendre sequence approximation
            x0 = np.ones(N)
            # Use a simple structured pattern
            for i in range(N):
                if (i // 10) % 3 == 0:
                    x0[i] = -1.0
        else:
            x0 = rng.choice([1.0, -1.0], N)
        
        # Continuous optimization using L-BFGS-B
        # Use a softer beta initially for better landscape navigation
        t_opt_start = time.time()
        max_opt_time = 15.0
        
        # First pass with softer beta
        beta_soft = 10.0
        def obj_soft(x):
            acf = _compute_autocorr(x)
            sidelobes = np.abs(acf[1:N])
            m = np.max(sidelobes)
            if m == 0:
                return 0.0
            exp_terms = np.exp(beta_soft * (sidelobes - m))
            return (m + np.log(np.sum(exp_terms))) / beta_soft / N
        
        # Numerical gradient for the soft objective
        eps = 1e-6
        def obj_and_grad_soft(x):
            val = obj_soft(x)
            grad = np.zeros_like(x)
            for i in range(0, N, 2):  # Skip every other for speed, still informative
                if time.time() - t_opt_start > max_opt_time:
                    break
                xp = x.copy()
                xp[i] += eps
                xneg = x.copy()
                xneg[i] -= eps
                grad[i] = (obj_soft(xp) - obj_soft(xneg)) / (2 * eps)
            # Fill in skipped gradients with 0 (approximation)
            for i in range(1, N, 2):
                if time.time() - t_opt_start > max_opt_time:
                    break
                xp = x.copy()
                xp[i] += eps
                xneg = x.copy()
                xneg[i] -= eps
                grad[i] = (obj_soft(xp) - obj_soft(xneg)) / (2 * eps)
            return val, grad
        
        try:
            res = minimize(
                obj_and_grad_soft,
                x0,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8},
                callback=None
            )
        except Exception:
            pass
        
        if time.time() > deadline - 20.0:
            break
        
        # Round to ±1
        x_opt = res.x if hasattr(res, 'x') else x0
        h_rounded = np.sign(x_opt)
        h_rounded[h_rounded == 0] = 1.0
        h_rounded = h_rounded.astype(np.float64)
        
        # Local search refinement
        h_ls, score_ls = _local_search(h_rounded, time_limit=8.0)
        
        if score_ls < best_score:
            best_score = score_ls
            best_h = h_ls.copy()
        
        # Second pass with harder beta from the rounded solution
        if time.time() > deadline - 15.0:
            break
        
        beta_hard = 50.0
        def obj_hard(x):
            acf = _compute_autocorr(x)
            sidelobes = np.abs(acf[1:N])
            m = np.max(sidelobes)
            if m == 0:
                return 0.0
            exp_terms = np.exp(beta_hard * (sidelobes - m))
            return (m + np.log(np.sum(exp_terms))) / beta_hard / N
        
        def obj_and_grad_hard(x):
            val = obj_hard(x)
            grad = np.zeros_like(x)
            eps2 = 1e-7
            for i in range(N):
                if time.time() - t_opt_start > max_opt_time:
                    break
                xp = x.copy()
                xp[i] += eps2
                xneg = x.copy()
                xneg[i] -= eps2
                grad[i] = (obj_hard(xp) - obj_hard(xneg)) / (2 * eps2)
            return val, grad
        
        try:
            res2 = minimize(
                obj_and_grad_hard,
                h_rounded,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
        
        x_opt2 = res2.x if hasattr(res2, 'x') else h_rounded
        h_rounded2 = np.sign(x_opt2)
        h_rounded2[h_rounded2 == 0] = 1.0
        h_rounded2 = h_rounded2.astype(np.float64)
        
        h_ls2, score_ls2 = _local_search(h_rounded2, time_limit=5.0)
        
        if score_ls2 < best_score:
            best_score = score_ls2
            best_h = h_ls2.copy()
    
    # Final discrete local search on the best found
    if time.time() < deadline - 5.0:
        best_h, best_score = _local_search(best_h, time_limit=10.0)
    
    return best_h
# EVOLVE-BLOCK-END
