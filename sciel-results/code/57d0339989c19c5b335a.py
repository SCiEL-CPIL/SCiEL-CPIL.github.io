"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _fourier_basis(N, K):
    """Create Fourier basis matrix: K complex modes mapped to real."""
    # Use first K positive frequencies (and their conjugates for real-valued)
    # h[i] = c0 + sum_{k=1}^{K-1} (a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    # Total params: 1 + 2*(K-1) = 2K-1
    i = np.arange(N)
    basis = np.zeros((N, 2*K - 1))
    basis[:, 0] = 1.0
    for k in range(1, K):
        basis[:, 2*k-1] = np.cos(2 * np.pi * k * i / N)
        basis[:, 2*k] = np.sin(2 * np.pi * k * i / N)
    return basis

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    # Start with constant solution
    h_const = np.full(N, 0.5)
    current_score = _score(h_const)  # = 0.5 due to zero padding at end
    h_final = h_const.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Fourier-based optimization with small number of modes
    try:
        if time.time() < deadline - 40.0:
            K = 8  # number of Fourier modes (2K-1 parameters)
            basis = _fourier_basis(N, K)
            n_params = 2 * K - 1
            
            # Initial coefficients: c0 = 0.5, rest zero
            c0_init = np.zeros(n_params)
            c0_init[0] = 0.5
            
            def fourier_h(c):
                """Reconstruct h from Fourier coefficients."""
                return basis @ c
            
            def fourier_objective(c):
                h = fourier_h(c)
                # Penalty for out-of-bounds
                penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
                # Penalty for sum constraint
                sum_penalty = 500.0 * (np.sum(h) - N/2.0)**2
                # Smooth max of overlap
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=200.0)
                return smax + penalty + sum_penalty
            
            def fourier_grad(c):
                h = fourier_h(c)
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                # Gradient of smooth max
                exps = np.exp(200.0 * (overlap - np.max(overlap)))
                weights = exps / np.sum(exps)
                grad_overlap = np.zeros(N)
                # d/dh[i] of h[i]*(1-h[i+1]) for i < N-1
                for i in range(N - 1):
                    grad_overlap[i] += (1.0 - h[i+1])
                    grad_overlap[i+1] += -h[i]
                grad_overlap[N-1] += 1.0  # h[N-1] * (1-0), derivative w.r.t h[N-1] is 1
                # Actually let me redo this properly
                # overlap[i] = h[i] * (1 - shifted[i])
                # For i < N-1: shifted[i] = h[i+1], so overlap[i] = h[i]*(1-h[i+1])
                # d(overlap[i])/dh[j]:
                #   if j == i: (1 - h[i+1])
                #   if j == i+1: -h[i]
                # For i = N-1: overlap[N-1] = h[N-1]*(1-0) = h[N-1]
                #   d/dh[N-1] = 1
                
                grad_smax = np.dot(weights, 1.0/1.0)  # This isn't right, let me use automatic approach
                
                # Use numerical gradient for reliability
                eps = 1e-7
                grad = np.zeros(n_params)
                for j in range(n_params):
                    c_plus = c.copy()
                    c_plus[j] += eps
                    grad[j] = (fourier_objective(c_plus) - fourier_objective(c)) / eps
                return grad
            
            # Use L-BFGS-B on Fourier coefficients
            result_f = minimize(
                fourier_objective,
                c0_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )
            
            if result_f.fun < 10.0:
                h_f = fourier_h(result_f.x)
                _try_update(h_f)
                
            # Try with more modes
            for K2 in [12, 16]:
                if time.time() > deadline - 35.0:
                    break
                basis2 = _fourier_basis(N, K2)
                n_params2 = 2 * K2 - 1
                c0_init2 = np.zeros(n_params2)
                c0_init2[0] = 0.5
                
                def fourier_objective2(c, b=basis2):
                    h = b @ c
                    penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
                    sum_penalty = 500.0 * (np.sum(h) - N/2.0)**2
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    smax = _smooth_max(overlap, alpha=200.0)
                    return smax + penalty + sum_penalty
                
                result_f2 = minimize(
                    fourier_objective2,
                    c0_init2,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-15, 'disp': False}
                )
                
                if result_f2.fun < 10.0:
                    h_f2 = basis2 @ result_f2.x
                    _try_update(h_f2)

    except Exception:
        pass

    # Phase 2: Full-space SLSQP from current best
    try:
        if time.time() < deadline - 25.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 10000.0 * np.maximum(0.0, smax - t)**2

            x0 = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False}
            )

            if result.fun < 100.0:
                _try_update(result.x[:N])

    except Exception:
        pass

    # Phase 3: Multi-start with diverse initializations
    try:
        if time.time() < deadline - 15.0:
            np.random.seed(123)
            n_starts = 8
            for trial in range(n_starts):
                if time.time() > deadline - 10.0:
                    break
                
                if trial == 0:
                    freq = 1.0
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 1:
                    freq = 2.0
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 2:
                    freq = 3.0
                    h_rand = 0.5 + 0.2 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 3:
                    h_rand = 0.5 + 0.2 * np.sin(2 * np.pi * np.arange(N) / N) + 0.1 * np.sin(4 * np.pi * np.arange(N) / N)
                elif trial == 4:
                    h_rand = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
                elif trial == 5:
                    h_rand = np.random.rand(N)
                    kernel = np.ones(7) / 7
                    h_rand = np.convolve(h_rand, kernel, mode='same')
                    h_rand = 0.2 + 0.6 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)
                elif trial == 6:
                    h_rand = np.tile([0.4, 0.6, 0.5], N // 3 + 1)[:N]
                else:
                    h_rand = np.random.rand(N)
                
                h_rand = np.clip(h_rand, 0.0, 1.0)
                h_rand = _enforce_sum(h_rand)
                
                def obj_ms(params):
                    h = params[:N]
                    t = params[N]
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    smax = _smooth_max(overlap, alpha=200.0)
                    return t + 5000.0 * np.maximum(0.0, smax - t)**2

                x0_ms = np.concatenate([h_rand, [_score(h_rand) + 0.05]])
                bounds_ms = [(0.0, 1.0)] * N + [(0.0, 1.0)]
                
                def cons_sum(params):
                    return np.sum(params[:N]) - N/2.0

                def cons_epi(params):
                    h = params[:N]
                    t = params[N]
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    return t - np.max(overlap)

                cons_ms = [
                    {'type': 'eq', 'fun': cons_sum},
                    {'type': 'ineq', 'fun': cons_epi}
                ]

                res_ms = minimize(
                    obj_ms,
                    x0_ms,
                    method='SLSQP',
                    bounds=bounds_ms,
                    constraints=cons_ms,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
                )

                if res_ms.fun < 100.0:
                    _try_update(res_ms.x[:N])

    except Exception:
        pass

    # Phase 4: Final polish
    try:
        if time.time() < deadline - 3.0:
            def obj_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=1000.0)
                return t + 50000.0 * np.maximum(0.0, smax - t)**2

            x0_p = np.concatenate([h_final, [current_score + 0.001]])
            bounds_p = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def cons_sum_p(params):
                return np.sum(params[:N]) - N/2.0

            def cons_epi_p(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            cons_p = [
                {'type': 'eq', 'fun': cons_sum_p},
                {'type': 'ineq', 'fun': cons_epi_p}
            ]

            res_p = minimize(
                obj_polish,
                x0_p,
                method='SLSQP',
                bounds=bounds_p,
                constraints=cons_p,
                options={'maxiter': 5000, 'ftol': 1e-18, 'disp': False}
            )

            if res_p.fun < 100.0:
                _try_update(res_p.x[:N])

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
