"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.01):
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def smooth_max_adaptive(x, tau_base=0.05, tau_min=0.001):
        x_max = np.max(x)
        x_spread = np.max(x) - np.min(x)
        if x_spread < 1e-6:
            tau = tau_min
        else:
            tau = max(tau_min, min(tau_base, 0.1 * x_spread))
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    # Zadoff-Chu sequence of length N (not necessarily prime, use general construction)
    def zadoff_chu(n, root=1):
        """Generate a Zadoff-Chu-like sequence of length n."""
        t = np.arange(n)
        # For general n, use a quadratic phase (Cauchy sequence)
        # z[t] = exp(j * pi * root * t * (t+1) / n)
        phase = np.pi * root * t * (t + 1) / n
        return np.exp(1j * phase)

    # Generate multiple Zadoff-Chu-like sequences with different roots
    def make_zc_h(root, window='hann', amp=1.0):
        """Create h from a Zadoff-Chu sequence with windowing."""
        zc = zadoff_chu(N, root)
        # Take real part (or use both real and imaginary as two sequences)
        h = amp * np.real(zc)
        if window == 'hann':
            w = np.hanning(N)
            h = h * w
        # Ensure sum is positive and reasonable
        s = np.sum(h)
        if abs(s) < 1e-6:
            h = h + 0.1
        return h

    def make_zc_h2(amp1, amp2, root1, root2, phase_shift, window='hann'):
        """Sum of two Zadoff-Chu sequences with different roots and a phase shift."""
        zc1 = zadoff_chu(N, root1)
        zc2 = zadoff_chu(N, root2)
        # Apply phase shift to second
        t = np.arange(N)
        zc2_shifted = zc2 * np.exp(1j * phase_shift * t / N)
        h = amp1 * np.real(zc1) + amp2 * np.real(zc2_shifted)
        if window == 'hann':
            w = np.hanning(N)
            h = h * w
        s = np.sum(h)
        if abs(s) < 1e-6:
            h = h + 0.1
        return h

    def objective_signed(params, tau=0.05):
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max_adaptive(conv, tau_base=tau, tau_min=0.001)
        return 2.0 * N * sm / (s * s)

    def dft_based_objective(params, tau=0.05):
        dc = params[0]
        h = np.full(N, dc)
        n_harm = (len(params) - 1) // 2
        for k in range(1, n_harm + 1):
            r = params[1 + 2*(k-1)]
            p = params[2 + 2*(k-1)]
            h += r * np.cos(2*np.pi*k*np.arange(N)/N + p)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max_adaptive(conv, tau_base=tau, tau_min=0.001)
        return 2.0 * N * sm / (s * s)

    # Epigraph formulation: minimize t subject to t >= 2*N*conv[k]/s^2 for all k
    # This is equivalent but can be handled with smooth constraints
    def epigraph_objective(params, tau=0.05):
        """Epigraph-style: use log-sum-exp as smooth upper bound on max."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        # We want to minimize max(conv), so use smooth max
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Build DCT-II basis
    i_idx = np.arange(N, dtype=float)
    k_idx = np.arange(N, dtype=float)
    dct_basis = np.cos(np.pi * np.outer(k_idx, (2*i_idx + 1)) / (2*N))
    dct_norm = np.sqrt(np.sum(dct_basis**2, axis=0))
    dct_basis = dct_basis / dct_norm

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 0: Try Zadoff-Chu based initializations
    if time.time() < deadline - 120.0:
        zc_configs = [
            # (amp1, amp2, root1, root2, phase_shift, window)
            (1.0, 0.0, 1, 1, 0.0, 'hann'),
            (1.0, 0.0, 2, 2, 0.0, 'hann'),
            (1.0, 0.0, 3, 3, 0.0, 'hann'),
            (1.0, 1.0, 1, 2, 0.0, 'hann'),
            (1.0, 1.0, 1, 3, 0.5, 'hann'),
            (1.0, 0.5, 2, 3, 0.3, 'hann'),
            (1.0, 0.0, 1, 1, 0.0, None),
            (1.0, 1.0, 1, 2, 1.0, None),
            (0.5, 0.5, 1, 3, 0.7, 'hann'),
            (1.0, 0.8, 2, 5, 0.2, 'hann'),
        ]
        for cfg in zc_configs:
            if time.time() > deadline - 115.0:
                break
            try:
                if len(cfg) == 6:
                    h_zc = make_zc_h2(*cfg)
                else:
                    h_zc = make_zc_h(cfg[0], cfg[1])
                val_zc = compute_value(h_zc)
                if val_zc < best_val and np.all(np.isfinite(h_zc)) and abs(np.sum(h_zc)) > 1e-6:
                    best_h = h_zc.copy()
                    best_val = val_zc
                    print(f"ZC init: improved to {best_val:.6f}", file=sys.stderr)
            except Exception:
                pass

    # Phase 1: DFT-based global search via differential evolution
    if time.time() < deadline - 80.0:
        try:
            from scipy.optimize import differential_evolution
            n_harm = 12
            bounds = [(-2.0, 2.0)]
            for k in range(1, n_harm + 1):
                bounds.append((-1.5, 1.5))
                bounds.append((0.0, 2.0 * np.pi))
            de_result = differential_evolution(
                dft_based_objective,
                bounds,
                maxiter=35,
                popsize=12,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                dc = de_result.x[0]
                h_dft = np.full(N, dc)
                for k in range(1, n_harm + 1):
                    r = de_result.x[1 + 2*(k-1)]
                    p = de_result.x[2 + 2*(k-1)]
                    h_dft += r * np.cos(2*np.pi*k*np.arange(N)/N + p)
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # Phase 2: DCT-spectral shaping with multiple restarts
    if time.time() < deadline - 50.0:
        try:
            from scipy.optimize import minimize
            rng = np.random.RandomState(123)
            for restart in range(8):
                if time.time() > deadline - 40.0:
                    break
                if restart == 0:
                    c0 = dct_basis[0] @ best_h
                    c_init = np.zeros(N)
                    c_init[0] = c0
                elif restart == 1:
                    c_init = dct_basis @ best_h
                elif restart == 2:
                    # Low-frequency with negative bias to push conv negative
                    c_init = np.zeros(N)
                    c_init[0] = 1.0
                    for k in range(1, 5):
                        c_init[k] = -0.3 * rng.uniform(0.5, 1.0)
                elif restart == 3:
                    # Mid-frequency emphasis
                    c_init = np.zeros(N)
                    c_init[0] = 0.8
                    for k in range(3, 15):
                        c_init[k] = rng.uniform(-0.4, 0.4)
                elif restart == 4:
                    # From ZC-like structure
                    zc = zadoff_chu(N, 1)
                    h_zc_init = np.real(zc) * np.hanning(N)
                    c_init = dct_basis @ h_zc_init
                    c_init[0] = max(c_init[0], 0.5)
                else:
                    c_init = np.zeros(N)
                    n_low = rng.randint(5, 20)
                    for k in range(1, n_low):
                        c_init[k] = rng.uniform(-0.5, 0.5)
                    c_init[0] = rng.uniform(0.3, 2.0)
                
                def dct_objective(c, tau=0.04):
                    h = dct_basis.T @ c
                    s = np.sum(h)
                    if abs(s) < 1e-4:
                        return 1e10 + (1e-4 - abs(s)) * 1e6
                    conv = np.convolve(h, h)
                    sm = smooth_max_adaptive(conv, tau_base=tau, tau_min=0.001)
                    return 2.0 * N * sm / (s * s)
                
                result = minimize(
                    dct_objective,
                    c_init,
                    method='L-BFGS-B',
                    options={'maxiter': 350, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
                )
                h_opt = dct_basis.T @ result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"DCT restart {restart}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DCT optimization error: {e}", file=sys.stderr)

    # Phase 3: Direct L-BFGS-B refinement in h-space
    if time.time() < deadline - 25.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Direct L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 4: Progressive tau tightening polish
    if time.time() < deadline - 12.0 and best_val < 2.0:
        try:
            from scipy.optimize import minimize
            for tau_pass in [0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 6.0:
                    break
                def objective_tight(params, tau=tau_pass):
                    h = params
                    s = np.sum(h)
                    if abs(s) < 1e-4:
                        return 1e10 + (1e-4 - abs(s)) * 1e6
                    conv = np.convolve(h, h)
                    sm = smooth_max(conv, tau=tau)
                    return 2.0 * N * sm / (s * s)
                result = minimize(
                    objective_tight,
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Polish tau={tau_pass}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Polish error: {e}", file=sys.stderr)

    # Phase 5: Subgradient-based exact polish
    # Directly minimize the true nonsmooth objective using subgradient descent
    if time.time() < deadline - 5.0 and best_val < 2.0:
        try:
            h_cur = best_h.copy()
            s_cur = np.sum(h_cur)
            conv_cur = np.convolve(h_cur, h_cur)
            max_idx = np.argmax(conv_cur)
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 3.0:
                    break
                h_cur = h_cur.copy()
                s = np.sum(h_cur)
                if abs(s) < 1e-6:
                    break
                conv = np.convolve(h_cur, h_cur)
                max_val = np.max(conv)
                val = abs(2.0 * N * max_val / (s * s))
                
                if val < best_val:
                    best_h = h_cur.copy()
                    best_val = val
                
                # Subgradient of max(conv) w.r.t. h: it's h shifted by max_idx
                # conv[k] = sum_i h[i]*h[i+k], d(conv[max_idx])/d(h[j]) = h[j+max_idx] + h[j-max_idx] (with bounds)
                # For the absolute value, sign matters
                sign = np.sign(max_val) if max_val != 0 else 1.0
                # Gradient of 2*N*max(conv)/s^2
                # = 2*N * (d(max(conv))/s^2 - 2*max(conv)*s*(ds/dh)/s^4)
                # = 2*N * (d(max(conv))/s^2 - 2*max(conv)*ds/dh/s^3)
                ds_dh = np.ones(N)
                dmax_dh = np.zeros(N)
                # d(conv[max_idx])/d(h[j]) = sum over valid i: h[i] where i+max_idx=j or i-max_idx=j
                # conv[max_idx] = sum_{i=0}^{N-1-max_idx} h[i]*h[i+max_idx]
                # d/dh[j] = h[j+max_idx] (if j+max_idx < N) + h[j-max_idx] (if j-max_idx >= 0)
                for j in range(N):
                    if j + max_idx < N:
                        dmax_dh[j] += h_cur[j + max_idx]
                    if j - max_idx >= 0:
                        dmax_dh[j] += h_cur[j - max_idx]
                
                grad = 2.0 * N * (sign * dmax_dh / (s * s) - 2.0 * sign * max_val * ds_dh / (s * s * s))
                
                # Adaptive learning rate
                norm_grad = np.linalg.norm(grad)
                if norm_grad > 1e-10:
                    h_cur -= lr * grad / norm_grad * 0.1
                else:
                    break
                
                # Keep h finite
                if not np.all(np.isfinite(h_cur)):
                    h_cur = best_h.copy()
                    break
                
                # Decay learning rate
                lr *= 0.995
        except Exception as e:
            print(f"Subgradient error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
