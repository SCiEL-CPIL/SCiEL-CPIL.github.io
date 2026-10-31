import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_metric(h):
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def build_h_from_params(params):
        """
        Parametric representation: piecewise-constant heights via smooth basis.
        params: [log_amplitude, phase, freq, asymmetry, width]
        Produces a nonnegative step function on N intervals.
        """
        log_amp, phase, freq, asym, width = params
        i = np.arange(N, dtype=np.float64)
        mid = (N - 1) / 2.0
        x = (i - mid) / (N / 2.0)  # in [-1, 1]
        
        # Base shape: raised cosine modulated by frequency
        base = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * x + phase)
        # Asymmetric modulation
        asym_mod = 1.0 + asym * np.sin(np.pi * x)
        # Width control via soft threshold
        w = np.exp(-width * x * x)
        
        h = np.exp(log_amp) * base * asym_mod * w
        h = np.maximum(h, 0.0)
        return h

    def objective(params):
        h = build_h_from_params(params)
        total = np.sum(h)
        if total == 0:
            return 1e10
        return compute_metric(h)

    def optimize_from_params(params0, max_time=None):
        """Run Nelder-Mead on the parametric representation."""
        if max_time is None:
            max_time = deadline - time.time() - 2.0
        if max_time < 2.0:
            return params0.copy(), objective(params0)
        
        best_params = params0.copy()
        best_val = objective(params0)
        
        # Use scipy's Nelder-Mead with time-based termination
        def time_check():
            return time.time() > deadline - 3.0
        
        result = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True,
            },
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x.copy()
        
        return best_params, best_val

    def local_refine(h_init, max_time=None):
        """Direct coordinate descent / projected gradient on h itself."""
        if max_time is None:
            max_time = deadline - time.time() - 2.0
        if max_time < 1.0:
            return h_init, compute_metric(h_init)
        
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        size = 2 * N
        
        lr = 0.02
        for it in range(5000):
            if time.time() > deadline - 3.0:
                break
            if time.time() > start_time + max_time:
                break
            
            total = np.sum(h)
            if total == 0:
                break
            h_norm = h / total
            
            H = np.fft.rfft(h_norm, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            peak_idx = np.argmax(acorr)
            
            # Subgradient of max acorr
            subgrad = np.zeros(N)
            for j in range(N):
                if peak_idx == 0:
                    subgrad[j] = 2.0 * h_norm[j]
                else:
                    subgrad[j] = h_norm[(j + peak_idx) % N] + h_norm[(j - peak_idx) % N]
            
            grad_proj = subgrad - np.mean(subgrad)
            h_new = h_norm - lr * grad_proj
            h_new = np.maximum(h_new, 0.0)
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new / s
            
            m_new = compute_metric(h_new)
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
            lr *= 0.999
        
        return best_h, best_m

    # Multiple parametric initializations
    param_inits = [
        [0.0, 0.0, 1.0, 0.0, 1.0],      # symmetric raised cosine
        [0.0, 0.0, 2.0, 0.0, 2.0],      # higher freq
        [0.0, np.pi/2, 1.0, 0.3, 1.5],  # shifted, asymmetric
        [0.0, 0.0, 1.5, -0.2, 1.0],     # different freq, slight asym
        [0.0, np.pi, 1.0, 0.0, 0.8],    # inverted cosine
        [0.0, 0.0, 3.0, 0.1, 2.5],      # high freq, narrow
        [0.0, np.pi/4, 0.5, 0.4, 1.2],  # low freq, strong asym
        [0.0, 0.0, 1.0, 0.0, 3.0],      # very narrow
    ]

    best_h = None
    best_m = float('inf')

    for pi_init in param_inits:
        if time.time() > deadline - 10:
            break
        
        h_cand = build_h_from_params(pi_init)
        m_cand = compute_metric(h_cand)
        
        if m_cand < best_m:
            best_m = m_cand
            best_h = h_cand.copy()
        
        # Optimize in parameter space
        if time.time() > deadline - 15:
            break
        p_opt, m_opt = optimize_from_params(np.array(pi_init, dtype=np.float64))
        h_opt = build_h_from_params(p_opt)
        m_opt = compute_metric(h_opt)
        
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt.copy()

    # Local refinement in direct space
    if best_h is not None and time.time() < deadline - 8:
        h_ref, m_ref = local_refine(best_h)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref

    # Final parametric polish from best found
    if best_h is not None and time.time() < deadline - 5:
        # Try to fit parametric form to best_h via a few Nelder-Mead restarts
        for trial in range(3):
            if time.time() > deadline - 4:
                break
            # Perturb a random parametric init
            p_rand = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0.5, 3.0),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.5, 4.0),
            ])
            p_opt, m_opt = optimize_from_params(p_rand)
            h_opt = build_h_from_params(p_opt)
            m_opt = compute_metric(h_opt)
            if m_opt < best_m:
                best_m = m_opt
                best_h = h_opt.copy()

    if best_h is None:
        best_h = np.ones(N)

    return list(best_h)

# EVOLVE-BLOCK-END
