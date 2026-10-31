import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

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

    def compute_acorr_and_subgrad(h):
        """Compute autoconvolution and subgradient of max(acorr) w.r.t. h."""
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak_val = np.max(acorr)
        peak_idx = np.argmax(acorr)
        subgrad = np.zeros(N)
        for j in range(N):
            if peak_idx == 0:
                subgrad[j] = 2.0 * h[j]
            else:
                subgrad[j] = h[(j + peak_idx) % N] + h[(j - peak_idx) % N]
        return peak_val, subgrad, acorr

    def subgradient_optimize(h_init, max_iters=20000):
        """Subgradient descent with normalization, projecting to nonnegativity."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        step_size = 0.5
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            total = np.sum(h)
            if total == 0:
                break
            
            h_norm = h / total
            peak_val, subgrad, _ = compute_acorr_and_subgrad(h_norm)
            grad_proj = subgrad - np.mean(subgrad)
            
            eta = step_size / (1.0 + 0.01 * it)
            
            h_new = h_norm - eta * grad_proj
            h_new = np.maximum(h_new, 0.0)
            
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new / s
            else:
                h_new = h_norm
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
        
        return best_h, best_m

    def spectral_subgrad_hybrid(h_init, max_iters=10000):
        """Hybrid: alternate between spectral flattening and subgradient steps."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        size = 2 * N
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if it % 3 == 0:
                H = np.fft.rfft(h, n=size)
                power = np.abs(H) ** 2
                mean_power = np.mean(power)
                threshold = mean_power * 1.3
                excess = np.maximum(power - threshold, 0)
                power_new = power - 0.8 * excess
                power_new = np.maximum(power_new, 0)
                phase = np.angle(H)
                H_new = np.sqrt(power_new) * np.exp(1j * phase)
                h_new = np.fft.irfft(H_new, n=size)[:N]
                h_new = np.maximum(h_new, 0.0)
                if np.sum(h_new) == 0:
                    h_new = h.copy()
            else:
                total = np.sum(h)
                if total == 0:
                    h_new = h.copy()
                else:
                    h_norm = h / total
                    peak_val, subgrad, _ = compute_acorr_and_subgrad(h_norm)
                    grad_proj = subgrad - np.mean(subgrad)
                    eta = 0.3 / (1.0 + 0.005 * it)
                    h_new = h_norm - eta * grad_proj
                    h_new = np.maximum(h_new, 0.0)
                    s = np.sum(h_new)
                    if s > 0:
                        h_new = h_new / s
                    else:
                        h_new = h_norm
            
            m_new = compute_metric(h_new)
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
        
        return best_h, best_m

    i_arr = np.arange(N)
    mid = N / 2.0

    initializations = [
        np.ones(N),
        0.5 * (1.0 + np.cos(np.pi * (i_arr - mid) / (N / 2.0))),
        np.exp(-2.0 * ((i_arr - mid) / (N / 4.0)) ** 2),
        1.0 - 0.3 * np.cos(2 * np.pi * i_arr / N),
        np.maximum(np.cos(np.pi * (i_arr - mid) / N), 0),
    ]

    best_h = None
    best_m = float('inf')

    for init in initializations:
        if time.time() > deadline:
            break
        init = np.maximum(init, 0.0)
        h_opt, m_opt = subgradient_optimize(init, max_iters=15000)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt

    if best_h is not None and time.time() < deadline - 15:
        h_ref, m_ref = spectral_subgrad_hybrid(best_h, max_iters=8000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref

    if best_h is not None and time.time() < deadline - 5:
        h_more, m_more = subgradient_optimize(best_h, max_iters=10000)
        if m_more < best_m:
            best_m = m_more
            best_h = h_more

    return list(best_h)

# EVOLVE-BLOCK-END
