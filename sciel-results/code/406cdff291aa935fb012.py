import numpy as np
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 148.0
    N = 24
    k_arr = np.arange(N)
    
    # Precompute roots of unity for fast evaluation
    FFT_SIZE = 16384
    theta_grid = 2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE
    roots = np.exp(1j * theta_grid)
    
    def grs(n):
        a = np.zeros(n)
        a[0] = 1.0
        for i in range(1, n):
            kk = i >> 1
            if i & 1:
                a[i] = -a[kk]
            else:
                a[i] = a[kk]
        return a
    
    def eval_sup_fast(c):
        """Evaluate max |P(e^{i theta})| using FFT-based dense sampling."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.max(np.abs(F)))
    
    def eval_lnorm(c, p=64):
        """Smooth Lp-norm surrogate: (sum |P|^p)^(1/p). Cheaper than full peak search."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.sum(np.abs(F)**p) ** (1.0/p))
    
    def eval_at_theta(c, theta):
        """Evaluate P at a specific theta."""
        return float(np.abs(np.dot(c, np.exp(1j * theta * k_arr))))
    
    def parabolic_refine(c, idx):
        """Refine peak near FFT index idx using parabolic interpolation + Newton."""
        dtheta = 2 * np.pi / FFT_SIZE
        # Get values at idx-1, idx, idx+1
        f_im1 = eval_at_theta(c, 2*np.pi*(idx-1)/FFT_SIZE)
        f_i   = eval_at_theta(c, 2*np.pi*idx/FFT_SIZE)
        f_ip1 = eval_at_theta(c, 2*np.pi*(idx+1)/FFT_SIZE)
        
        # Parabolic vertex
        denom = (f_im1 - 2*f_i + f_ip1)
        if abs(denom) > 1e-15:
            offset = 0.5 * (f_im1 - f_ip1) / denom
        else:
            offset = 0.0
        offset = max(-0.5, min(0.5, offset))
        theta = 2*np.pi*(idx + offset)/FFT_SIZE
        
        # Newton refinement: find where derivative of |P|^2 is zero
        # d/dtheta |P|^2 = 2 Re(conj(P) * P')
        # P' = sum c_k * k * i * exp(i*k*theta)
        for _ in range(8):
            z = np.exp(1j * theta)
            P_val = np.dot(c, z**k_arr)
            P_prime = np.dot(c * k_arr, 1j * z**k_arr)
            # gradient of |P|^2
            grad = 2.0 * np.real(np.conj(P_val) * P_prime)
            if abs(grad) < 1e-14:
                break
            # second derivative approximation
            h = 1e-5
            z_h = np.exp(1j * (theta + h))
            P_h = np.dot(c, z_h**k_arr)
            Pp_h = np.dot(c * k_arr, 1j * z_h**k_arr)
            grad_h = 2.0 * np.real(np.conj(P_h) * Pp_h)
            z_mh = np.exp(1j * (theta - h))
            P_mh = np.dot(c, z_mh**k_arr)
            Pp_mh = np.dot(c * k_arr, 1j * z_mh**k_arr)
            grad_mh = 2.0 * np.real(np.conj(P_mh) * Pp_mh)
            hess = (grad_h - grad_mh) / (2*h)
            if abs(hess) < 1e-15:
                break
            step = -grad / hess
            if abs(step) > 0.1:
                step = np.sign(step) * 0.1
            theta += step
            theta = theta % (2*np.pi)
        
        return float(np.abs(np.dot(c, np.exp(1j * theta * k_arr))))
    
    def find_peak_precise(c):
        """Find the true sup-norm peak using FFT grid + parabolic/Newton refinement."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        
        # Find top 5 peaks
        top_indices = np.argsort(abs_F)[::-1][:5]
        
        # Cluster: keep only distinct peaks (separated by > FFT_SIZE//200)
        peaks = []
        for idx in top_indices:
            if all(abs(idx - p) > FFT_SIZE // 200 for p in peaks):
                peaks.append(idx)
            if len(peaks) >= 5:
                break
        
        best_val = 0.0
        for idx in peaks:
            val = parabolic_refine(c, idx)
            if val > best_val:
                best_val = val
        
        return best_val
    
    # Generate starting candidates
    candidates = []
    
    # GRS and variants
    g = grs(N)
    candidates.append(g)
    candidates.append(g[::-1].copy())
    candidates.append(-g)
    candidates.append(-g[::-1].copy())
    
    # Cyclic shifts of GRS
    for shift in range(1, N):
        candidates.append(np.roll(g, shift))
        candidates.append(np.roll(g, shift)[::-1].copy())
    
    # All single-bit flips of GRS
    for i in range(N):
        c = g.copy()
        c[i] = -c[i]
        candidates.append(c)
    
    # All single-bit flips of reversed GRS
    gr = g[::-1].copy()
    for i in range(N):
        c = gr.copy()
        c[i] = -c[i]
        candidates.append(c)
    
    # All double-bit flips of GRS
    for i in range(N):
        for j in range(i + 1, N):
            c = g.copy()
            c[i] = -c[i]
            c[j] = -c[j]
            candidates.append(c)
    
    # All double-bit flips of reversed GRS
    for i in range(N):
        for j in range(i + 1, N):
            c = gr.copy()
            c[i] = -c[i]
            c[j] = -c[j]
            candidates.append(c)
    
    # All triple-bit flips of GRS
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                c = g.copy()
                c[i] = -c[i]
                c[j] = -c[j]
                c[k] = -c[k]
                candidates.append(c)
    
    # Random starts with different seeds
    for seed in [42, 123, 777, 2024, 31415, 271828]:
        rng = np.random.RandomState(seed)
        for _ in range(30):
            candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Structured patterns: alternating blocks
    for block in [2, 3, 4, 6, 8, 12]:
        c = np.zeros(N)
        for i in range(N):
            c[i] = 1.0 if (i // block) % 2 == 0 else -1.0
        candidates.append(c)
        candidates.append(-c)
        candidates.append(c[::-1].copy())
    
    # Evaluate all candidates using fast FFT sup-norm
    scored = []
    for c in candidates:
        if time.time() > deadline - 60.0:
            break
        val = eval_sup_fast(c)
        scored.append((val, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates for precise peak evaluation
    top_k = min(80, len(scored))
    best_c = None
    best_val = float('inf')
    
    for val, c in scored[:top_k]:
        if time.time() > deadline - 45.0:
            break
        pv = find_peak_precise(c)
        if pv < best_val:
            best_val = pv
            best_c = c.copy()
    
    if best_c is None:
        best_c = g.copy()
        best_val = find_peak_precise(best_c)
    
    # Local search: single-bit flips using Lp-norm as smooth surrogate
    improved = True
    passes = 0
    while improved and passes < 80 and time.time() < deadline - 20.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        cur_lp = eval_lnorm(best_c, p=32)
        
        for j in range(N):
            if time.time() > deadline - 20.0:
                break
            trial = best_c.copy()
            trial[j] = -trial[j]
            quick_lp = eval_lnorm(trial, p=32)
            if quick_lp < cur_lp:
                pv = find_peak_precise(trial)
                if pv < cur - 1e-10:
                    best_c = trial
                    cur = pv
                    cur_lp = quick_lp
                    improved = True
    
    # Double-bit flips
    improved = True
    passes = 0
    while improved and passes < 15 and time.time() < deadline - 10.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        cur_lp = eval_lnorm(best_c, p=32)
        
        for j in range(N):
            for l in range(j + 1, N):
                if time.time() > deadline - 10.0:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                trial[l] = -trial[l]
                quick_lp = eval_lnorm(trial, p=32)
                if quick_lp < cur_lp:
                    pv = find_peak_precise(trial)
                    if pv < cur - 1e-10:
                        best_c = trial
                        cur = pv
                        cur_lp = quick_lp
                        improved = True
    
    # Triple-bit flips
    improved = True
    passes = 0
    while improved and passes < 5 and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        cur_lp = eval_lnorm(best_c, p=16)
        
        for j in range(N):
            for l in range(j + 1, N):
                for m in range(l + 1, N):
                    if time.time() > deadline - 5.0:
                        break
                    trial = best_c.copy()
                    trial[j] = -trial[j]
                    trial[l] = -trial[l]
                    trial[m] = -trial[m]
                    quick_lp = eval_lnorm(trial, p=16)
                    if quick_lp < cur_lp:
                        pv = find_peak_precise(trial)
                        if pv < cur - 1e-10:
                            best_c = trial
                            cur = pv
                            cur_lp = quick_lp
                            improved = True
    
    # Final high-precision polish: verify with more refinement iterations
    if time.time() < deadline - 2.0:
        # Do a final check with even finer FFT
        FFT_FINE = 32768
        c_pad = np.zeros(FFT_FINE, dtype=np.complex128)
        c_pad[:N] = best_c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        top_idx = np.argsort(abs_F)[::-1][:5]
        dtheta_fine = 2*np.pi / FFT_FINE
        for idx in top_idx:
            theta = 2*np.pi*idx/FFT_FINE
            # Newton polish
            for _ in range(10):
                z = np.exp(1j * theta)
                P_val = np.dot(best_c, z**k_arr)
                P_prime = np.dot(best_c * k_arr, 1j * z**k_arr)
                grad = 2.0 * np.real(np.conj(P_val) * P_prime)
                if abs(grad) < 1e-14:
                    break
                h = 1e-5
                z_h = np.exp(1j * (theta + h))
                P_h = np.dot(best_c, z_h**k_arr)
                Pp_h = np.dot(best_c * k_arr, 1j * z_h**k_arr)
                grad_h = 2.0 * np.real(np.conj(P_h) * Pp_h)
                z_mh = np.exp(1j * (theta - h))
                P_mh = np.dot(best_c, z_mh**k_arr)
                Pp_mh = np.dot(best_c * k_arr, 1j * z_mh**k_arr)
                grad_mh = 2.0 * np.real(np.conj(P_mh) * Pp_mh)
                hess = (grad_h - grad_mh) / (2*h)
                if abs(hess) < 1e-15:
                    break
                step = -grad / hess
                if abs(step) > 0.05:
                    step = np.sign(step) * 0.05
                theta = (theta + step) % (2*np.pi)
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out
