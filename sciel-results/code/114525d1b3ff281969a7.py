"""Improved program: Fast local search on signed step functions using spectral updates."""
N = 400
import numpy as np
import time

def compute_autocorrelation_bound(f):
    """Compute an upper bound on C3 based on autocorrelation sidelobes."""
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    # Try: ratio of sidelobe energy to peak energy
    if abs(acf[0]) > 1e-12:
        sidelobe_energy = np.sum(acf[1:] ** 2)
        peak_energy = acf[0] ** 2
        bound = np.sqrt(sidelobe_energy / peak_energy)
    else:
        bound = np.sqrt(np.sum(acf[1:] ** 2))
    return bound

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    nfft = 1 << (2 * N - 1).bit_length()
    
    # Precompute phases for fast spectral updates
    # F has length nfft//2 + 1
    # We need e^{-i 2 pi m k / nfft} for m=0..nfft//2, k=0..N-1
    m_vals = np.arange(nfft // 2 + 1)[:, None]
    k_vals = np.arange(N)[None, :]
    phases = np.exp(-1j * 2 * np.pi * m_vals * k_vals / nfft)
    
    # Generate initial candidates
    # 1. M-sequence (LFSR)
    def lfsr_sequence(degree, poly_taps, length):
        state = [1] * degree
        seq = []
        for _ in range(length):
            seq.append(state[-1])
            fb = 0
            for tap in poly_taps:
                if tap == degree:
                    continue
                if tap == 0:
                    fb ^= state[0]
                else:
                    fb ^= state[tap - 1]
            state = state[:-1] + [fb]
        return seq

    # Primitive polynomials for degree 9
    taps1 = [9, 4, 0]
    seq1 = lfsr_sequence(9, taps1, 511)
    f1 = np.array([1 if b == 1 else -1 for b in seq1[:N]])
    
    # Random candidate
    f2 = np.random.choice([-1, 1], size=N)
    
    # Sinc-like / Windowed (quantized)
    t = np.linspace(-1, 1, N)
    f3 = np.sign(np.sin(np.pi * t) / (np.pi * t + 1e-9))
    # Ensure no zeros
    f3[f3 == 0] = 1
    
    candidates = [f1, f2, f3]
    
    best_f = None
    best_score = float('inf')
    
    for f_init in candidates:
        if time.time() > deadline:
            break
            
        f = f_init.copy()
        F = np.fft.rfft(f, n=nfft)
        
        # Calculate initial cost (sum of |F|^4)
        cost = np.sum(np.abs(F) ** 4)
        
        # Local search loop
        # We will do hill climbing with random restarts if stuck
        # To keep it simple, just do greedy hill climbing with random perturbations if no improvement for a while
        
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < deadline:
            # Pick random index to flip
            k = np.random.randint(N)
            
            # Calculate delta F
            # f[k] is either 1 or -1
            # New f[k] = -f[k]
            # Delta = -2 * f[k] * phase[:, k]
            delta_F = -2.0 * f[k] * phases[:, k]
            F_cand = F + delta_F
            
            # Calculate new cost
            cost_cand = np.sum(np.abs(F_cand) ** 4)
            
            if cost_cand < cost:
                # Accept
                F = F_cand
                f[k] = -f[k]
                cost = cost_cand
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck for too long, do a random restart or larger perturbation
                if no_improve_count > max_no_improve:
                    # Break out of this local search and try another initial candidate
                    # Or just break and move to next candidate
                    break
            # Check time periodically
            if time.time() > deadline:
                break
        
        # Evaluate final score using the original metric
        score = compute_autocorrelation_bound(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
            
    if best_f is None:
        # Fallback
        best_f = np.ones(N)
        
    return best_f
