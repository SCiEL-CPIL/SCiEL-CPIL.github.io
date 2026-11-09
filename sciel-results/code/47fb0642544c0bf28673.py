"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute max absolute autocorrelation sidelobe using FFT."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # r[0] is DC (lag 0), r[1:N] are lags 1..N-1
    # For periodic autocorrelation, lags > N/2 mirror, but we take max over all non-zero lags
    return np.max(np.abs(r[1:N]))


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)
    
    current_peak = _autocorr_peak(h)
    best_peak = current_peak
    best_h = h.copy()
    
    # Local search: flip bits that reduce peak autocorrelation
    # Use iterative improvement with multiple passes
    max_passes = 50
    pass_count = 0
    
    while time.time() < deadline - 5.0 and pass_count < max_passes:
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            # Flip bit i
            h[i] = -h[i]
            new_peak = _autocorr_peak(h)
            
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
            else:
                # Revert
                h[i] = -h[i]
        
        pass_count += 1
        if not improved:
            # No improvement in this pass, try a few random restarts
            if time.time() > deadline - 10.0:
                break
            for restart in range(3):
                if time.time() > deadline - 10.0:
                    break
                h = rng.choice([-1.0, 1.0], size=N)
                current_peak = _autocorr_peak(h)
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
                # Quick local search on new random
                for _ in range(2):
                    for i in range(N):
                        if time.time() > deadline - 10.0:
                            break
                        h[i] = -h[i]
                        new_peak = _autocorr_peak(h)
                        if new_peak < current_peak:
                            current_peak = new_peak
                            if current_peak < best_peak:
                                best_peak = current_peak
                                best_h = h.copy()
                        else:
                            h[i] = -h[i]
    
    return best_h
# EVOLVE-BLOCK-END
