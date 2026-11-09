"""Improved program: Projected Gradient Descent on continuous relaxation of signed step functions."""
import numpy as np
import time

N = 400

def compute_autocorrelation_bound(f):
    """Compute an upper bound on C3 based on autocorrelation sidelobes."""
    # Linear autocorrelation via FFT
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf_full = np.fft.irfft(F * np.conj(F), n=nfft)
    # Linear autocorrelation: acf[k] = sum_{i=0}^{N-1-k} f[i]*f[i+k] for k >= 0
    acf = np.zeros(2 * N - 1)
    acf[:N] = acf_full[:N]
    acf[N:] = acf_full[nfft - N + 1:nfft][::-1]
    
    # The "third autocorrelation inequality" upper bound on C3
    # Use the normalized sidelobe energy as the metric
    peak = acf[0]  # This is sum(f^2) = N for ±1 signals
    if abs(peak) < 1e-12:
        return np.sqrt(np.sum(acf[1:] ** 2))
    sidelobe_energy = np.sum(acf[1:] ** 2)
    return np.sqrt(sidelobe_energy / (peak * peak))

def construct():
    start_time = time.time()
    
    # Generate an m-sequence using a Linear Feedback Shift Register (LFSR).
    # m-sequences have optimal periodic autocorrelation properties (sidelobes = -1).
    # Using a segment of an m-sequence yields very low linear autocorrelation sidelobes.
    
    # We use a 9-bit LFSR with primitive polynomial x^9 + x^4 + 1.
    # This generates a sequence of period 511.
    # We take the first N=400 samples.
    
    # Polynomial x^9 + x^4 + 1 corresponds to taps at bit 9 and bit 4.
    # In a 0-indexed 9-bit register (bits 8..0):
    # Recurrence: x^9 = x^4 + x^0
    # Feedback bit (new MSB) = current bit 4 XOR current bit 0.
    
    state = 1  # Initial state (non-zero)
    f = np.zeros(N)
    
    for i in range(N):
        # Output LSB (bit 0)
        bit = state & 1
        f[i] = 1.0 if bit == 0 else -1.0
        
        # Compute feedback: x^4 XOR x^0
        # Bit 4 is (state >> 4) & 1
        # Bit 0 is state & 1
        feedback = ((state >> 4) & 1) ^ (state & 1)
        
        # Shift register right by 1, insert feedback at MSB (bit 8)
        state = (state >> 1) | (feedback << 8)
        
        # Time check (though this loop is extremely fast, < 1ms)
        if time.time() > start_time + 110.0:
            break

    # Fallback if something went wrong (e.g., all zeros)
    if np.all(f == 1.0) or np.all(f == -1.0):
        # Generate random sequence
        rng = np.random.default_rng(42)
        f = rng.choice([-1.0, 1.0], size=N)
        
    return f
