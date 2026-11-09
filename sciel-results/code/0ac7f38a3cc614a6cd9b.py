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
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_f = None
    best_bound = np.inf
    
    # Initialization: Smooth windowed sinusoids
    t = np.arange(N)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * t / (N - 1))
    
    # Try several initial frequencies and random phases
    for freq in [1, 2, 5, 10, 20, 50]:
        for phase in [0, np.pi/2, np.pi]:
            f_init = window * np.cos(2 * np.pi * freq * t / N + phase)
            # Project to ±1
            f_proj = np.sign(f_init)
            # Handle zeros
            zeros = f_proj == 0
            if np.any(zeros):
                f_proj[zeros] = rng.choice([-1.0, 1.0], size=np.sum(zeros))
            
            bound = compute_autocorrelation_bound(f_proj)
            if bound < best_bound:
                best_bound = bound
                best_f = f_proj.copy()
                best_f_cont = f_init.copy()

    if best_f is None:
        # Fallback to random
        best_f = rng.choice([-1.0, 1.0], size=N)
        best_f_cont = best_f.astype(float)
        best_bound = compute_autocorrelation_bound(best_f)

    # Projected Gradient Descent
    f = best_f.copy()
    f_cont = best_f_cont.copy()
    current_bound = best_bound
    eta = 0.05
    nfft = 1 << (2 * N - 1).bit_length()
    
    # Run PGD until deadline
    while time.time() < deadline:
        # Compute Gradient
        # J(f) = sum_{k=1}^{N-1} R[k]^2
        # grad = 2 * IFFT(FFT(R) * FFT(f))
        # FFT(R) = F * conj(F)
        # Grad_fft = (F * conj(F)) * F
        F = np.fft.rfft(f, n=nfft)
        Grad_fft = (F * np.conj(F)) * F
        grad = np.fft.irfft(Grad_fft, n=nfft)[:N]
        grad *= 2.0
        
        # Update continuous signal
        f_cont -= eta * grad
        
        # Project to ±1
        f_new = np.sign(f_cont)
        zeros = f_new == 0
        if np.any(zeros):
            f_new[zeros] = rng.choice([-1.0, 1.0], size=np.sum(zeros))
        
        # Evaluate
        new_bound = compute_autocorrelation_bound(f_new)
        
        if new_bound < current_bound:
            current_bound = new_bound
            f = f_new
            # Decay learning rate slightly if improving
            eta *= 0.999
        else:
            # If not improving, reduce learning rate to stabilize
            eta *= 0.95
            # Optional: If eta becomes too small, restart with a new random initialization?
            # For now, just continue.
            
        # Safety check: if eta is very small, we might be stuck.
        # But the time limit will stop us.
        
    return f
