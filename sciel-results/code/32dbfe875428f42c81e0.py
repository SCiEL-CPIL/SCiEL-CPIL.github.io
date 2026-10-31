# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    N = 400
    n_fft = 2 * N
    
    def autoconv_energy(heights):
        """Compute the L2 norm of the autoconvolution (as a proxy for C3)."""
        f_ext = np.zeros(n_fft)
        f_ext[:N] = heights
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        # Minimize sum of squares of non-zero lags
        return np.sum(autocorr[1:N] ** 2)
    
    def full_local_search(h, max_iters=10):
        """Perform deterministic full local search (flip every index if it improves)."""
        current_energy = autoconv_energy(h)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                h[i] *= -1
                new_energy = autoconv_energy(h)
                if new_energy < current_energy - 1e-10:
                    current_energy = new_energy
                    improved = True
                else:
                    h[i] *= -1
        return h, current_energy

    rng = np.random.RandomState(42)
    
    best_result = None
    best_score = float('inf')
    
    num_restarts = 200
    max_sa_steps = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize with smooth, band-limited signal (low freq sines)
        # This creates blocks of +1 and -1, potentially better than white noise
        # for minimizing autocorrelation tails.
        
        # Generate a random low-frequency signal
        # Use 3-5 random low frequencies
        num_freqs = rng.randint(2, 6)
        freqs = rng.uniform(1, 10, size=num_freqs) # Low frequencies
        phases = rng.uniform(0, 2 * np.pi, size=num_freqs)
        amplitudes = rng.uniform(0.5, 1.5, size=num_freqs)
        
        t = np.linspace(0, 1, N)
        signal = np.zeros(N)
        for k in range(num_freqs):
            signal += amplitudes[k] * np.sin(2 * np.pi * freqs[k] * t + phases[k])
        
        # Threshold to binary {-1, 1}
        h = np.sign(signal)
        # Handle zeros (unlikely with random phases/amps)
        h[h == 0] = 1.0
        
        current_energy = autoconv_energy(h)
        
        if current_energy < best_score:
            best_score = current_energy
            best_result = h.copy()
            
        # Simulated Annealing
        T = 100.0
        T_min = 1.0
        cooling_rate = 0.999
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 10:
                break
                
            i = rng.randint(N)
            h[i] *= -1
            new_energy = autoconv_energy(h)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or (T > 0 and rng.rand() < np.exp(-delta_e / T)):
                current_energy = new_energy
                if current_energy < best_score:
                    best_score = current_energy
                    best_result = h.copy()
            else:
                h[i] *= -1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Polish the current best from this restart
        if time.time() < deadline - 5:
            h_polished, energy_polished = full_local_search(h.copy(), max_iters=5)
            if energy_polished < best_score:
                best_score = energy_polished
                best_result = h_polished.copy()
                
    # Final polish on the global best
    if best_result is not None and time.time() < deadline - 2:
        h_final, energy_final = full_local_search(best_result.copy(), max_iters=20)
        if energy_final < best_score:
            best_score = energy_final
            best_result = h_final.copy()
            
    if best_result is None:
        best_result = np.ones(N)
        
    return best_result
# EVOLVE-BLOCK-END
