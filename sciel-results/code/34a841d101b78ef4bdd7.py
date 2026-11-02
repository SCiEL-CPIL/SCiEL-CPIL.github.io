"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Approach: Simulated Annealing to minimize Peak Sidelobe Level (PSL)
    # The previous approach minimized Energy (L2 norm), which likely resulted in
    # sequences with high peak sidelobes, leading to a poor score if the metric
    # is PSL (L-infinity norm). We switch to minimizing PSL directly.
    # We use FFT to compute autocorrelation efficiently.
    
    np.random.seed(42) # Fixed seed for reproducibility
    f = np.random.choice([-1, 1], size=N)
    
    # Pre-compute FFT size for zero-padding to avoid circular convolution issues
    # We need aperiodic autocorrelation. Zero-padding to at least 2N-1 is required.
    # Next power of 2 >= 2N-1 is efficient.
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
        
    def calculate_psl(seq):
        """
        Calculate the Peak Sidelobe Level (PSL) of the sequence.
        PSL = max(|R(k)|) for k != 0.
        """
        # Zero-pad the sequence
        f_padded = np.zeros(fft_size)
        f_padded[:N] = seq
        
        # Compute FFT
        F = np.fft.fft(f_padded)
        
        # Compute Power Spectrum
        power = np.abs(F)**2
        
        # Inverse FFT to get Autocorrelation
        ac = np.fft.ifft(power).real
        
        # ac[0] is R(0) = N
        # The aperiodic autocorrelation R(k) for k>0 is in ac[k]
        # For k<0, it is in ac[fft_size + k] which is symmetric to ac[-k]
        # We only need to check the first N-1 elements (k=1 to N-1)
        # Due to symmetry, max(|R(k)|) for k!=0 is max(ac[1:N], ac[fft_size-(N-1):fft_size])
        # But ac is real and symmetric. ac[k] = ac[fft_size-k].
        # So we just take max of ac[1:N]
        
        # Note: ac[0] is N. We exclude it.
        sidelobes = ac[1:N]
        return np.max(np.abs(sidelobes))

    current_psl = calculate_psl(f)
    best_f = f.copy()
    best_psl = current_psl
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Simulated Annealing Parameters
    # PSL is an integer (since f is +/-1, R(k) is integer).
    # Initial T should be large enough to allow uphill moves.
    # Max possible PSL is N=400.
    T = 20.0 
    T_min = 0.1
    # Estimate iterations with FFT:
    # FFT of size 1024 is very fast, ~10-20 us.
    # Overhead ~ 50 us per iteration.
    # 110s / 50us = 2,200,000 iterations.
    # Cooling rate to reach T_min:
    # T * r^I = T_min
    # r = (T_min/T)^(1/I)
    I_est = 2000000
    cooling_rate = (T_min / T) ** (1.0 / I_est)
    
    while (time.time() - start_time) < max_time:
        # Flip a random bit
        i = np.random.randint(N)
        f[i] = -f[i]
        
        new_psl = calculate_psl(f)
        delta = new_psl - current_psl
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_f = f.copy()
        else:
            # Revert
            f[i] = -f[i]
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_f
# EVOLVE-BLOCK-END
