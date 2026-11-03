"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Use a local search to minimize the energy of the autocorrelation sidelobes.
    # This approach targets a "flat" spectrum, reducing the peak sidelobe levels
    # which are often the dominant factor in the C3' metric for non-trivial sequences.
    
    # Initialize with random +/- 1 sequence
    x = np.random.choice([1.0, -1.0], size=N)
    
    # Helper function to compute the sum of squared sidelobes
    def compute_sidelobe_energy(seq):
        # Zero-pad to 2N to compute linear autocorrelation via FFT
        padded = np.zeros(2 * N)
        padded[:N] = seq
        X = np.fft.fft(padded)
        C = np.fft.ifft(X * np.conj(X)).real
        # C[:N] contains lags 0 to N-1
        # C[0] is the main lobe (total energy), C[1:] are sidelobes
        return np.sum(C[1:N] ** 2)
    
    current_energy = compute_sidelobe_energy(x)
    
    # Perform local search by flipping random bits
    # 10,000 iterations is fast enough for N=400
    for _ in range(10000):
        i = np.random.randint(N)
        old_val = x[i]
        x[i] = -old_val
        
        new_energy = compute_sidelobe_energy(x)
        
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            x[i] = old_val
            
    return x
# EVOLVE-BLOCK-END
