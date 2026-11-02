"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy:
    1. Spectral Phase Retrieval with a flat magnitude spectrum to find low-sidelobe sequences.
    2. Compare against the trivial solution (all 1s) which maximizes the integral S.
    3. Return the sequence with the lowest objective value (max |autoconv| / S^2).
    """
    
    def compute_objective(x):
        # Circular autoconvolution via FFT
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        max_c = np.max(np.abs(c))
        return max_c / (S * S)

    # Candidate 1: Trivial solution (all 1s)
    # S = N, max_c = N, obj = N / N^2 = 1/N
    x_trivial = np.ones(N)
    obj_trivial = compute_objective(x_trivial)

    # Candidate 2: Spectral Phase Retrieval with Flat Magnitude
    # This aims to minimize sidelobes, but might reduce S.
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Target magnitude: Flat
    # Sum of |X|^2 is N^2. If flat, |X_k| = sqrt(N) for all k.
    target_mag = np.sqrt(N) * np.ones(N)
    
    for _ in range(100):
        X = np.fft.fft(x)
        # Preserve phase, replace magnitude
        phase = np.angle(X)
        # Handle zeros in X carefully to avoid NaNs in angle, though np.angle(0) is 0
        X_new = target_mag * np.exp(1j * phase)
        x_new = np.fft.ifft(X_new).real
        # Project to {-1, 1}
        x = np.sign(x_new)
        # Ensure no zeros (sign(0) is 0, replace with 1 or random)
        x[x == 0] = 1.0
        
    obj_spectral = compute_objective(x)

    # Select the better candidate
    if obj_trivial < obj_spectral:
        return x_trivial
    else:
        return x

# EVOLVE-BLOCK-END
