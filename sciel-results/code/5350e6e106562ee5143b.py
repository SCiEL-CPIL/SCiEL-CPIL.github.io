"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy:
    1. Formulate as spectral phase retrieval.
    2. Fix magnitude spectrum to a flat-top window (constant magnitude sqrt(N)).
    3. Optimize phase via iterative projection (Fienup's algorithm) to enforce
       the step-function constraint (x in {-1, 1}) in the time domain.
    4. This minimizes the peak-to-sidelobe ratio in the autoconvolution domain.
    5. Post-process to maximize the integral S (sum of x) to improve the score,
       as score = max|c| / S^2.
    """
    
    np.random.seed(42)
    
    # Target magnitude spectrum
    # For a binary sequence x, sum |x_i|^2 = N.
    # By Parseval, sum |X_k|^2 = N * sum |x_i|^2 = N^2.
    # If |X_k| is constant M, then N * M^2 = N^2 => M = sqrt(N).
    M = np.sqrt(N)
    
    # Initialize X with random phase and target magnitude
    phase = np.random.uniform(0, 2 * np.pi, N)
    X = M * np.exp(1j * phase)
    
    # Iterative phase retrieval
    for _ in range(200):
        # Time domain: IFFT
        x = np.fft.ifft(X).real
        
        # Enforce step function constraint: project to {-1, 1}
        # We want to maximize S, so we might bias the projection slightly towards +1?
        # Standard projection is sign(x).
        x_proj = np.sign(x)
        # Ensure no zeros (sign(0) is 0, map to 1)
        x_proj[x_proj == 0] = 1.0
        
        # Frequency domain: FFT
        X_calc = np.fft.fft(x_proj)
        
        # Enforce magnitude constraint
        mag_calc = np.abs(X_calc)
        # Avoid division by zero
        mag_calc[mag_calc < 1e-10] = 1e-10
        X = M * (X_calc / mag_calc)
        
    # Final x_proj
    x = x_proj
    
    # Post-processing: Try to increase S (sum of x) to minimize score.
    # Score = max|c| / S^2.
    # Increasing S helps. Flipping a -1 to +1 increases S by 2.
    # Check if this increases max|c| too much.
    
    def compute_autoconv(x):
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        return c

    def get_score(x):
        c = compute_autoconv(x)
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        max_c = np.max(np.abs(c))
        return max_c / (S * S)
    
    current_score = get_score(x)
    
    # Local search: flip bits to improve score
    # Prioritize flipping -1 to +1 to increase S
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if x[i] == -1.0:
                x[i] = 1.0
                new_score = get_score(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    x[i] = -1.0
            elif x[i] == 1.0:
                # Also try flipping +1 to -1 if it reduces max|c| significantly
                # but this decreases S, so usually bad unless max|c| drops a lot.
                x[i] = -1.0
                new_score = get_score(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    x[i] = 1.0
                    
    return x

# EVOLVE-BLOCK-END
