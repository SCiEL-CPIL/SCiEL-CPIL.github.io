"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral phase retrieval approach: fix magnitude spectrum to a flat-top window
    and optimize phase via iterative projection to enforce +/-1 step constraint.
    """
    np.random.seed(42)
    
    # Flat-top window in frequency domain
    # A flat-top window has a flat main lobe and low sidelobes
    k = np.arange(N)
    # Use a raised-cosine-like window that is flat in the center
    # Normalize so DC component is large
    # Try: |X_k| = 1 for all k (white spectrum) -> autoconv is delta at 0
    # But we need x to be +/-1, so let's use a window that gives good properties
    
    # Actually, for +/-1 sequences, the power spectrum |X|^2 has specific properties.
    # Let's use a flat magnitude (all ones) as target, which corresponds to
    # an ideal delta autoconvolution (only c_0 = N, all other c_k = 0).
    # This is the ideal case we're trying to approach.
    
    target_mag = np.ones(N)
    # Ensure DC is reasonable
    target_mag[0] = np.sqrt(N)  # This will make S ~ sqrt(N)
    
    # Initialize phase randomly
    phase = np.random.uniform(0, 2 * np.pi, size=N)
    X = target_mag * np.exp(1j * phase)
    
    # Iterative phase retrieval
    for iteration in range(2000):
        # Compute time domain
        x = np.fft.ifft(X).real
        
        # Project to +/-1
        x_proj = np.sign(x)
        # Handle zeros
        x_proj[x_proj == 0] = 1.0
        
        # Recompute spectrum
        X_new = np.fft.fft(x_proj)
        
        # Update phase, keep target magnitude
        phase = np.angle(X_new)
        X = target_mag * np.exp(1j * phase)
        
        # Check convergence
        if iteration % 500 == 0:
            c = np.fft.ifft(X_new * np.conj(X_new)).real
            S = np.sum(x_proj)
            if abs(S) > 1e-10:
                score = np.max(np.abs(c)) / (S * S)
    
    # Final result
    x = np.fft.ifft(X).real
    x_proj = np.sign(x)
    x_proj[x_proj == 0] = 1.0
    
    # Local refinement: try flipping bits to improve score
    def score_fn(x):
        c = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x))).real
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        return np.max(np.abs(c)) / (S * S)
    
    current_score = score_fn(x_proj)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            x_proj[i] *= -1
            new_score = score_fn(x_proj)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x_proj[i] *= -1
    
    return x_proj

# EVOLVE-BLOCK-END
