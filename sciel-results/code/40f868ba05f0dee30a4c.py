"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize # Not strictly needed if implementing SA, but good for reference
import random

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function to minimize the normalized maximum
    of its autoconvolution.
    
    Approach:
    Represent the step function f as the squared magnitude of a complex signal g.
    g is constructed from a small number of frequency components with adjustable
    phases and amplitudes. We use Simulated Annealing to optimize these parameters
    to minimize the peak of the autoconvolution of f, assuming a max-normalization
    (Linf) which favors peaked structures over uniform ones in this context 
    (implied by the potential for improvement over the uniform seed).
    """
    
    # Number of frequency components in the phase-coded signal
    M = 8
    # Initial random seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Parameters to optimize:
    # - M frequencies (omega)
    # - M phases (phi)
    # - M amplitudes (amp)
    # Total params: 3*M
    
    # Initial values
    # Frequencies: spread over [0, pi]
    inits = rng.uniform(0, np.pi, M)
    phases = rng.uniform(0, 2 * np.pi, M)
    amps = rng.uniform(0.5, 1.0, M)
    
    # Pack into a single vector for SA
    # Indices: 0..M-1: omega, M..2M-1: phi, 2M..3M-1: amp
    current_params = np.concatenate([inits, phases, amps])
    
    def get_signal(params):
        """Construct the complex signal g from parameters."""
        omega = params[:M]
        phi = params[M:2*M]
        amp = params[2*M:3*M]
        
        k = np.arange(N)
        g = np.zeros(N, dtype=complex)
        for j in range(M):
            g += amp[j] * np.exp(1j * (phi[j] + omega[j] * k))
        return g

    def calculate_score(params):
        """
        Calculate the peak of the autoconvolution of f=|g|^2,
        normalized such that max(f) = 1.
        Lower is better.
        """
        g = get_signal(params)
        f = np.abs(g) ** 2
        
        # Normalize to max 1
        max_f = np.max(f)
        if max_f < 1e-12:
            return np.inf
        f_norm = f / max_f
        
        # Autoconvolution
        # Using FFT for speed
        # Zero pad to 2*N to avoid circular convolution effects
        fft_len = 1 << (2 * N - 1).bit_length()
        f_fft = np.fft.rfft(f_norm, n=fft_len)
        conv_fft = f_fft * f_fft
        conv = np.fft.irfft(conv_fft, n=fft_len)
        
        # The linear autoconvolution is in the first 2*N-1 samples
        # The peak is usually at the center (lag 0) for non-negative functions
        # But we take the global max to be safe
        peak = np.max(conv)
        
        return peak

    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    steps = 5000
    
    best_params = current_params.copy()
    best_score = calculate_score(current_params)
    current_score = best_score
    
    for i in range(steps):
        # Generate new candidate
        # Perturb frequencies, phases, and amplitudes
        noise_scale = 0.1 * T + 0.01
        new_params = current_params + rng.normal(0, noise_scale, size=len(current_params))
        
        # Constrain amplitudes to be positive
        new_params[2*M:3*M] = np.abs(new_params[2*M:3*M])
        
        # Constrain frequencies to [0, pi] (wrap around)
        new_params[:M] = new_params[:M] % np.pi
        
        new_score = calculate_score(new_params)
        
        # Acceptance criterion
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()
        else:
            # Boltzmann acceptance
            delta = new_score - current_score
            if delta < 0:
                pass # Already handled
            else:
                if T > T_min:
                    accept_prob = np.exp(-delta / T)
                    if rng.random() < accept_prob:
                        current_params = new_params
                        current_score = new_score
        
        T = max(T_min, T * alpha)
        
        # Early stopping if score is very good
        if best_score < 1.5: # Theoretical min for delta is 1.0
            break

    # Construct final signal
    g = get_signal(best_params)
    f = np.abs(g) ** 2
    
    # Normalize to max 1
    max_f = np.max(f)
    if max_f > 1e-12:
        f = f / max_f
        
    return list(f)
# EVOLVE-BLOCK-END
