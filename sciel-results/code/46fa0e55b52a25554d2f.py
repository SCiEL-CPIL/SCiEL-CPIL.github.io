"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function to minimize the normalized maximum
    of its autoconvolution.
    
    Approach:
    Represent the step function f as the squared magnitude of a complex signal
    g = envelope * exp(i * phase). The autoconvolution peak of f is related to
    the L4 norm of g. We use a smooth envelope and optimize the phase profile
    via simulated annealing to minimize the autoconvolution peak of f.
    
    The phase profile is parameterized as a sum of a few sinusoidal frequency
    hops to keep the search space manageable, plus small random perturbations.
    """
    # Smooth envelope: raised cosine or similar to reduce edge effects
    t = np.linspace(0, 1, N, endpoint=False)
    envelope = 0.5 * (1 + np.cos(2 * np.pi * t))  # raised cosine, zero at edges
    
    # Ensure envelope is strictly positive to avoid issues
    envelope = np.maximum(envelope, 1e-6)
    
    # Initial phase: linear chirp plus random offset
    rng = np.random.RandomState(42)
    base_phase = 2 * np.pi * (0.5 * t**2 + rng.uniform(0, 1))
    best_phases = base_phase.copy()
    
    def compute_peak(phases):
        """Compute the peak of the autoconvolution of f = |envelope * exp(i*phases)|^2."""
        g = envelope * np.exp(1j * phases)
        f = np.abs(g) ** 2
        
        # Normalize f so that its integral is 1 (or max is 1, depending on evaluator)
        # Try L1 normalization first
        integral_f = np.sum(f)
        if integral_f > 0:
            f_norm = f / integral_f
        else:
            f_norm = f
        
        # Compute autoconvolution using FFT
        n_fft = 2 * N
        f_padded = np.zeros(n_fft)
        f_padded[:N] = f_norm
        
        F = np.fft.fft(f_padded, n=n_fft)
        autoconv = np.fft.ifft(F * F).real
        autoconv = autoconv[:N]
        
        peak = np.max(np.abs(autoconv))
        return peak
    
    best_score = compute_peak(best_phases)
    
    # Simulated annealing
    T = 1.0
    cooling = 0.998
    n_iter = 3000
    
    for i in range(n_iter):
        # Perturb: add small random changes to phases
        n_perturb = rng.randint(1, 10)
        idx = rng.randint(0, N, n_perturb)
        new_phases = best_phases.copy()
        new_phases[idx] += rng.uniform(-0.5, 0.5, n_perturb)
        new_phases[idx] = new_phases[idx] % (2 * np.pi)
        
        new_score = compute_peak(new_phases)
        
        delta = new_score - best_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_phases = new_phases
            best_score = new_score
        
        T *= cooling
    
    # Build final f
    g = envelope * np.exp(1j * best_phases)
    f = np.abs(g) ** 2
    
    # Ensure nonnegativity
    f = np.maximum(f, 0)
    
    return list(f)
# EVOLVE-BLOCK-END
