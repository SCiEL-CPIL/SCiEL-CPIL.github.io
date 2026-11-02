"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(x):
    """Compute the C3' score: max|autoconv| / (sum(x))^2"""
    n = len(x)
    # Circular autoconvolution via FFT
    X = np.fft.fft(x)
    autoconv = np.fft.ifft(X * np.conj(X)).real
    max_abs = np.max(np.abs(autoconv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs / (integral * integral)

def _make_binary(x):
    """Project to binary ±1"""
    return np.sign(x)

def _local_search(x0, max_iters=500, time_limit=100):
    """Local search by flipping individual bits to reduce max|autoconv|"""
    x = x0.copy()
    best_score = _score(x)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the peak location in autoconv
        X = np.fft.fft(x)
        autoconv = np.fft.ifft(X * np.conj(X)).real
        peak_idx = np.argmax(np.abs(autoconv))
        
        # Try flipping each bit and see if it reduces the peak
        best_flip = -1
        best_improvement = 0
        
        # Sample a subset of positions to try
        n_tries = min(100, len(x))
        positions = np.random.choice(len(x), n_tries, replace=False)
        
        for pos in positions:
            x_trial = x.copy()
            x_trial[pos] = -x_trial[pos]
            
            # Quick score check
            X_t = np.fft.fft(x_trial)
            ac_t = np.fft.ifft(X_t * np.conj(X_t)).real
            max_abs_t = np.max(np.abs(ac_t))
            integral_t = np.sum(x_trial)
            if abs(integral_t) < 1e-12:
                continue
            score_t = max_abs_t / (integral_t * integral_t)
            
            if score_t < best_score - 1e-10:
                best_improvement = best_score - score_t
                best_flip = pos
        
        if best_flip >= 0:
            x[best_flip] = -x[best_flip]
            best_score = _score(x)
        else:
            # No improvement found, try random restarts
            if np.random.rand() < 0.1:
                # Small random perturbation
                n_perturb = np.random.randint(1, 10)
                perturb_positions = np.random.choice(len(x), n_perturb, replace=False)
                x[perturb_positions] = -x[perturb_positions]
                new_score = _score(x)
                if new_score < best_score:
                    best_score = new_score
                else:
                    x[perturb_positions] = -x[perturb_positions]  # revert
    
    return x, best_score

def _spectral_phase_retrieval(magnitude, n_iters=1000, time_limit=80):
    """
    Spectral phase retrieval: fix magnitude spectrum, optimize phase
    to minimize peak autoconvolution while enforcing binary constraint.
    """
    start_time = time.time()
    n = len(magnitude)
    
    # Initialize with random phase
    phase = np.random.uniform(0, 2*np.pi, n)
    best_x = None
    best_score = float('inf')
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Reconstruct signal from magnitude and phase
        spectrum = magnitude * np.exp(1j * phase)
        x = np.fft.ifft(spectrum).real
        
        # Enforce binary constraint
        x_bin = _make_binary(x)
        
        # Compute current score
        score = _score(x_bin)
        if score < best_score:
            best_score = score
            best_x = x_bin.copy()
        
        # Update phase: gradient descent on peak autoconvolution
        X = np.fft.fft(x_bin)
        autoconv = np.fft.ifft(X * np.conj(X)).real
        
        # Find peak
        peak_val = autoconv[np.argmax(np.abs(autoconv))]
        
        # Gradient of peak w.r.t. phase
        # d(peak)/d(phase_k) involves the imaginary part of X_k
        grad_phase = np.zeros(n)
        for k in range(n):
            # Approximate gradient: how changing phase_k affects the peak
            # This is complex, so use a simple heuristic:
            # Push phase to reduce the peak contribution
            grad_phase[k] = -np.imag(X[k]) * np.sign(peak_val)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_phase)
        if grad_norm > 1e-10:
            grad_phase = grad_phase / grad_norm
        
        # Step size with annealing
        step_size = 0.5 * (1.0 - it / n_iters) + 0.1
        phase = phase + step_size * grad_phase
        phase = np.mod(phase, 2 * np.pi)
    
    return best_x, best_score

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s limit
    
    # Start with a good initialization: flat-top magnitude spectrum
    # Flat-top window in frequency domain minimizes peak-to-sidelobe ratio
    n_freq = N
    freq_indices = np.arange(n_freq)
    
    # Create flat-top magnitude spectrum
    # Use a raised-cosine window for smoother transition
    w = np.ones(n_freq)
    # Apply a smooth window to reduce spectral leakage
    for i in range(n_freq):
        # Symmetric raised-cosine window
        if i < n_freq // 4:
            w[i] = 0.5 * (1 + np.cos(np.pi * (2 * i / (n_freq // 2))))
        elif i > 3 * n_freq // 4:
            w[i] = 0.5 * (1 + np.cos(np.pi * (2 * (n_freq - 1 - i) / (n_freq // 2))))
    
    magnitude = w
    
    best_x = None
    best_score = float('inf')
    
    # Run multiple spectral phase retrieval attempts with different initial phases
    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Different initial phases
        if attempt == 0:
            phase_init = np.zeros(n_freq)
        elif attempt == 1:
            phase_init = np.pi * np.random.randn(n_freq)
        else:
            phase_init = np.random.uniform(0, 2 * np.pi, n_freq)
        
        # Run spectral phase retrieval with this initial phase
        spectrum = magnitude * np.exp(1j * phase_init)
        x = np.fft.ifft(spectrum).real
        x_bin = _make_binary(x)
        
        score = _score(x_bin)
        if score < best_score:
            best_score = score
            best_x = x_bin.copy()
        
        # Refine with local search
        if time.time() - start_time < time_limit - 20:
            x_refined, score_refined = _local_search(x_bin, max_iters=200, time_limit=10)
            if score_refined < best_score:
                best_score = score_refined
                best_x = x_refined.copy()
    
    # Also try a few pure random binary sequences as baselines
    for _ in range(10):
        if time.time() - start_time > time_limit - 10:
            break
        x_rand = np.random.choice([-1, 1], N)
        score_rand = _score(x_rand)
        if score_rand < best_score:
            best_score = score_rand
            best_x = x_rand.copy()
    
    # Final local search on the best solution
    if best_x is not None and time.time() - start_time < time_limit - 5:
        x_final, score_final = _local_search(best_x, max_iters=500, time_limit=10)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
