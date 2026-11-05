import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import irfft, rfft

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _evaluate(h):
    """
    Computes the objective value exactly as defined in the task.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    if not np.all(np.isfinite(h)):
        return np.inf
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    # Full autoconvolution
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (S * S))

def _construct_from_spectrum(params):
    """
    Constructs a real signal h of length N from spectral parameters.
    params: [freq_indices (K ints), amplitudes (K floats), phases (K floats)]
    Here we assume a fixed set of K frequency indices for the optimizer,
    and params contains only the amplitudes and phases for those indices.
    
    To allow the optimizer to choose the frequencies, we could make them continuous,
    but IDFT requires integer bins. Instead, we fix a set of candidate bins
    and optimize the weights (amplitude and phase) for a sparse subset.
    
    For this strategy, we will fix a set of candidate frequency indices 
    (e.g., low frequencies and some high frequencies) and optimize 
    the complex coefficients for a sparse mask.
    """
    K = len(params) // 2
    amps = params[:K]
    phases = params[K:]
    
    # We need to know which frequency indices to use.
    # This function assumes the indices are provided via closure or global,
    # but since minimize expects a flat vector, we'll structure the params
    # to include indices if we want to optimize them.
    # However, optimizing integer indices with L-BFGS is hard.
    # Alternative: Fix a set of candidate indices and optimize the weights.
    # Let's assume the caller passes the indices.
    # Wait, the signature of _construct_from_spectrum in the prompt inspiration
    # implies it handles the construction. 
    # Let's define a global list of candidate indices for this run.
    # We will use a mix of low and mid frequencies.
    
    # To make it self-contained, we will define the indices inside the objective function
    # or pass them as part of the state. 
    # Let's modify the approach: 
    # We will define a function that takes (amps, phases) and uses a FIXED set of indices.
    # We will try multiple FIXED sets of indices in the main loop.
    
    # This function is a helper for a specific set of indices.
    # It expects `indices` to be available. 
    # Let's change the design: 
    # _objective_sparse(indices, params)
    
    # For now, this function is a placeholder to match the name if needed,
    # but we will implement the logic in _objective_sparse.
    raise NotImplementedError("Use _objective_sparse")

def _objective_sparse(indices, params):
    """
    Objective function for sparse spectral optimization.
    indices: list of integer frequency bins (1..N/2) to use.
    params: [a1, ..., aK, p1, ..., pK] where a are amplitudes, p are phases.
    """
    K = len(params) // 2
    amps = params[:K]
    phases = params[K:]
    
    # Create the complex spectrum
    # rfft of length N gives N//2 + 1 bins (0 to N//2)
    spec = np.zeros(N // 2 + 1, dtype=complex)
    
    # DC component (index 0)
    # We might want to allow DC to be non-zero to control the mean.
    # Let's include index 0 in the candidate indices if we want to optimize mean.
    # If indices contains 0, it's handled below.
    
    for i, idx in enumerate(indices):
        if idx < 0 or idx > N // 2:
            continue
        spec[idx] = amps[i] * np.exp(1j * phases[i])
        
    # Inverse rfft to get real signal h
    h = irfft(spec, n=N)
    
    # Evaluate
    return _evaluate(h)

def _optimize_sparse(indices, start_params, deadline):
    """
    Optimizes the sparse spectral parameters for a given set of indices.
    """
    K = len(indices)
    
    # Bounds for amplitudes and phases
    # Amplitudes can be negative? No, we have phase. So amplitude >= 0.
    # But allowing negative amplitude is equivalent to phase shift by pi.
    # Let's allow amplitude in [0, 10] and phase in [0, 2*pi].
    bounds = []
    for _ in range(K):
        bounds.append((0.0, 10.0)) # Amplitude
    for _ in range(K):
        bounds.append((0.0, 2 * np.pi)) # Phase
        
    try:
        res = minimize(
            lambda p: _objective_sparse(indices, p),
            start_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        return res.x, res.fun
    except Exception:
        return start_params, _objective_sparse(indices, start_params)

def construct():
    start_time = time.time()
    
    # Baseline
    best_h = np.ones(N)
    best_val = _evaluate(best_h) # 2.0
    
    np.random.seed(42)
    
    # Strategy: Try multiple sets of sparse frequency indices.
    # The autoconvolution of a signal is related to the squared magnitude of its FFT.
    # Specifically, circular autoconvolution corresponds to |FFT(h)|^2.
    # Linear autoconvolution is different, but for N=400, if we zero-pad or if the signal
    # has compact support or decays, they are related.
    # However, we are minimizing the MAX of the linear autoconvolution.
    # A sparse spectrum in the DFT domain creates a signal that is a sum of sinusoids.
    # The autoconvolution of a sum of sinusoids will have peaks at the frequencies
    # corresponding to the differences of the sinusoid frequencies.
    # To minimize the max peak, we want to spread the energy or create destructive
    # interference in the autoconvolution domain.
    
    # Let's define several candidate sets of indices.
    # We should include low frequencies (for large scale structure) and some higher ones.
    
    candidate_index_sets = []
    
    # Set 1: Low frequencies
    candidate_index_sets.append(list(range(1, 6))) # 1..5
    
    # Set 2: Mix of low and mid
    candidate_index_sets.append([1, 2, 5, 10, 20])
    
    # Set 3: Higher frequencies
    candidate_index_sets.append([10, 20, 30, 50, 80])
    
    # Set 4: Random sparse set
    candidate_index_sets.append(np.random.choice(np.arange(1, N//2), size=8, replace=False).tolist())
    
    # Set 5: Another random
    candidate_index_sets.append(np.random.choice(np.arange(1, N//2), size=10, replace=False).tolist())
    
    # Set 6: Include DC (index 0) to control mean
    # Note: If sum(h) is close to 0, the value blows up. 
    # We need sum(h) to be significant. 
    # DC component controls the mean. 
    # Let's add a set with DC.
    candidate_index_sets.append([0, 1, 2, 5, 10])
    
    for indices in candidate_index_sets:
        if time.time() > DEADLINE - 10:
            break
            
        K = len(indices)
        # Initial params: Random small amplitudes, random phases
        # Start with small amplitudes to avoid huge values initially?
        # Or start with ones?
        start_params = np.random.rand(2 * K) * 5.0 # Amplitudes
        start_params[K:] = np.random.rand(K) * 2 * np.pi # Phases
        
        # If index 0 is included, we might want to initialize its amplitude differently
        # to ensure non-zero sum.
        
        opt_params, opt_val = _optimize_sparse(indices, start_params, DEADLINE - 5)
        
        # Reconstruct h
        # We need to replicate the logic from _objective_sparse to get h
        # Let's extract h from the optimization result if possible, or reconstruct.
        # Since _optimize_sparse returns params, we reconstruct.
        
        amps = opt_params[:K]
        phases = opt_params[K:]
        spec = np.zeros(N // 2 + 1, dtype=complex)
        for i, idx in enumerate(indices):
            if 0 <= idx <= N // 2:
                spec[idx] = amps[i] * np.exp(1j * phases[i])
        
        h_candidate = irfft(spec, n=N)
        
        val = _evaluate(h_candidate)
        
        if val < best_val and np.all(np.isfinite(h_candidate)):
            best_val = val
            best_h = h_candidate
            
    # Refinement: Greedy local search on the best candidate
    if time.time() < DEADLINE - 5:
        # Simple greedy refinement
        refined_h = best_h.copy()
        refined_val = best_val
        scale = np.std(refined_h)
        if scale < 1e-6:
            scale = 1.0
        step = 0.1 * scale
        
        for _ in range(100):
            if time.time() > DEADLINE - 2:
                break
                
            improved = False
            indices = np.random.choice(N, size=20, replace=False)
            
            for idx in indices:
                h_test = refined_h.copy()
                h_test[idx] += step
                val_plus = _evaluate(h_test)
                
                h_test[idx] = refined_h[idx] - 2 * step
                val_minus = _evaluate(h_test)
                
                if val_plus < refined_val:
                    refined_h[idx] += step
                    refined_val = val_plus
                    improved = True
                elif val_minus < refined_val:
                    refined_h[idx] -= step
                    refined_val = val_minus
                    improved = True
            
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
            else:
                if np.random.rand() < 0.1:
                    step *= 1.1
        
        if refined_val < best_val:
            best_val = refined_val
            best_h = refined_h

    # Final check
    final_val = _evaluate(best_h)
    if final_val != np.inf and np.all(np.isfinite(best_h)):
        return best_h
    else:
        return np.ones(N)

if __name__ == "__main__":
    h = construct()
    val = _evaluate(h)
    print(f"Final Value: {val}")
    print(f"Score: {2.0 / val}")
