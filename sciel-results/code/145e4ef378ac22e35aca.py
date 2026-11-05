import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use FFT for faster convolution if needed, but N=400 is small enough for direct
    # However, for consistency with the prompt's definition, we use convolve.
    # np.convolve(h, h) is full convolution.
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Strategy:
    # 1. Parameterize h via its DFT.
    # 2. Restrict optimization to the first K frequency bins (low frequencies).
    # 3. Use Differential Evolution to find optimal magnitudes/phases for these bins.
    # 4. Reconstruct h, ensure real-valued and sum != 0.
    
    # We will optimize the real and imaginary parts of the first K DFT coefficients.
    # To ensure h is real, we must enforce conjugate symmetry: H[k] = conj(H[N-k]).
    # Since N=400 is even, H[0] and H[200] are real. The rest come in pairs.
    # Let's optimize the magnitudes and phases of the first K/2 pairs, plus H[0].
    
    K = 20 # Number of frequency bins to consider (plus symmetry)
    # Variables: 
    # - mag[0] (real, H[0])
    # - For k in 1..K//2: mag[k], phase[k]
    # Total variables: 1 + K//2 * 2
    
    K_half = K // 2
    n_vars = 1 + K_half * 2
    
    # Bounds for magnitudes and phases
    # Magnitudes should be positive. Phases in [0, 2*pi).
    # We need to scale the resulting h such that sum(h) is not zero.
    # Actually, the objective is scale invariant: value(h) = value(c*h) for c!=0.
    # sum(c*h) = c*sum(h). max(conv(c*h)) = c^2 max(conv(h)).
    # value = 2*N*c^2*max(conv(h)) / (c*sum(h))^2 = 2*N*max(conv(h))/sum(h)^2.
    # So we can fix a normalization, e.g., sum(h) = 1, or just let DE find any scale.
    # However, if sum(h) is close to 0, the value explodes. 
    # H[0] = sum(h). So we should keep H[0] away from 0.
    
    # Let's define the objective function for DE
    def de_objective(params):
        if time.time() > deadline:
            return np.inf
            
        # Unpack params
        h0_mag = params[0] # This is H[0], which is real and equals sum(h)
        # If h0_mag is 0, sum(h)=0, which is invalid.
        # We can add a penalty or just ensure bounds keep it away from 0.
        # Let's assume bounds for h0_mag are [0.1, 10.0]
        
        H = np.zeros(N, dtype=complex)
        H[0] = h0_mag
        
        for i in range(K_half):
            idx = i + 1
            mag = params[1 + 2*i]
            phase = params[2 + 2*i]
            val = mag * np.exp(1j * phase)
            H[idx] = val
            H[N - idx] = np.conj(val)
        
        # Reconstruct h
        h = np.fft.ifft(H).real
        
        # Check validity
        if not np.all(np.isfinite(h)):
            return np.inf
            
        s = np.sum(h)
        if abs(s) < 1e-6:
            return np.inf
            
        # Evaluate
        conv = np.convolve(h, h)
        val = abs(2.0 * N * np.max(conv) / (s * s))
        
        return val

    # Define bounds
    # h0_mag: [0.1, 10.0]
    # mags: [0.0, 10.0]
    # phases: [0.0, 2*np.pi]
    
    bounds = [(0.1, 10.0)]
    for _ in range(K_half):
        bounds.append((0.0, 10.0)) # magnitude
        bounds.append((0.0, 2 * np.pi)) # phase

    # Run Differential Evolution
    # popsize=15, maxiter=50, tol=1e-4
    # We need to be careful with time.
    
    best_val = np.inf
    best_h = np.ones(N)
    
    try:
        # Use a fixed seed for reproducibility if needed, but DE is stochastic.
        # We can run multiple short DE runs or one longer one.
        # Given 145s budget, one DE run with maxiter=100 might be too slow if each eval is slow.
        # Eval cost: ifft (400) + convolve (400*400=160k) + max. ~1ms.
        # DE with popsize 15, maxiter 100 -> 1500 evals. ~1.5s. This is very fast.
        # We can increase maxiter or popsize.
        
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False # Skip local polish to save time and avoid local minima traps
        )
        
        if result.fun < best_val:
            best_val = result.fun
            # Reconstruct best_h
            params = result.x
            h0_mag = params[0]
            H = np.zeros(N, dtype=complex)
            H[0] = h0_mag
            for i in range(K_half):
                idx = i + 1
                mag = params[1 + 2*i]
                phase = params[2 + 2*i]
                val = mag * np.exp(1j * phase)
                H[idx] = val
                H[N - idx] = np.conj(val)
            best_h = np.fft.ifft(H).real
            
    except Exception:
        pass

    # If DE didn't find a good solution or failed, try a few random spectral initializations
    # to see if we can beat the seed (2.0)
    if best_val >= 2.0 or not np.isfinite(best_val):
        rng = np.random.default_rng(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            # Random low-freq spectrum
            H = np.zeros(N, dtype=complex)
            H[0] = 1.0 # Ensure sum is non-zero
            for i in range(1, K_half + 1):
                mag = rng.uniform(0, 1)
                phase = rng.uniform(0, 2*np.pi)
                val = mag * np.exp(1j * phase)
                H[i] = val
                H[N-i] = np.conj(val)
            h = np.fft.ifft(H).real
            val = _evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h

    # Final fallback: Seed
    if not np.isfinite(best_val) or best_val >= 2.0:
        best_h = np.ones(N)
        best_val = 2.0

    return best_h
