import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # The strategy:
    # 1. Represent h as |IDFT(mask)| where mask is sparse in frequency.
    # 2. Optimize the mask parameters (amplitudes and phases) using Differential Evolution.
    # 3. This transforms the problem into a smooth (mostly) continuous search over frequency space,
    #    potentially escaping local minima of coordinate descent in time domain.
    
    # We will try a few different sparsities for the mask.
    # A sparse mask in frequency corresponds to a signal with limited bandwidth/structure.
    # Non-negativity is enforced by taking the absolute value of the IDFT.
    
    # Parameters for the mask:
    # We select K frequency bins. For each bin, we have an amplitude (>=0) and a phase.
    # Since h must be real, the mask must be Hermitian symmetric.
    # It's easier to define the mask in terms of real and imaginary parts, or just amplitudes and phases for positive frequencies.
    # However, Differential Evolution works on real vectors.
    # Let's define the variable vector as: [amp_1, phase_1, amp_2, phase_2, ...] for K positive frequency bins.
    # Plus maybe a DC component (real, non-negative).
    
    # To ensure h is non-negative, we take h = |IDFT(mask)|.
    # Note: IDFT of a Hermitian mask is real. So h = |real_part(IDFT(mask))|?
    # No, if the mask is Hermitian, IDFT is real. So h = |IDFT(mask)| is just abs of a real signal.
    # This ensures h >= 0.
    
    # Let's define the objective function for Differential Evolution.
    # The variable x represents the spectral parameters.
    
    def objective(x):
        if time.time() > deadline - 5.0:
            return -1e9  # Penalize timeout
        
        # Parse x into spectral parameters
        # x = [dc_amp, amp_1, phase_1, amp_2, phase_2, ...]
        # Let K be the number of positive frequency bins to consider.
        # We need to know K. We'll fix K for each run of DE.
        
        # This function is called by DE, so K must be fixed in the closure or global.
        # We'll handle this by defining separate objective functions for different K.
        
        # For now, this is a placeholder. The actual logic will be in the loop below.
        return 0.0

    # We will run DE for a few different sparsities K.
    # K=1: 1 positive freq bin + DC.
    # K=2: 2 positive freq bins + DC.
    # ...
    
    # The vector length for DE:
    # If we use K positive bins, we have K amplitudes and K phases.
    # Plus 1 DC amplitude (real, non-negative).
    # Total variables = 1 + 2*K.
    
    # Bounds for DE:
    # Amplitudes: [0, 10] (scale invariant, so we can normalize later, but DE needs bounds)
    # Phases: [0, 2*pi]
    # DC: [0, 10]
    
    # Let's try K = 1, 2, 3, 5.
    
    np.random.seed(42)
    
    for K in [1, 2, 3, 5]:
        if time.time() > deadline - 10.0:
            break
            
        # Define the specific objective for this K
        def obj_k(x):
            if time.time() > deadline - 5.0:
                return -1e9
                
            # x = [dc, amp_1, phase_1, amp_2, phase_2, ...]
            dc = x[0]
            amps = x[1::2]
            phases = x[2::2]
            
            # Construct the frequency domain mask H
            # N=50. Frequencies are 0, 1, ..., N-1.
            # Positive frequencies: 1 to N//2.
            # We pick the first K positive frequencies: 1, 2, ..., K.
            # And their Hermitian counterparts: N-1, N-2, ..., N-K.
            
            H = np.zeros(N, dtype=complex)
            H[0] = dc
            
            for i in range(K):
                freq = i + 1
                amp = amps[i]
                phase = phases[i]
                
                # Positive frequency
                H[freq] = amp * np.exp(1j * phase)
                
                # Negative frequency (Hermitian)
                # For real signal, H[N-freq] = conj(H[freq])
                H[N - freq] = np.conj(H[freq])
            
            # Compute IDFT
            # numpy.fft.ifft returns complex. If H is Hermitian, it should be real (up to numerical error).
            h_complex = np.fft.ifft(H)
            h_real = np.real(h_complex)
            
            # Enforce non-negativity
            h = np.abs(h_real)
            
            # Normalize? The ratio is scale invariant.
            # But if h is all zeros, ratio is undefined/0.
            # DE might find a small amplitude. We can normalize h to sum=1 before computing ratio
            # to avoid numerical issues with very small/large values, but it's not strictly necessary
            # if we handle the 0 case.
            
            if np.sum(h) < 1e-15:
                return 0.0
                
            # Compute ratio
            return _compute_ratio(h)
        
        # Bounds for DE
        # x[0] = DC: [0, 10]
        # x[1], x[3], ... = Amps: [0, 10]
        # x[2], x[4], ... = Phases: [0, 2*pi]
        
        bounds = [(0, 10)]  # DC
        for i in range(K):
            bounds.append((0, 10))  # Amp
            bounds.append((0, 2 * np.pi))  # Phase
        
        # Run Differential Evolution
        # maxiter and popsize control the effort.
        # With N=50 and small K, the function evaluation is fast.
        # We can afford more iterations.
        
        try:
            result = differential_evolution(
                obj_k, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-5, 
                seed=42,
                polish=True  # Use L-BFGS-B to polish the result
            )
            
            if result.fun > best_ratio:
                best_ratio = result.fun
                # Reconstruct h from result.x
                x_opt = result.x
                dc = x_opt[0]
                amps = x_opt[1::2]
                phases = x_opt[2::2]
                
                H = np.zeros(N, dtype=complex)
                H[0] = dc
                for i in range(K):
                    freq = i + 1
                    amp = amps[i]
                    phase = phases[i]
                    H[freq] = amp * np.exp(1j * phase)
                    H[N - freq] = np.conj(H[freq])
                
                h_complex = np.fft.ifft(H)
                h_real = np.real(h_complex)
                h_candidate = np.abs(h_real)
                
                # Verify ratio
                r = _compute_ratio(h_candidate)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_candidate.copy()
                    
        except Exception as e:
            print(f"DE failed for K={K}: {e}")
            continue
            
        # Check time
        if time.time() > deadline - 10.0:
            break

    # Fallback: If DE didn't find anything good, use the previous best or a simple heuristic
    if best_ratio < 1.0:
        # Use the uniform initialization from the previous program as a fallback
        h_uniform = np.ones(N) / N
        r = _compute_ratio(h_uniform)
        if r > best_ratio:
            best_ratio = r
            best_h = h_uniform.copy()

    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
