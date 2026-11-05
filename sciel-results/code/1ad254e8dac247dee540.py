"""
Strategy: DFT Spectral Balancing with Complex Gradient Descent

The current program uses DCT (real, cosine basis) which biases towards symmetric,
low-frequency shapes. However, the autocorrelation of a sequence is the inverse
DFT of its power spectrum. To minimize the peak autocorrelation (aside from the
DC term which is fixed by the sum constraint), we want the power spectrum to be
as flat as possible (white noise).

This strategy:
1. Parameterizes h via its DFT coefficients H[k].
2. Uses the identity: autocorr = IDFT(|H|^2).
3. Optimizes the real and imaginary parts of H[k] (for k=1..N-1, H[0] fixed to 
   ensure sum(h) is constant/controlled) to flatten |H|^2.
4. Uses a smooth-max surrogate on the resulting autocorrelation to make the 
   objective differentiable.
5. Employs L-BFGS-B on the flattened real/imag vector.

This explores a different region of the solution space: sequences with flat 
spectral density (pseudo-random appearance) rather than smooth shapes.
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.fft

N = 400

def _value(h):
    """Compute the exact evaluator value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _smooth_max(x, tau=0.05):
    """Soft max using log-sum-exp."""
    x_max = np.max(x)
    x_shifted = (x - x_max) / tau
    x_shifted = np.clip(x_shifted, -100, 100)
    return x_max + tau * np.log(np.sum(np.exp(x_shifted)))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Best result found so far
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Target sum: We can scale h later, so the sum doesn't need to be 1.
    # However, the objective is homogeneous of degree 0: 
    # value(c*h) = value(h). 
    # So we can fix sum(h) = 1 to remove the scaling degree of freedom,
    # or just let the optimizer handle it. 
    # Fixing H[0] (DC component) is easier in Fourier domain.
    # Let's fix sum(h) = S. 
    # If we optimize H, h = IDFT(H). sum(h) = H[0].
    # We want to minimize abs(2*N*max(IDFT(|H|^2)) / |H[0]|^2).
    # Since the objective is scale-invariant, we can fix |H[0]| = 1 (or sum(h)=1).
    # Let's fix H[0] = 1 (real, positive). This implies sum(h) = 1.
    # Wait, h must be real. So H must satisfy conjugate symmetry: H[N-k] = conj(H[k]).
    # This means we only optimize H[1..N/2] (and H[N/2] is real if N even).
    
    # Variables: 
    # H[0] is fixed to 1.0 (real).
    # For k in 1..N/2-1: H[k] is complex (2 vars).
    # H[N/2] is real (1 var).
    # Total vars: 2*(N/2 - 1) + 1 = N - 1.
    # Actually, for N=400:
    # k=1..199: complex (398 vars)
    # k=200: real (1 var)
    # Total 399 vars.
    
    num_vars = N - 1
    # Indices for complex coeffs: 1 to N//2 - 1
    # Index for real coeff: N//2
    
    def _unpack_h(params):
        """
        params: array of length N-1.
        Returns h (real array of length N).
        """
        H = np.zeros(N, dtype=complex)
        H[0] = 1.0 # Fix DC to 1
        
        # Complex part
        # params[0:2*(N//2 - 1)] correspond to H[1], H[2], ..., H[N//2 - 1]
        num_complex = N // 2 - 1
        complex_params = params[:2 * num_complex]
        complex_part = complex_params[0::2] + 1j * complex_params[1::2]
        H[1 : 1 + num_complex] = complex_part
        
        # Conjugate symmetry
        H[N - (1 + num_complex) : N - 1] = np.conj(complex_part[::-1])
        
        # Real part at Nyquist
        H[N // 2] = params[2 * num_complex]
        
        # IDFT to get h
        # scipy.fft.idftn returns complex, but due to symmetry it should be real.
        h = scipy.fft.idftn(H, norm='backward') # Note: idftn includes 1/N factor
        # Wait, standard IDFT: x[n] = sum X[k] exp(2*pi*i*n*k/N).
        # scipy.fft.ifftn does: x[n] = 1/N sum X[k] exp(...).
        # So if I set H[0]=1, sum(h) = H[0] * (1/N * N) = 1? 
        # Let's check: sum(h) = h[0]+...+h[N-1]. 
        # h[n] = 1/N sum_k H[k] e^{...}.
        # sum_n h[n] = 1/N sum_k H[k] sum_n e^{...}.
        # sum_n e^{2*pi*i*n*k/N} is N if k=0, 0 otherwise.
        # So sum(h) = 1/N * H[0] * N = H[0] = 1.
        # Correct.
        
        # Ensure real
        h = h.real
        return h

    def _dft_objective(params, tau=0.1):
        """
        Objective function for DFT optimization.
        Minimize peak autocorrelation (smoothed) given sum(h)=1.
        """
        h = _unpack_h(params)
        s = np.sum(h) # Should be 1, but numerical errors might creep in
        if abs(s) < 1e-12:
            return 1e10
            
        # Autocorrelation via FFT is faster and more stable for large N
        # autocorr = IDFT(|FFT(h)|^2)
        # But we already have H. |H|^2 is the power spectrum.
        # autocorr = IDFT(|H|^2)
        # Note: scipy.fft.idftn(|H|^2) gives the circular autocorrelation.
        # The evaluator uses np.convolve(h, h) which is linear autocorrelation.
        # For N=400, linear and circular differ at the edges.
        # However, if we zero-pad, circular becomes linear.
        # Let's compute linear autocorrelation explicitly to be safe, 
        # or use FFT with zero-padding.
        
        # To match evaluator exactly:
        # conv = np.convolve(h, h)
        # This is linear.
        # Using FFT: 
        # fft_len = 2*N - 1
        # H_full = fft(h, n=fft_len)
        # autocorr = ifft(|H_full|^2).real
        # This is slightly more expensive but exact.
        
        fft_len = 2 * N - 1
        H_full = scipy.fft.fft(h, n=fft_len)
        power = np.abs(H_full) ** 2
        autocorr = scipy.fft.ifft(power).real
        
        # The evaluator takes max(autocorr). 
        # autocorr[0] is sum(h^2). 
        # We want to minimize max(autocorr) / sum(h)^2.
        # Since sum(h)=1, we minimize max(autocorr).
        
        # Use smooth max
        sm = _smooth_max(autocorr, tau=tau)
        
        # Add a small penalty if h becomes too "spiky" (optional, helps stability)
        # The smooth max already penalizes peaks.
        
        return sm

    # Initializations
    rng = np.random.default_rng(42)
    
    best_params = None
    best_obj_val = np.inf
    
    # Candidate 1: White Noise (Flat Spectrum)
    # H[k] random phase, constant magnitude
    H_init = np.zeros(N, dtype=complex)
    H_init[0] = 1.0
    for k in range(1, N):
        H_init[k] = np.exp(1j * rng.uniform(0, 2*np.pi))
    # Enforce conjugate symmetry
    for k in range(1, N//2 + 1):
        H_init[N-k] = np.conj(H_init[k])
    # Pack into params
    num_complex = N // 2 - 1
    params_init1 = np.zeros(N-1)
    complex_part = H_init[1 : 1 + num_complex]
    params_init1[0::2] = complex_part.real
    params_init1[1::2] = complex_part.imag
    params_init1[2 * num_complex] = H_init[N//2].real
    
    # Candidate 2: Chirp Spectrum (Linear phase)
    H_init2 = np.zeros(N, dtype=complex)
    H_init2[0] = 1.0
    phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    H_init2[1:] = np.exp(1j * phases)
    for k in range(1, N//2 + 1):
        H_init2[N-k] = np.conj(H_init2[k])
    params_init2 = np.zeros(N-1)
    complex_part2 = H_init2[1 : 1 + num_complex]
    params_init2[0::2] = complex_part2.real
    params_init2[1::2] = complex_part2.imag
    params_init2[2 * num_complex] = H_init2[N//2].real
    
    candidates = [params_init1, params_init2]
    
    # Evaluate candidates
    for p in candidates:
        val = _dft_objective(p, tau=0.1)
        if val < best_obj_val:
            best_obj_val = val
            best_params = p.copy()
            
    # Optimization Phase 1: L-BFGS-B with moderate tau
    if time.time() < deadline - 40.0:
        try:
            result = minimize(
                _dft_objective,
                best_params,
                args=(0.1,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_obj_val:
                best_obj_val = result.fun
                best_params = result.x
                print(f"DFT Phase 1: {best_obj_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT Phase 1 error: {e}", file=sys.stderr)

    # Convert to time domain
    h_dft = _unpack_h(best_params)
    val_dft = _value(h_dft)
    if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
        best_h = h_dft.copy()
        best_val = val_dft
        print(f"DFT Time Domain: {best_val:.6f}", file=sys.stderr)

    # Optimization Phase 2: Time Domain Polish
    # The DFT optimization might leave some local artifacts. 
    # Polish in time domain with a very tight smooth max.
    if time.time() < deadline - 20.0:
        try:
            def time_obj(h, tau):
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10
                conv = np.convolve(h, h)
                sm = _smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)
            
            result = minimize(
                time_obj,
                best_h,
                args=(0.02,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            h_opt = result.x
            val_opt = _value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Time Domain Polish: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Time Domain Polish error: {e}", file=sys.stderr)

    # Final check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

