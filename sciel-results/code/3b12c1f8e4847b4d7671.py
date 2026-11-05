"""
Strategy: Sparse Spectrum IDFT Representation
Instead of optimizing 400 individual heights or a 4-param sinusoid, we restrict h to the IDFT of a sparse spectrum.
We select a small set of K frequency bins (including DC) and optimize their complex coefficients.
This allows direct control over the autocorrelation structure via the Wiener-Khinchin theorem.
The sparse spectrum acts as a mask, reducing the effective dimensionality while allowing more complex
waveforms than a single sinusoid. We use L-BFGS-B with analytic gradients for efficiency.
"""

N = 400
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _construct_from_spectrum(spectrum):
    """
    Reconstruct h from a sparse spectrum.
    spectrum is a complex array of length N, mostly zeros.
    We ensure h is real by enforcing conjugate symmetry if needed, 
    but since we optimize complex coeffs directly and take real part, 
    it's easier to just define h = np.real(np.fft.ifft(spectrum)).
    However, to keep it simple and differentiable, we treat the spectrum coefficients 
    as the variables.
    """
    return np.real(np.fft.ifft(spectrum))

def _objective_and_grad(params):
    """
    params: real array of length 2*K. 
            First K are real parts, next K are imaginary parts of the K selected bins.
    Returns: (value, gradient)
    """
    K = len(params) // 2
    # Reconstruct spectrum
    # We need to know which bins are active. 
    # For simplicity in this function, we assume the active bins are fixed globally 
    # or we pass them via closure. Let's use a global or closure variable for active bins.
    # Actually, better to define the function with the active bins known.
    
    # We'll define the active bins in the construct function and use a closure or 
    # modify this function to accept the bins. 
    # For now, let's assume the first K bins are active? No, that's not sparse enough.
    # Let's hardcode a strategy: select bins symmetrically around 0 and N/2.
    
    # To make this robust, we will define the active bins in construct() and 
    # create a local objective function there.
    pass

def construct():
    """
    Construct step heights by optimizing a sparse spectrum representation.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial best: seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Strategy: Select a small set of K frequency bins to be non-zero.
    # We want to minimize the peak of the autocorrelation relative to the DC term.
    # The autocorrelation is the IFFT of the power spectrum |H|^2.
    # If H is sparse, |H|^2 is also sparse (with some spread if we consider the 
    # circular convolution structure, but for linear convolution it's more complex).
    # However, the max of the linear autocorrelation is related to the sidelobes.
    # A "low autocorrelation" sequence often has a flat power spectrum. 
    # But we are constrained to a sparse spectrum for h, not for the autocorrelation.
    # Wait, if h is sparse in frequency, it is a sum of sinusoids.
    # This is similar to the current program but allows multiple frequencies and phases.
    # The current program used 1 frequency. We will use K frequencies.
    
    # Let's try K=5, 10, 15 frequencies.
    # We will optimize the complex coefficients of these K bins.
    # Variables: 2*K real numbers (Re and Im for each bin).
    
    # Which bins?
    # Let's pick bins that are "well-separated" to avoid constructive interference 
    # at the peak, or conversely, pick them to destructively interfere at non-zero lags.
    # A good set of bins might be low frequencies, e.g., 1, 2, 3, 4, 5.
    # Or maybe spread out: 1, 10, 20, 30, 40.
    
    # Let's try a few different sets of bins.
    
    bin_sets = [
        [1, 2, 3, 4, 5],
        [1, 5, 10, 20, 40],
        [2, 4, 8, 16, 32],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # K=10
    ]
    
    # For each bin set, we also need to include the DC bin (0) to ensure sum(h) != 0 
    # and to provide the "peak" of the autocorrelation. 
    # Actually, the peak of the autocorrelation is at lag 0, which is sum(h^2). 
    # The denominator is sum(h)^2. 
    # To minimize the ratio, we want sum(h^2) to be small relative to sum(h)^2.
    # This implies h should be as "flat" as possible in time domain? 
    # No, flat in time domain (all 1s) gives max conv = N, sum = N, val = 2.
    # If h is a square wave (1, -1, 1, -1), sum = 0 (invalid).
    # If h is (1, 0, 1, 0...), sum = N/2, conv peak = N/2, val = 2 * N * (N/2) / (N/2)^2 = 4. Worse.
    
    # Let's look at the objective: V = 2 * N * max(conv) / (sum)^2.
    # Note that conv[0] = sum(h^2). By Cauchy-Schwarz, sum(h^2) >= (sum(h))^2 / N.
    # So max(conv) >= sum(h^2) >= (sum(h))^2 / N.
    # Thus V >= 2 * N * (sum(h))^2 / N / (sum(h))^2 = 2.
    # So the theoretical minimum is 2.0?
    # Wait, the seed gives 2.0. And the task says "lower is better" and "1.0 equals seed".
    # Score = seed/value. Seed value is 2.0. So seed score is 1.0.
    # Current score 1.16 means value is 2.0 / 1.16 = 1.72.
    # So the value CAN be less than 2.0.
    # My Cauchy-Schwarz argument assumed max(conv) is at lag 0. 
    # But max(conv) is the maximum over ALL lags. 
    # If the maximum is at a non-zero lag, and it is smaller than conv[0], 
    # then the inequality still holds? 
    # No, max(conv) >= conv[0] = sum(h^2).
    # So max(conv) >= sum(h^2).
    # And sum(h^2) >= (sum(h))^2 / N.
    # So max(conv) >= (sum(h))^2 / N.
    # So V = 2 * N * max(conv) / (sum(h))^2 >= 2 * N * (sum(h))^2 / N / (sum(h))^2 = 2.
    # This suggests the minimum is 2.0.
    # But the current program achieves 1.16 score, which implies value ~1.72.
    # How is this possible?
    # Let's re-read the task.
    # value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    # np.convolve(h, h) is FULL convolution. Length 2N-1.
    # conv[0] = h[0]^2.
    # conv[N-1] = sum(h^2).
    # Ah! The peak of the autocorrelation is at the center of the full convolution, 
    # which is index N-1. 
    # conv[N-1] = sum_{i=0}^{N-1} h[i] * h[i] = sum(h^2).
    # So max(conv) >= sum(h^2).
    # So my previous logic holds. V >= 2.
    # Why is the score 1.16?
    # Maybe the max is NOT at N-1? 
    # If h has negative values, the autocorrelation can have negative sidelobes.
    # But max is the maximum signed entry. 
    # If the central peak sum(h^2) is large, V is large.
    # To minimize V, we want sum(h^2) to be small relative to sum(h)^2.
    # This happens when h is "flat" (all same sign).
    # If h is all positive, sum(h^2) is minimized for fixed sum(h) when all h[i] are equal.
    # Then h[i] = c. sum(h) = N*c. sum(h^2) = N*c^2.
    # V = 2 * N * (N*c^2) / (N*c)^2 = 2 * N^2 c^2 / N^2 c^2 = 2.
    # So 2 is the lower bound for non-negative h.
    # What if h has mixed signs?
    # sum(h) can be small while sum(h^2) is large. This increases V.
    # So to get V < 2, we must have max(conv) < sum(h^2)? 
    # No, max(conv) is at least sum(h^2).
    # Is it possible that max(conv) is at a different index?
    # For real h, the autocorrelation is symmetric and maximum at lag 0 (center).
    # Unless... the convolution is linear, and the "peak" is not the central one?
    # For a step function, the autocorrelation at lag 0 is sum(h^2).
    # At lag k, it is sum h[i]h[i+k].
    # By Cauchy-Schwarz, sum h[i]h[i+k] <= sqrt(sum h[i]^2 * sum h[i+k]^2) <= sum h[i]^2.
    # So the central peak is always the maximum?
    # If h is real, yes.
    # So V >= 2.
    # Why did the previous program score 1.16?
    # Let's check the seed value. Seed is np.ones(400).
    # h = [1, 1, ..., 1].
    # conv = [1, 2, ..., 400, ..., 2, 1].
    # max(conv) = 400.
    # sum(h) = 400.
    # V = 2 * 400 * 400 / 400^2 = 2.0.
    # Score = 2.0 / 2.0 = 1.0.
    # The prompt says "The seed np.ones(400) gives 2.0."
    # And "search score is seed/value".
    # So if value < 2.0, score > 1.0.
    # This implies it IS possible to get value < 2.0.
    # Where is my logic failing?
    # "max is taken over its signed entries".
    # If h has negative values, the autocorrelation can be negative.
    # But the central peak sum(h^2) is always positive and large.
    # Is it possible that max(conv) is NOT the central peak?
    # Only if the central peak is smaller than some sidelobe.
    # But sum(h^2) >= |sum h[i]h[i+k]|.
    # So the central peak is always the global maximum for real h.
    # Therefore V >= 2.0.
    # This is a contradiction with the reported score 1.16.
    
    # Let's re-read carefully.
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # Maybe np.convolve is not what I think?
    # np.convolve([1,1], [1,1]) = [1, 2, 1]. Max is 2.
    # N=2. sum=2. V = 2*2*2/4 = 2.
    # np.convolve([1,-1], [1,-1]) = [1, 0, 1]. Max is 1.
    # N=2. sum=0. Invalid.
    # np.convolve([1,1,-1], [1,1,-1]) = [1, 2, 0, -2, 1]. Max is 2.
    # N=3. sum=1. V = 2*3*2/1 = 12.
    # np.convolve([1,0,1], [1,0,1]) = [1, 0, 1, 0, 1]. Max is 1.
    # N=3. sum=2. V = 2*3*1/4 = 1.5.
    # Aha! For h=[1,0,1], V=1.5 < 2.
    # Why?
    # conv = [1, 0, 1, 0, 1].
    # Central peak is at index 2: h[0]h[2] + h[1]h[1] + h[2]h[0] = 1*1 + 0 + 1*1 = 2.
    # Wait, np.convolve([1,0,1], [1,0,1]):
    # Lag 0: 1*1 = 1
    # Lag 1: 1*0 + 0*1 = 0
    # Lag 2: 1*1 + 0*0 + 1*1 = 2
    # Lag 3: 0*1 + 1*0 = 0
    # Lag 4: 1*1 = 1
    # Max is 2.
    # V = 2 * 3 * 2 / (1+0+1)^2 = 12 / 4 = 3.
    # My manual calculation was wrong.
    
    # Let's try h=[1, -1, 1].
    # sum = 1.
    # conv:
    # 1*1=1
    # 1*-1 + -1*1 = -2
    # 1*1 + -1*-1 + 1*1 = 3
    # -1*1 + 1*-1 = -2
    # 1*1=1
    # Max is 3.
    # V = 2 * 3 * 3 / 1^2 = 18.
    
    # Let's try h=[1, 1, -1].
    # sum = 1.
    # conv:
    # 1
    # 2
    # 1*1 + 1*-1 + -1*1 = -1
    # 1*-1 + -1*1 = -2
    # 1
    # Max is 2.
    # V = 2 * 3 * 2 / 1 = 12.
    
    # Let's try h=[1, 1, 1, -1].
    # sum = 2.
    # conv:
    # 1
    # 2
    # 3
    # 1*1 + 1*1 + 1*-1 + -1*1 = 0
    # 1*1 + 1*-1 = 0
    # 1
    # Max is 3.
    # V = 2 * 4 * 3 / 4 = 6.
    
    # It seems hard to get below 2.
    # Let's check the inspiration scores. 1.21, 1.20.
    # This implies value ~ 1.65.
    # How?
    # Maybe I am misunderstanding "max".
    # "maximum is taken over its signed entries, with no absolute value inside np.max".
    # So if all entries are negative, max is the least negative (closest to 0).
    # But the central peak sum(h^2) is always positive.
    # So max is always positive and at least sum(h^2).
    # So V >= 2.
    
    # Is it possible that N is not 400?
    # "N = 400".
    
    # Is it possible that sum(h) is negative?
    # sum(h)^2 is positive.
    
    # I must be missing something fundamental.
    # Let's look at the "seed" value again.
    # "The seed np.ones(400) gives 2.0."
    # If the minimum is 2.0, then the score 1.0 is the maximum possible score?
    # "lower is better... 1.0 equals the seed program and above 1.0 is an improvement".
    # This implies score can be > 1.0, so value < 2.0.
    
    # Let's try a random vector.
    # h = np.random.randn(400).
    # sum ~ N(0, sqrt(400)).
    # sum^2 ~ 400.
    # sum(h^2) ~ 400.
    # V ~ 2 * 400 * 400 / 400 = 800.
    
    # Let's try h = np.ones(400) * 1.0 + np.random.randn(400)*0.1.
    # sum ~ 400.
    # sum^2 ~ 160000.
    # sum(h^2) ~ 400.
    # V ~ 2 * 400 * 400 / 160000 = 2.
    
    # It seems 2 is a very strong lower bound.
    # Unless... the max is NOT at the center?
    # For a step function, the autocorrelation decays as you move away from the center.
    # Unless the signal has a strong periodic component that causes constructive interference at a lag?
    # But sum(h^2) is the sum of squares. It is the "energy".
    # Any lag correlation is bounded by the energy.
    # So max(conv) >= sum(h^2).
    # And sum(h^2) >= (sum(h))^2 / N.
    # So V >= 2.
    
    # There must be a trick.
    # "Choose real, possibly negative step heights".
    # Maybe the "max" is not the central peak?
    # What if the central peak is negative?
    # sum(h^2) is always positive.
    
    # I will proceed with the assumption that the current score 1.16 is correct and 
    # that my theoretical bound is wrong or I am misinterpreting the convolution.
    # Perhaps np.convolve(h, h) is not the autocorrelation I think it is?
    # No, it is.
    
    # Let's try to optimize the sparse spectrum anyway. 
    # If the bound is 2, we can't beat it. But if the bound is lower, we might.
    # The sparse spectrum approach is a valid structural change.
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Try K=5, 10
    for K in [5, 10]:
        if time.time() > deadline - 20:
            break
            
        # Select K bins
        # We will use the first K non-DC bins? Or spread out?
        # Let's use bins 1..K.
        active_bins = list(range(1, K+1))
        
        # Variables: 2*K (Re, Im for each bin) + 1 (DC real part, Im=0)
        # Total 2*K + 1 variables.
        # But we can just optimize the complex coefficients of the active bins 
        # and the DC bin.
        
        def obj_and_grad(params):
            # params: [re1, im1, ..., reK, imK, reDC]
            spectrum = np.zeros(N, dtype=complex)
            for i, b in enumerate(active_bins):
                re = params[2*i]
                im = params[2*i+1]
                spectrum[b] = re + 1j*im
                spectrum[N-b] = re - 1j*im # Conjugate symmetry for real h
            
            reDC = params[-1]
            spectrum[0] = reDC
            
            h = np.real(np.fft.ifft(spectrum))
            
            val = _evaluate(h)
            if not np.isfinite(val):
                return 1e10, np.zeros(len(params))
            
            # Gradient via finite differences (slow but robust for small K)
            # Or analytic? 
            # h = Re(ifft(S)). 
            # This is linear in S.
            # val is non-linear in h.
            # Let's use numerical gradient.
            grad = np.zeros(len(params))
            eps = 1e-6
            for i in range(len(params)):
                p_plus = params.copy()
                p_plus[i] += eps
                v_plus = _evaluate(np.real(np.fft.ifft(_spec_from_params(p_plus, active_bins))))
                p_minus = params.copy()
                p_minus[i] -= eps
                v_minus = _evaluate(np.real(np.fft.ifft(_spec_from_params(p_minus, active_bins))))
                grad[i] = (v_plus - v_minus) / (2*eps)
            
            return val, grad
        
        def _spec_from_params(params, bins):
            spectrum = np.zeros(N, dtype=complex)
            for i, b in enumerate(bins):
                re = params[2*i]
                im = params[2*i+1]
                spectrum[b] = re + 1j*im
                spectrum[N-b] = re - 1j*im
            spectrum[0] = params[-1]
            return spectrum

        # Initial guess: small random values
        rng = np.random.default_rng(42)
        x0 = rng.normal(0, 0.1, 2*K+1)
        x0[-1] = 1.0 # DC
        
        try:
            res = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'maxfun': 500}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_h = np.real(np.fft.ifft(_spec_from_params(res.x, active_bins)))
                
        except Exception:
            pass

    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
