"""
Strategy: Direct Non-Parametric Coordinate Descent with FFT-Accelerated Autocorrelation.

This approach structurally differs from the current program's parametric sinusoidal
basis optimization in the following ways:
1. **Representation**: No parametric model (no chirps, no sinusoids). We optimize
   the full 400-dimensional vector h directly.
2. **Optimization Method**: Coordinate Descent (cyclic or random) with 1D line search.
   This is a first-order/zeroth-order method on the raw variables, not a gradient
   method on a low-dimensional parameter space.
3. **Acceleration**: We use FFT to compute the autocorrelation in O(N log N) instead
   of O(N^2), allowing for thousands of objective evaluations per second, which makes
   the line search feasible.

The objective is to minimize V(h) = 2*N*max(h*h) / (sum(h))^2.
We fix sum(h) = 1 (scaling invariance) and minimize max(h*h).
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit


def _autocorr_max_fft(h):
    """
    Compute the maximum of the full autocorrelation of h using FFT.
    h is a 1D array of length N.
    Returns the max value of conv(h, h).
    """
    # Full convolution length is 2N-1
    # We can use FFT of size 2N
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    # Power spectrum
    P = H * np.conj(H)
    # Inverse FFT to get autocorrelation
    # Note: np.fft.irfft returns real values
    # The autocorrelation is centered at N-1 in the full convolution output?
    # np.convolve(h, h) has length 2N-1.
    # FFT method:
    # If we do ifft(|fft(h, 2N)|^2), we get circular autocorrelation.
    # For linear autocorrelation, we need to handle the wrap-around.
    # Actually, for zero-padded h, the circular autocorrelation of length 2N
    # contains the linear autocorrelation in the first N elements (lag 0 to N-1)
    # and the negative lags in the last N elements (lag -(N-1) to -1)?
    # Let's verify.
    # h = [h0, h1, ..., h_{N-1}, 0, ..., 0]
    # (h*h)_k = sum_i h_i h_{i+k}
    # This is the standard autocorrelation.
    # The FFT method computes the circular autocorrelation of the zero-padded signal.
    # Since the signal is zero-padded to 2N, and the support of h is N,
    # the linear autocorrelation is exactly the circular autocorrelation
    # for lags k in [-(N-1), N-1].
    # The output of ifft(|fft|^2) has length 2N.
    # Index 0 corresponds to lag 0.
    # Index k corresponds to lag k.
    # Index 2N-k corresponds to lag -k.
    # So we can just take the max of the entire output?
    # The max of the signed entries.
    # The autocorrelation is symmetric: C_k = C_{-k}.
    # So we just need to look at the first N elements (lags 0 to N-1) 
    # or the whole array.
    
    ac = np.fft.irfft(P, n=n_fft)
    # The linear autocorrelation is in ac[0:N] (lags 0 to N-1)
    # and ac[N:2N] contains lags N to 2N-1 which are actually negative lags wrapped?
    # Actually, for zero-padded signal of length 2N, the circular autocorrelation
    # at lag k is sum_{i=0}^{2N-1} x[i] x[(i+k) mod 2N].
    # Since x[i]=0 for i>=N, the non-zero terms are only when i < N and (i+k) mod 2N < N.
    # This effectively computes the linear autocorrelation for all lags.
    # The max of the full autocorrelation is the max of ac.
    
    return np.max(ac)


def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use FFT for speed
    max_ac = _autocorr_max_fft(h)
    return abs(2.0 * N * max_ac / (s * s))


def _normalize_sum(h):
    """Normalize h so that sum(h) = 1."""
    s = np.sum(h)
    if s == 0:
        return h
    return h / s


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Initialize with a flat-top window, which has low sidelobes
    # This is a good starting point for minimizing peak autocorrelation
    n = np.arange(N)
    # Flat-top window coefficients
    # Standard flat-top: a0=0.21557895, a1=0.41663158, a2=0.277263158, a3=0.083578947, a4=0.006947368
    # We can just use a cosine taper or rectangular.
    # Let's try a Gaussian window first, then refine.
    # Or simply start with ones (the seed) and let coordinate descent move it.
    # Starting with ones gives value 2.0.
    h = np.ones(N)
    h = _normalize_sum(h)
    
    best_h = h.copy()
    best_val = _evaluate(best_h)
    
    # Coordinate Descent
    # We will cycle through coordinates.
    # For each coordinate, we perform a 1D search.
    
    # To handle the non-convexity and local minima, we can use
    # Randomized Coordinate Descent with restarts or large steps.
    
    # Line search function for a single coordinate
    def line_search_coord(idx, h_current):
        """
        Find the best value for h[idx] to minimize the objective.
        Uses ternary search or golden section search.
        """
        # Range for h[idx]
        # The other elements sum to S_other.
        # To keep sum(h) = 1, we don't actually need to renormalize in the search
        # if we define the objective in terms of the raw h and the final normalization.
        # But it's easier to work with normalized h.
        # If we change h[idx] by delta, we must renormalize the whole vector.
        # This couples all coordinates.
        
        # Alternative: Fix sum(h)=1.
        # Let h_old be the current vector.
        # We want to change h[idx] to x.
        # New vector h_new: h_new[idx] = x, h_new[j] = h_old[j] for j != idx.
        # But then sum(h_new) = x + sum(h_old[j], j!=idx).
        # Let S_other = sum(h_old) - h_old[idx] = 1 - h_old[idx].
        # sum(h_new) = x + S_other.
        # Normalized h_new: h_norm = h_new / (x + S_other).
        # Objective: V(x) = 2*N*max(h_norm * h_norm) / 1^2.
        # We can compute max(h_norm * h_norm) efficiently.
        
        h_old = h_current
        S_other = 1.0 - h_old[idx]
        
        def obj_x(x):
            # Construct h_new
            # h_new is h_old with idx set to x
            # h_norm = h_new / (x + S_other)
            # We need max(h_norm * h_norm)
            # h_norm * h_norm = (h_new * h_new) / (x + S_other)^2
            # So max(h_norm * h_norm) = max(h_new * h_new) / (x + S_other)^2
            
            # Compute h_new * h_new
            # h_new differs from h_old only at idx.
            # h_new[i] = h_old[i] for i != idx
            # h_new[idx] = x
            
            # Autocorrelation of h_new
            # This is still O(N log N) with FFT.
            # We can just build the array and FFT it.
            h_new = h_old.copy()
            h_new[idx] = x
            
            denom = x + S_other
            if abs(denom) < 1e-12:
                return 1e10
                
            max_ac_new = _autocorr_max_fft(h_new)
            val = max_ac_new / (denom * denom)
            return val
            
        # Golden Section Search
        # We need a range for x.
        # h_old[idx] is the current value.
        # Let's search in [h_old[idx] - 5, h_old[idx] + 5]
        # Or relative range.
        center = h_old[idx]
        # If center is 0, search [-1, 1]. If center is 1, search [0, 2].
        # Let's use a range that allows significant changes.
        # Since we normalize, the scale of h doesn't matter, only the shape.
        # But the "shape" is defined by relative values.
        # Let's search in a range that covers reasonable values.
        # A flat window has values ~1/N.
        # A peaked window might have values ~1.
        # Let's search in [-2, 2] relative to the mean?
        # Or just [-1, 1] if we start with ones (normalized to 1/N).
        # Wait, if h = ones(N), sum=400. Normalized h = 1/400.
        # So h_old[idx] is 0.0025.
        # Let's search in [-0.01, 0.01].
        
        # Adaptive range: start small, expand if needed.
        # For simplicity, let's use a fixed range around the current value.
        # The values of h are roughly 1/N.
        # Let's use a range of +/- 0.05.
        
        a = center - 0.05
        b = center + 0.05
        
        # Ensure we don't make the denominator zero
        # denom = x + S_other. S_other is approx 1.
        # So x should not be -1.
        # Our range is small, so we are safe.
        
        gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = obj_x(c)
        fd = obj_x(d)
        
        # Max 20 iterations
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = obj_x(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = obj_x(d)
                
        # Return the best x
        x_best = (a + b) / 2
        return x_best

    # Main Loop: Cyclic Coordinate Descent
    # We do multiple passes.
    # To escape local minima, we can add noise or use larger steps occasionally.
    
    # Initialize with a better starting point?
    # Let's try a Gaussian window.
    # h_gauss = exp(-n^2 / (2*sigma^2))
    # This has a lower peak autocorrelation than a rectangular window?
    # Actually, the peak of autocorrelation of a Gaussian is at lag 0 and is the sum of squares.
    # For a rectangular window, peak is sum of squares (at lag 0) but sidelobes are high.
    # We want to minimize the MAX autocorrelation.
    # The max is always at lag 0 if h is non-negative and sum of squares is dominant.
    # But if we allow negative values, we might reduce the lag 0 peak?
    # No, lag 0 is sum h_i^2. This is always positive.
    # Can we make lag 0 smaller than other lags?
    # If we make h have alternating signs, lag 0 is still sum h_i^2.
    # Lag 1 is sum h_i h_{i+1}. If h_i = (-1)^i, lag 1 is negative.
    # But lag 0 is positive.
    # The objective is abs(2*N*max(conv) / sum^2).
    # If max(conv) is at lag 0, it is sum h_i^2.
    # If we can make another lag larger than lag 0, we might increase the max?
    # No, we want to MINIMIZE the max.
    # So we want the autocorrelation to be as flat as possible.
    # Ideally, all lags are equal?
    # Sum of autocorrelation values is sum h_i^2 (Parseval).
    # If all lags are equal, each is sum h_i^2 / (2N-1).
    # Then max is small.
    # This is the "Golay" or "Complementary" ideal.
    
    # Let's start with a random vector to see if CD can find a flat spectrum.
    # Or start with the seed (ones).
    
    # Let's run CD for a fixed number of iterations or until time limit.
    
    # To improve performance, we can use a "block" CD where we update a few coords at once?
    # No, 1D line search is robust.
    
    # Optimization: Pre-compute FFT of h_old?
    # Changing one element changes the FFT linearly.
    # H_new = H_old + (x - h_old[idx]) * e^{-j 2 pi k idx / N}
    # This allows O(N) update of the spectrum instead of O(N log N) FFT.
    # Then |H_new|^2 can be updated?
    # |H_new|^2 = |H_old + delta * E|^2 = |H_old|^2 + 2 Re(H_old * conj(delta * E)) + |delta|^2 |E|^2
    # |E|^2 = 1.
    # So we can update the power spectrum in O(N).
    # Then ifft to get autocorrelation in O(N log N).
    # This saves the forward FFT.
    
    # Let's implement this optimization in the line search.
    
    def line_search_coord_fast(idx, h_current, H_current):
        """
        Fast line search using FFT update.
        """
        h_old = h_current
        S_other = 1.0 - h_old[idx]
        
        # Pre-compute the exponential term for the update
        # E[k] = exp(-j 2 pi k idx / N_fft)
        # We use n_fft = 2N
        n_fft = 2 * N
        k = np.arange(n_fft // 2 + 1)
        E = np.exp(-1j * 2 * np.pi * k * idx / n_fft)
        
        def obj_x_fast(x):
            delta = x - h_old[idx]
            if abs(delta) < 1e-15:
                # No change
                max_ac = _autocorr_max_fft(h_old)
                denom = 1.0
                return max_ac / (denom * denom)
            
            # Update Spectrum
            # H is rfft, so length n_fft//2 + 1
            # H_new = H_old + delta * E
            H_new = H_current + delta * E
            
            # Power Spectrum
            P_new = H_new * np.conj(H_new)
            
            # Inverse FFT
            ac_new = np.fft.irfft(P_new, n=n_fft)
            
            denom = x + S_other
            if abs(denom) < 1e-12:
                return 1e10
            
            max_ac_new = np.max(ac_new)
            val = max_ac_new / (denom * denom)
            return val

        center = h_old[idx]
        a = center - 0.05
        b = center + 0.05
        
        gr = (np.sqrt(5) - 1) / 2
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = obj_x_fast(c)
        fd = obj_x_fast(d)
        
        for _ in range(15):  # Reduced iterations for speed
            if time.time() > deadline - 10:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = obj_x_fast(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = obj_x_fast(d)
                
        x_best = (a + b) / 2
        return x_best

    # Initialize H
    h = _normalize_sum(h)
    H = np.fft.rfft(h, n=2*N)
    
    best_val = _evaluate(h)
    best_h = h.copy()
    
    # CD Loop
    # We will do passes.
    # To avoid getting stuck, we can randomize the order or add noise.
    
    rng = np.random.default_rng(42)
    
    # Time budget for CD
    while time.time() < deadline - 5:
        # Randomize order for this pass
        indices = rng.permutation(N)
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            # Perform line search
            new_val_coord = line_search_coord_fast(idx, h, H)
            
            # Update h
            old_val = h[idx]
            if abs(new_val_coord - old_val) > 1e-10:
                h[idx] = new_val_coord
                
                # Update H
                delta = new_val_coord - old_val
                k = np.arange(2*N // 2 + 1)
                E = np.exp(-1j * 2 * np.pi * k * idx / (2*N))
                H += delta * E
                
                # Normalize h to sum 1?
                # The line search assumed sum(h)=1.
                # But changing h[idx] changes the sum.
                # We need to renormalize.
                # If we renormalize, H changes globally.
                # This breaks the fast update.
                
                # Solution: Don't renormalize in the inner loop.
                # Instead, let the sum drift?
                # No, the objective depends on the sum.
                # The line search function obj_x_fast assumed the denominator was (x + S_other).
                # And it returned max_ac / denom^2.
                # This is exactly the objective value for the vector h_new (unnormalized)
                # if we define the objective as 2*N*max(h*h)/sum(h)^2.
                # So we don't need to normalize h!
                # We just need to keep track of the actual h.
                
                # However, the "shape" is what matters.
                # If the sum drifts to 0, we have a problem.
                # We should periodically renormalize.
                
                # Let's check the sum periodically.
                if idx % 50 == 0:
                    s = np.sum(h)
                    if abs(s) < 1e-6:
                        # Reset to best_h
                        h = best_h.copy()
                        H = np.fft.rfft(h, n=2*N)
                        break
                    # Renormalize
                    h = h / s
                    H = H / s
                    
                    # Update best
                    v = _evaluate(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
            else:
                # No change, just check time
                pass
        
        # After a pass, check best
        v = _evaluate(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
            
        # Occasionally, add noise to escape local minima
        if time.time() > deadline - 20:
            # Final refinement
            # Use a smaller step size?
            pass
            
    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h
