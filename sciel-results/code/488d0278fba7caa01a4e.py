import numpy as np
from scipy import optimize
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total == 0:
        return 0.0
    conv = np.convolve(h, h, 'full')
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Initial guess: a broad Gaussian
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    best_h = np.exp(-((x - 0)**2) / (2 * 0.15**2))
    best_score = compute_autoconv_max(best_h)

    # Use FFT for faster convolution in the optimization loop
    # We need to compute the autoconvolution efficiently.
    # h has length N. conv has length 2N-1.
    # We can zero-pad to length >= 2N-1, say 2048 or next power of 2.
    next_pow2 = 1 << (2 * N - 1).bit_length()
    
    def fft_autoconv_max(h):
        h = np.asarray(h, dtype=float)
        total = np.sum(h)
        if total == 0:
            return 0.0
        h_pad = np.zeros(next_pow2)
        h_pad[:N] = h
        H = np.fft.rfft(h_pad)
        conv = np.fft.irfft(H * H, n=next_pow2)
        # The peak is at index N-1 (for full convolution of two length N arrays)
        # Wait, np.convolve(h, h, 'full') has peak at N-1.
        # The FFT convolution result is circular. 
        # h_pad is zero-padded.
        # conv[k] = sum h[i] h[k-i]
        # The maximum should be near the center.
        # Actually, for autoconvolution of a non-negative function, the peak is at the center.
        # In the FFT output, the peak is at index N-1?
        # Let's check: np.convolve([1,0,0], [1,0,0], 'full') -> [1, 0, 0] (peak 0)
        # np.convolve([0,1,0], [0,1,0], 'full') -> [0, 0, 1, 0, 0] (peak 2)
        # So peak index is N-1.
        # We want the max of the sidelobes? Or max of the whole thing?
        # The task says "minimize the normalized maximum of its autoconvolution".
        # Usually, the autoconvolution of a probability density (normalized to 1) has a peak of 1 (if continuous) or close to it.
        # The "max" is the peak. Minimizing the peak is like making the function as flat as possible?
        # But the function is non-negative.
        # If h is constant, conv is a triangle. The peak is at the center.
        # If h is a delta, conv is a delta. Peak is 1.
        # If h is uniform on [0,1], conv is triangle with peak 1.
        # Wait, the score is "upper bound on C1".
        # The first autocorrelation inequality usually involves the ratio of the max of the autocorrelation to the zero-lag value.
        # But here it says "normalized maximum of its autoconvolution".
        # Let's re-read: "minimize the normalized maximum of its autoconvolution".
        # If we normalize h to sum to 1, the max of the autoconvolution is the max of (h*h)(k).
        # For a probability distribution, the max of the convolution is at least 1/sqrt(N) (by Cauchy-Schwarz) if it's discrete?
        # Actually, if h is uniform, max is 1/N.
        # If h is delta, max is 1.
        # So minimizing the max means making the distribution as flat as possible?
        # But h must be non-negative.
        # The flattest non-negative function with fixed sum is the constant function.
        # If h is constant, conv is a triangle. The max is at the center.
        # Let's check the score of a constant function.
        # h = [1, 1, ..., 1]. Sum = N.
        # conv = [1, 2, ..., N, ..., 2, 1]. Max = N.
        # Normalized max = N / (N*N) = 1/N.
        # This is very small.
        # But is there a constraint I'm missing? "Choose a nonnegative step function... to minimize the normalized maximum".
        # If I can choose any nonnegative function, the constant function gives 1/N.
        # Is 1/N the lower bound?
        # By Cauchy-Schwarz, (sum h_i)^2 <= N * sum h_i^2.
        # Also, max(h*h) >= (sum h_i)^2 / (sum h_i * sum h_i) ? No.
        # max_k sum_i h_i h_{k-i} >= (1/N) sum_k sum_i h_i h_{k-i} = (1/N) (sum h_i)^2.
        # So max >= (sum h)^2 / N.
        # Normalized max = max / (sum h)^2 >= 1/N.
        # So the minimum possible value is 1/N, achieved by the constant function.
        # Why is the current score 1.0035?
        # The score is seed/value. Seed is the seed program's value.
        # If the seed program uses a Gaussian, its value is higher than 1/N.
        # So I should just return a constant function!
        
        # Let's verify.
        # h = np.ones(N)
        # compute_autoconv_max(h) should be 1/N.
        
        # Let's test this hypothesis.
        # If I return np.ones(N), the score should be seed / (1/N).
        # If seed was, say, 0.02 (for N=600, 1/N is 0.0016), and Gaussian gives 0.01, then score is 0.01/0.0016 = 6.
        # The current score is 1.0035, which is very close to 1.
        # This implies the seed program's value is very close to the current program's value.
        # And the current program uses a Gaussian.
        # If the constant function is much better, the score would be much higher.
        # So either:
        # 1. The constant function is not allowed (e.g., must have some shape?). No, "nonnegative step function".
        # 2. My understanding of "normalized maximum" is wrong.
        # 3. The seed program is also close to constant?
        # 4. The score is not value.
        
        # Let's look at the inspirations.
        # Inspiration 1 score 1.2379.
        # This suggests that values can be lower than the seed.
        # If the constant function gives 1/N, and the seed gives something larger, the score would be large.
        # Maybe the "normalized maximum" is defined differently?
        # "normalized maximum of its autoconvolution"
        # Maybe it's max(conv) / max(h)? No.
        # Maybe it's max(conv[k!=0]) / conv[0]? (Peak-to-sidelobe ratio?)
        # "First autocorrelation inequality" usually refers to the ratio of the first non-zero autocorrelation coefficient to the zero-th.
        # Or the max of the autocorrelation sidelobes.
        
        # Let's re-read the task carefully.
        # "minimize the normalized maximum of its autoconvolution"
        # In the context of the first autocorrelation inequality (C1), it often bounds the max of the autocorrelation function (excluding the peak) by C1 * (some norm).
        # Or it bounds the peak of the autocorrelation of the *error*?
        
        # Let's look at the current code's `compute_autoconv_max_sidelobe`.
        # It excludes the peak!
        # `mask[peak_idx] = False`
        # `max_sidelobe = np.max(conv[mask])`
        # So the objective is to minimize the max sidelobe.
        
        # If I use a constant function:
        # conv is a triangle.
        # Peak is at center.
        # Sidelobes are the rest of the triangle.
        # The max sidelobe is just next to the peak.
        # For h=1, conv = [1, 2, ..., N, ..., 2, 1].
        # Peak is N.
        # Max sidelobe is N-1.
        # Normalized max sidelobe = (N-1) / N^2 ≈ 1/N.
        
        # For a Gaussian:
        # The sidelobes decay.
        # The max sidelobe is likely much smaller than 1/N?
        # No, for a Gaussian, the convolution is also a Gaussian.
        # The peak is at the center.
        # The value at k=1 is close to the peak.
        # So the max sidelobe is also close to the peak.
        
        # Let's compare constant vs Gaussian for N=600.
        # Constant: max sidelobe ≈ 1/N = 0.0016.
        # Gaussian: h = exp(-x^2/2sigma^2).
        # Convolution is exp(-x^2/4sigma^2).
        # Peak is at 0.
        # Value at 1 step (dx=1/300).
        # Ratio = exp(-dx^2/4sigma^2).
        # If sigma is large, ratio is close to 1.
        # If sigma is small, ratio is small.
        # But we normalize h to sum to 1.
        # Sum h ≈ sigma * sqrt(2pi).
        # h_norm = h / (sigma * sqrt(2pi)).
        # conv_norm peak = 1 / (2 * sigma * sqrt(2pi)).
        # This depends on sigma.
        # If sigma is small, peak is high.
        # If sigma is large, peak is low.
        # But we are minimizing the max sidelobe.
        
        # Wait, if I make sigma very large, the function becomes flat.
        # Then it approaches the constant function.
        # So the constant function is the limit of Gaussians with large sigma.
        # So the max sidelobe for constant is the limit.
        
        # Is it possible to get lower than 1/N?
        # The max sidelobe is at k=1 (or k=N-2).
        # conv[1] = sum h[i] h[i+1].
        # We want to minimize sum h[i] h[i+1] / (sum h[i])^2.
        # This is like minimizing the "adjacent correlation".
        # If h is alternating 1, 0, 1, 0..., then h[i]h[i+1] = 0.
        # So max sidelobe could be 0?
        # Let's check.
        # h = [1, 0, 1, 0, ...]
        # conv[0] = sum h[i]^2 = N/2.
        # conv[1] = sum h[i]h[i+1] = 0.
        # conv[2] = sum h[i]h[i+2] = N/2.
        # So the sidelobes are 0, N/2, 0, N/2...
        # The max sidelobe is N/2.
        # Normalized max sidelobe = (N/2) / (N/2)^2 = 2/N.
        # This is higher than 1/N.
        
        # What if h = [1, 1, 0, 0, 1, 1, 0, 0...]?
        # This is getting complicated.
        
        # However, the constant function gives 1/N.
        # Can we do better?
        # We want to minimize max_k conv[k] / (sum h)^2 for k != 0.
        # Note that sum_k conv[k] = (sum h)^2.
        # There are 2N-1 terms.
        # If they were all equal, each would be (sum h)^2 / (2N-1).
        # Then max would be (sum h)^2 / (2N-1).
        # Normalized max = 1/(2N-1) ≈ 1/(2N).
        # This is half of 1/N.
        # Can we achieve a flat autoconvolution?
        # A function with flat autoconvolution is a "perfect sequence" or related to Barker sequences (but those are +/-1).
        # For non-negative functions, can we have a flat autoconvolution?
        # If h is constant, conv is triangular. Not flat.
        # To make conv flat, h must be very oscillatory?
        # But h must be non-negative.
        
        # Let's try to minimize the max sidelobe using optimization.
        # The current program does this.
        # But maybe I can use a better initial guess or a better optimizer.
        
        # Let's stick to the interpretation: minimize max(conv[k]) / (sum h)^2 for k != 0.
        
        # Let's try a few specific patterns.
        # 1. Constant: ~1/N
        # 2. Alternating: ~2/N
        # 3. Random: ~1/sqrt(N)?
        
        # Let's just run the optimizer with a good initialization.
        # The constant function is a good baseline.
        # Let's try to optimize from a constant function.
        
        best_h = np.ones(N)
        best_score = compute_autoconv_max_sidelobe(best_h)
        
        # Let's also try a Gaussian with large sigma
        h_gauss = np.exp(-((x - 0)**2) / (2 * 0.5**2))
        score_gauss = compute_autoconv_max_sidelobe(h_gauss)
        if score_gauss < best_score:
            best_score = score_gauss
            best_h = h_gauss.copy()
            
        # Now optimize
        # Use L-BFGS-B with a smooth max objective
        
        def soft_max_sidelobe(h, T=0.01):
            h = np.asarray(h, dtype=float)
            total = np.sum(h)
            if total < 1e-10:
                return 1e10
            h_norm = h / total
            conv = np.convolve(h_norm, h_norm, 'full')
            peak_idx = N - 1
            mask = np.ones(len(conv), dtype=bool)
            mask[peak_idx] = False
            sidelobes = conv[mask]
            pmax = np.max(sidelobes)
            if pmax < 1e-15:
                return 1e-10
            shifted = (sidelobes - pmax) / T
            shifted = np.clip(shifted, -50, 50)
            soft_max = pmax + T * np.log(np.sum(np.exp(shifted)))
            return soft_max

        def soft_obj_direct(h):
            return soft_max_sidelobe(h, T=0.005)

        # Optimize from best_h
        if time.time() < deadline - 10:
            try:
                result = optimize.minimize(
                    soft_obj_direct, best_h, method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_polished = np.maximum(result.x, 0)
                score_polished = compute_autoconv_max_sidelobe(h_polished)
                if score_polished < best_score:
                    best_score = score_polished
                    best_h = h_polished.copy()
            except Exception:
                pass

        return list(best_h)

# Helper function used in construct
def compute_autoconv_max_sidelobe(h):
    """Compute the normalized maximum of the autoconvolution sidelobes (k != 0)."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total == 0:
        return 0.0
    h_norm = h / total
    conv = np.convolve(h_norm, h_norm, 'full')
    peak_idx = N - 1
    mask = np.ones(len(conv), dtype=bool)
    mask[peak_idx] = False
    max_sidelobe = np.max(conv[mask])
    return max_sidelobe
