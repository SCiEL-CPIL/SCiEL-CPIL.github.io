"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max(signed_steps, interval_width):
    """Compute the max absolute value of the autoconvolution of a step function."""
    # signed_steps: array of N values
    # The function is defined on [-1/4, 1/4] with N equal intervals
    # Each interval has width = (1/2) / N
    # 
    # The autoconvolution (f*f)(x) = integral of f(t)f(x-t) dt
    # For a step function, this can be computed efficiently.
    
    # Using FFT-based convolution for efficiency
    # The step function sampled at the interval centers
    # Actually, let's use the fact that autoconvolution of a step function
    # can be computed via correlation
    
    # f is piecewise constant. Let's compute f*f using the structure.
    # f(t) = sum_i a_i * I(t in interval_i)
    # f*f(x) = sum_i sum_j a_i a_j * |overlap(interval_i, interval_j shifted by x)|
    
    # This is expensive for N=400. Let's use FFT.
    # Represent f as a vector of values at fine resolution, then convolve.
    
    # Better: use the discrete structure.
    # The autoconvolution of a step function with N intervals can be computed
    # by noting that it's a piecewise linear function with at most 2N-1 pieces.
    
    # For efficiency, let's use numpy's correlate or fft.
    # Create a dense representation with enough resolution.
    
    # Actually, for N=400, we can compute the autoconvolution at the 
    # critical points (shifts that align interval boundaries).
    
    # The autoconvolution (f*f)(x) for step function:
    # Let the intervals be [x_i, x_{i+1}] with values a_i
    # (f*f)(x) = sum_i a_i * (measure of {t: t in interval_i and t-x in domain of f with f(t-x) defined})
    #           = sum_i a_i * integral over t in interval_i of f(t-x) dt
    
    # This equals sum_i a_i * (sum_j a_j * |interval_i ∩ (interval_j + x)|)
    
    # Let's compute this at a fine grid of x values.
    
    # The support of f*f is [-1/2, 1/2] since f is supported on [-1/4, 1/4]
    # The critical x values are where interval boundaries align: multiples of interval_width
    
    # Number of critical points: about 2*N
    # Let's evaluate at all shifts that are multiples of interval_width
    
    w = interval_width
    # x values from -N*w to N*w in steps of w
    x_vals = np.arange(-N, N+1) * w
    # But we only need x where the overlap is non-trivial, which is [-1/2, 1/2] = [-N*w, N*w]
    
    # For each x, compute (f*f)(x)
    # (f*f)(x) = sum_i a_i * sum_j a_j * |I_i ∩ (I_j + x)|
    
    # |I_i ∩ (I_j + x)| = max(0, min(x_{i+1}, x_j + x + w) - max(x_i, x_j + x))
    # where I_i = [x_i, x_{i+1}], x_i = -1/4 + i*w
    
    # This is O(N^2) per x, and O(N) x values, so O(N^3) total. For N=400, that's 64M ops.
    # Might be too slow. Let's use FFT.
    
    # FFT approach: 
    # Create a dense array representing f, zero-pad, FFT, multiply, IFFT.
    # Resolution: need enough points to capture the step function.
    # Use M = 2*N or 4*N points for the dense representation.
    
    M = 4 * N  # oversample by 4x
    # Create dense representation of f
    # f is supported on [-1/4, 1/4], length 1/2
    # Map to [0, M) where the function is nonzero in the middle N*4/M * ... 
    # Actually, let's think in terms of the dense array.
    
    # The total domain for f*f is [-1/2, 1/2], length 1.
    # Let's use M points over [-1/2, 1/2].
    # f is nonzero on [-1/4, 1/4], which is M/2 points in the middle.
    
    # Create dense f array of length M
    f_dense = np.zeros(M)
    # Map interval i (i=0..N-1) to dense indices
    # Interval i covers [-1/4 + i*w, -1/4 + (i+1)*w]
    # In the dense array over [-1/2, 1/2], position t maps to index (t + 1/2) * M
    # Interval i: t from -1/4 + i*w to -1/4 + (i+1)*w
    # Index from (-1/4 + i*w + 1/2) * M = (1/4 + i*w) * M to (1/4 + (i+1)*w) * M
    # w = 1/(2*N), so i*w = i/(2*N)
    # Index from M*(1/4 + i/(2*N)) to M*(1/4 + (i+1)/(2*N))
    # = M/4 + i*M/(2*N) to M/4 + (i+1)*M/(2*N)
    # M/(2*N) = 4*N/(2*N) = 2
    # So interval i maps to dense indices [M/4 + 2*i, M/4 + 2*(i+1))
    
    start_idx = M // 4  # = N
    for i in range(N):
        f_dense[start_idx + 2*i : start_idx + 2*(i+1)] = signed_steps[i]
    
    # Now compute autoconvolution via FFT
    # f*f(x) = integral f(t)f(x-t)dt
    # In discrete terms, this is a correlation.
    # Use ifft(fft(f_dense) * conj(fft(f_dense))) for circular correlation,
    # but we need linear convolution. Zero-pad.
    
    # For linear convolution, we need to zero-pad to at least 2*M-1
    # But since f is already zero outside its support, and we're doing circular
    # convolution on length M, it's equivalent to linear if M is large enough.
    # f is supported on M/2 consecutive points, so f*f is supported on M points.
    # Circular convolution of length M with support M/2 gives correct linear result.
    
    F = np.fft.fft(f_dense)
    ff = np.fft.ifft(F * np.conj(F)).real
    # ff[k] approximates (f*f)(x_k) * (1/M) where x_k = k/M - 1/2 ... 
    # Actually, the FFT convolution gives the discrete convolution.
    # The continuous autoconvolution is approximated by sum_j f_dense[j] * f_dense[(j-k) % M] * (1/M)
    # which is what ifft(fft(f)*conj(fft(f))) gives (up to scaling).
    
    # The scaling: ifft(fft(f)*conj(fft(f)))[k] = sum_j f[j] f[(j-k)%M]
    # The continuous integral is approximated by (1/M) * sum_j f_dense[j] f_dense[(j-k)%M] * (1/M) * M
    # = (1/M) * sum_j f_dense[j] f_dense[(j-k)%M]
    # Wait, let me be more careful.
    
    # f_dense[j] approximates f(t_j) where t_j = -1/2 + j/M, j=0..M-1
    # The integral (f*f)(x) = integral f(t) f(x-t) dt ≈ (1/M) sum_j f_dense[j] f_dense[j - k] 
    # where x = k/M (approximately)
    # So (f*f)(k/M) ≈ (1/M) * sum_j f_dense[j] f_dense[(j-k) % M]
    # = (1/M) * ff[k]
    
    autoconv = ff / M
    # The x values are x_k = k/M - 1/2 for k=0..M-1? Or k/M?
    # In the circular convolution, shift k corresponds to x = k * (1/M) in some sense.
    # The max absolute value is what we want.
    
    return np.max(np.abs(autoconv))


def _score(signed_steps):
    """Compute the upper bound on C3 (the raw value to minimize)."""
    w = 1.0 / (2 * N)  # interval width
    return _autoconv_max(signed_steps, w)


def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Initialize with random ±1
    np.random.seed(42)
    current = np.random.choice([-1.0, 1.0], size=N)
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Local search: flip individual signs
    # Precompute some structure for fast score evaluation
    # Actually, _score is O(M log M) with M=4*N=1600, which is fast.
    # Let's estimate: FFT of 1600 points is very fast, maybe ~100us
    # So we can do many iterations.
    
    # Try flipping each position and see if it improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Flip position i
            current[i] = -current[i]
            new_score = _score(current)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current.copy()
                improved = True
            else:
                current[i] = -current[i]  # revert
    
    return best
# EVOLVE-BLOCK-END
