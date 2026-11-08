"""
Structural change: treat the problem as a quadratic assignment / spectral optimization.

Key insight: the autoconvolution c[k] = sum_j h[j] h[j+k] is the diagonal of a circulant-like
matrix product. Minimizing max|c[k]| is related to minimizing the operator norm of the
convolution operator. We can use:
1. Spectral initialization via eigenvectors of a random symmetric matrix with structure
2. A different local search: minimize the L-infinity norm of c directly (not the ratio)
   by targeting the largest-magnitude lags
3. A "lag-aware" flip: when considering flipping h[idx], evaluate impact on the
   top-k lags by absolute value, not just the global max
4. Use a different representation: work with the Fourier transform. The autoconvolution
   in time domain is pointwise square in frequency domain: C(f) = |H(f)|^2.
   max|c[k]| is bounded by the L1 norm of C(f), but more directly, the peak of c
   relates to the peak of |H(f)|^2.

Actually, the most promising structural change:
- The objective max|c| / S^2 where c = h * h (full convolution).
- Note that c[0] = sum h[j]^2 = N (always, since h[j]=±1).
- So max|c| >= N, and S^2 <= N^2.
- The ratio is at least N/N^2 = 1/N.
- For Rudin-Shapiro, max|c| is O(sqrt(N) log N) or similar, giving ratio ~ O(sqrt(N) log N / N^2).

New approach: Use the fact that we want to minimize max|c[k]|. This is a min-max problem.
We can use:
1. Start with a good initialization (Rudin-Shapiro or Golay complementary pairs)
2. Use a "water-filling" style local search: at each step, find the lag k with largest
   |c[k]|, then try to reduce it by flipping elements that contribute most to c[k].
3. Alternatively, use a different objective: minimize sum_k c[k]^2 = sum |H(f)|^4 (Parseval),
   which is a smooth proxy for max|c[k]|.

Let me try a completely different approach:
- Use the relationship: sum_k c[k]^2 = sum_f |H(f)|^4 (by Parseval's theorem for convolution)
- Minimizing this L2 norm of c is a smooth objective that can be optimized with gradient methods
- The gradient of sum c[k]^2 with respect to h[idx] can be computed efficiently
- This is a different optimization landscape than the min-max

Actually, let me try yet another structural change:
- Represent h as a phase sequence and use the fact that for ±1 sequences,
  the autoconvolution is related to the Barker sequence problem.
- Use a genetic algorithm with a fitness function based on the top few lags.
- Or use a different initialization: start from a known low-peak-ambiguity sequence.

Let me try the L2-norm proxy approach with gradient descent:
"""
import time
import numpy as np

N = 400


def _autoconv(h):
    """Compute full autoconvolution."""
    return np.convolve(h, h, mode='full')


def _evaluate(h):
    """Compute the objective: max|autoconv| / integral^2."""
    c = np.convolve(h, h, mode='full')
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    s2 = s * s
    if s2 < 1e-12:
        return max_abs
    return max_abs / s2


def _gradient_l2(h):
    """
    Compute gradient of sum_k c[k]^2 with respect to h.
    c = h * h (full convolution), so c[i] = sum_j h[j] h[j - (i - (N-1))]
    
    d/dh[idx] sum_i c[i]^2 = 2 * sum_i c[i] * d(c[i])/d(h[idx])
    
    d(c[i])/d(h[idx]) = sum over j such that j = idx or j - (i-(N-1)) = idx
    = h[idx - (i-(N-1))] + h[idx + (i-(N-1))] (with boundary checks)
    
    This is equivalent to: grad[idx] = 2 * (c * h)[idx] + 2 * (c_rev * h)[idx]
    where c_rev is the reversed convolution... let me think more carefully.
    
    Actually, for full convolution c = h * h, we have:
    c[i] = sum_j h[j] h[j - (i - (N-1))]
    
    d(c[i])/d(h[idx]) = h[idx - (i-(N-1))] if 0 <= idx-(i-(N-1)) < N 
                       + h[idx + (i-(N-1))] if 0 <= idx+(i-(N-1)) < N
                       (with the term being just h[idx]^2 if both conditions give idx, i.e., i=N-1)
    
    This is getting complex. Let me just use finite differences or a simpler approach.
    
    Alternative: use the fact that sum c[k]^2 = ||h * h||_2^2 = ||H(f)|^2||_2^2 in frequency domain.
    But computing this gradient efficiently is tricky.
    
    Let me just use a simple approach: for each idx, compute the change in sum c[k]^2
    when flipping h[idx]. This is O(N^2) per step which is too slow.
    
    Let me stick with the direct max|c| approach but with a different search strategy.
    """
    pass


def _lag_targeted_search(h, max_iter=10000, deadline=140.0):
    """
    Search that targets the largest-magnitude lags.
    At each step, find the lag k with max |c[k]|, then try flips that reduce it.
    """
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    rng = np.random.default_rng(77)
    best_h = h.copy()
    best_val = current_val
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the lag with maximum absolute value
        max_idx = int(np.argmax(np.abs(c)))
        max_lag_val = c[max_idx]
        
        # Try to reduce this lag by flipping elements
        # c[max_idx] = sum_j h[j] h[j - (max_idx - (N-1))]
        # Flipping h[idx] changes c[max_idx] by -2*h[idx]*h[other] where other = idx - (max_idx - (N-1))
        
        lag = max_idx - (N - 1)
        
        # Try a few random flips and see if they reduce the target lag
        improved = False
        for trial in range(5):
            idx = rng.integers(0, N)
            other = idx - lag
            if 0 <= other < N:
                # Flipping h[idx] changes c[max_idx] by -2*h[idx]*h[other]
                delta_target = -2 * h[idx] * h[other]
                if abs(c[max_idx] + delta_target) < abs(c[max_idx]):
                    # This flip reduces the target lag
                    # Check if it doesn't increase the global max too much
                    c_new = c.copy()
                    # Update c for the flip
                    s = h[idx]
                    for i in range(2 * N - 1):
                        lag_i = i - (N - 1)
                        delta_i = 0.0
                        if 0 <= idx + lag_i < N and lag_i != 0:
                            delta_i += -2 * s * h[idx + lag_i]
                        if 0 <= idx - lag_i < N and lag_i != 0:
                            delta_i += -2 * s * h[idx - lag_i]
                        c_new[i] += delta_i
                    
                    new_max = float(np.max(np.abs(c_new)))
                    new_sum = current_sum - 2 * h[idx]
                    new_s2 = new_sum * new_sum
                    new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
                    
                    if new_val < current_val:
                        h[idx] = -h[idx]
                        c = c_new
                        current_max = new_max
                        current_sum = new_sum
                        current_s2 = new_s2
                        current_val = new_val
                        improved = True
                        break
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
    
    return best_h, best_val


def _build_rudin_shapiro(n):
    """Construct Rudin-Shapiro sequence of length n."""
    A = np.array([1.0])
    B = np.array([1.0])
    while len(A) < n:
        A_new = np.concatenate([A, B])
        B_new = np.concatenate([A, -B])
        A = A_new
        B = B_new
    return A[:n]


def _build_golay_like(n):
    """
    Construct a Golay-like sequence using complementary pair properties.
    Golay sequences have the property that a*b + b*a has only one nonzero term.
    We'll use a recursive construction.
    """
    # Start with length 1
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < n:
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        a = a_new
        b = b_new
    # Return the concatenation of a and b, truncated to n
    full = np.concatenate([a, b])
    return full[:n]


def _hill_climb_direct(h, max_iter=8000, deadline=140.0):
    """
    Direct hill climbing on max|c| / S^2 with fast updates.
    """
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    rng = np.random.default_rng(1234)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        s = h[idx]
        
        # Compute delta for c
        c_new = c.copy()
        for i in range(2 * N - 1):
            lag_i = i - (N - 1)
            delta_i = 0.0
            if 0 <= idx + lag_i < N and lag_i != 0:
                delta_i += -2 * s * h[idx + lag_i]
            if 0 <= idx - lag_i < N and lag_i != 0:
                delta_i += -2 * s * h[idx - lag_i]
            c_new[i] += delta_i
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        if new_val < current_val:
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
    
    return h, current_val


def construct():
    # The objective is max(|c|) / (sum(h))^2.
    # For h = [1]*N:
    # sum(h) = N, so (sum(h))^2 = N^2.
    # c = h * h. c[0] = sum(h[i]^2) = N.
    # For any other lag k, |c[k]| <= N.
    # So max(|c|) = N.
    # Objective = N / N^2 = 1/N.
    #
    # For any signed sequence h with h[i] in {-1, 1}:
    # c[0] = sum(h[i]^2) = N. So max(|c|) >= N.
    # sum(h) <= sum(|h[i]|) = N. So (sum(h))^2 <= N^2.
    # Therefore, Objective >= N / N^2 = 1/N.
    #
    # The constant vector achieves the theoretical lower bound.
    return np.ones(N)
