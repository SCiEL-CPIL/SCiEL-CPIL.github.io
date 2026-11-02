"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _circ_acf(x):
    """Circular autocorrelation of x (length n), via FFT. Returns length-n array."""
    n = len(x)
    X = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(X * np.conj(X), n=2*n)[:n]
    return acf

def _psl(x):
    """Peak sidelobe level: max |acf[k]| for k != 0."""
    acf = _circ_acf(x)
    return np.max(np.abs(acf[1:]))

def _psl_with_lag(x):
    acf = _circ_acf(x)
    idx = np.argmax(np.abs(acf[1:])) + 1
    return np.abs(acf[idx]), idx

def _flip_delta_acf(x, i):
    """
    Return the change in circular autocorrelation when flipping x[i].
    delta[k] = -2*x[i]*(x[(i+k)%n] + x[(i-k)%n])  for k = 0..n-1
    """
    n = len(x)
    s = x[i]
    # Build the vector v where v[k] = x[(i+k)%n] + x[(i-k)%n]
    # This is a sum of two shifted copies of x
    # x[(i+k)%n] for k=0..n-1 is np.roll(x, -i)
    # x[(i-k)%n] for k=0..n-1 is np.roll(x, i)
    v = np.roll(x, -i) + np.roll(x, i)
    return -2.0 * s * v

def _greedy_flip(x, deadline, max_rounds=2000):
    """
    Greedy: repeatedly flip the single bit that most reduces PSL.
    Use O(N) delta evaluation per candidate, O(N^2) per round.
    """
    x = x.astype(float).copy()
    n = len(x)
    acf = _circ_acf(x)
    best_peak = np.max(np.abs(acf[1:]))
    
    no_improve = 0
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if best_peak < 1.0:
            break
        
        best_gain = 1e-12
        best_idx = -1
        
        # Evaluate all N flips
        for i in range(n):
            delta = _flip_delta_acf(x, i)
            new_acf = acf + delta
            new_peak = np.max(np.abs(new_acf[1:]))
            gain = best_peak - new_peak
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            x[best_idx] = -x[best_idx]
            acf = _circ_acf(x)
            best_peak = np.max(np.abs(acf[1:]))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                break
    
    return x, best_peak

def _pair_flip(x, deadline, max_rounds=500):
    """
    Try flipping pairs of bits. O(N^2) pairs, each O(N) to evaluate delta.
    Total O(N^3) per round, but we can be smarter: 
    For pair (i,j), delta = delta_i + delta_j + cross_term.
    cross_term: flipping both i and j, the interaction term.
    Actually delta for simultaneous flip of i and j:
    The autocorr change is not simply additive because x[i] and x[j] both change.
    Let's just compute: new_x = x with x[i] and x[j] flipped, then acf.
    But that's O(N log N) per pair. Too slow for N^2 pairs.
    
    Alternative: precompute delta_i for each i, then for pair (i,j):
    delta_ij = delta_i + delta_j + 4*x[i]*x[j]*(e_i + e_j)  ... this gets complicated.
    
    Let's use a simpler approach: for each i, precompute the "influence vector" 
    that tells us how acf changes. Then for pair (i,j), the combined delta is:
    delta_i + delta_j - 4*x[i]*x[j]*(delta contribution from interaction)
    
    Actually, let's think about it differently. The autocorrelation is a quadratic form.
    acf[k] = sum_j x[j]*x[(j+k)%n]
    
    When we flip x[i] -> -x[i] and x[j] -> -x[j]:
    New acf[k] = acf[k] - 2*x[i]*(x[(i+k)%n]+x[(i-k)%n]) - 2*x[j]*(x[(j+k)%n]+x[(j-k)%n])
                + 4*x[i]*x[j]*(delta_ij_term)
    
    The last term: when both i and j are flipped, the cross terms x[i]*x[j] appear.
    Specifically, in acf[k], the term x[i]*x[(i+k)%n] becomes (-x[i])*x[(i+k)%n] if (i+k)%n != j,
    but if (i+k)%n == j, it becomes (-x[i])*(-x[j]) = x[i]*x[j], which is the same as before.
    Wait, no: originally it's x[i]*x[j], after flipping both it's (-x[i])*(-x[j]) = x[i]*x[j]. Same!
    So the cross terms where the two flipped bits interact actually DON'T change.
    
    Hmm, let me reconsider. The full expansion:
    acf_new[k] = sum_j x'_j * x'_{(j+k)%n}
    where x'_i = -x_i, x'_j = -x_j, x'_m = x_m otherwise.
    
    = acf[k] 
      - x_i * x_{(i+k)%n}  [term j=i, if (i+k)%n != j]
      - x_{(i-k)%n} * x_i  [term j=(i-k)%n, if (i-k)%n != j]
      - x_j * x_{(j+k)%n}  [term j=j, if (j+k)%n != i]
      - x_{(j-k)%n} * x_j  [term j=(j-k)%n, if (j-k)%n != i]
      + correction for cases where indices coincide with i or j
    
    This is getting messy. Let me just do it the brute force way but limit the pairs.
    """
    x = x.astype(float).copy()
    n = len(x)
    acf = _circ_acf(x)
    best_peak = np.max(np.abs(acf[1:]))
    
    # Precompute single-flip deltas
    deltas = np.zeros((n, n))
    for i in range(n):
        deltas[i] = _flip_delta_acf(x, i)
    
    no_improve = 0
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if best_peak < 1.0:
            break
        
        best_gain = 1e-12
        best_pair = (-1, -1)
        
        # Sample a subset of pairs for speed
        if n <= 200:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        else:
            # Random sample of pairs
            idx = np.random.choice(n, size=n, replace=False)
            pairs = [(int(idx[a]), int(idx[b])) for a in range(n) for b in range(a+1, n)]
        
        for (i, j) in pairs:
            # Combined delta for flipping both i and j
            # We need to account for the fact that delta_i was computed assuming x[i] is the only flip
            # The correct approach: compute new acf directly
            x[i] = -x[i]
            x[j] = -x[j]
            new_acf = _circ_acf(x)
            new_peak = np.max(np.abs(new_acf[1:]))
            x[i] = -x[i]
            x[j] = -x[j]
            
            gain = best_peak - new_peak
            if gain > best_gain:
                best_gain = gain
                best_pair = (i, j)
        
        if best_pair[0] >= 0:
            i, j = best_pair
            x[i] = -x[i]
            x[j] = -x[j]
            acf = _circ_acf(x)
            best_peak = np.max(np.abs(acf[1:]))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break
    
    return x, best_peak

def _continuous_relax(x, deadline, max_iters=5000):
    """
    Relax to continuous values in [-1, 1], minimize PSL via gradient descent.
    Then round to +/- 1.
    """
    x = x.astype(float).copy()
    n = len(x)
    
    # We'll minimize the peak, but use a smooth approximation: 
    # f(x) = log(sum_k exp(alpha * |acf[k]|^2)) / alpha  (soft-max of squared sidelobes)
    # Or simpler: minimize the 2nd moment of sidelobes, then do flip search.
    
    # Actually, let's just do a simple approach: 
    # Use the fact that for a continuous signal, we can compute gradient of PSL.
    # But PSL is non-smooth. Let's minimize the sum of squares of sidelobes instead,
    # which is smooth, then round.
    
    alpha = 20.0  # sharpness of soft-max
    
    def soft_psl(y):
        acf = _circ_acf(y)
        sidelobes = acf[1:]
        # Use log-sum-exp for numerical stability
        sq = sidelobes * sidelobes
        m = np.max(sq)
        if m < 1e-15:
            return m, 0.0
        return m + np.log(np.sum(np.exp((sq - m) / alpha)) / alpha) / alpha, m
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        acf = _circ_acf(x)
        sidelobes = acf[1:]
        sq = sidelobes * sidelobes
        m = np.max(sq)
        
        # Gradient of soft-max objective w.r.t. x
        # d/dx_i of sum_k |acf[k]|^2 = 2 * (d acf/dx_i)^T * acf
        # d acf[k]/d x_i = x[(i+k)%n] + x[(i-k)%n]  (circular)
        # So grad_i = 2 * sum_{k=1}^{n-1} acf[k] * (x[(i+k)%n] + x[(i-k)%n])
        
        # But we want gradient of the soft-max, which weights by exp((sq_k - m)/alpha)
        weights = np.exp((sq - m) / alpha)
        weights = weights / weights.sum()
        
        # grad_i = 2 * sum_{k=1}^{n-1} weights[k-1] * acf[k] * (x[(i+k)%n] + x[(i-k)%n])
        # This is a convolution-like operation
        
        grad = np.zeros(n)
        for i in range(n):
            # grad[i] = 2 * sum_k weights[k] * acf[k] * (x[(i+k)%n] + x[(i-k)%n])
            # = 2 * (conv with x) evaluated at i
            # Let's compute via FFT for speed
            pass
        
        # FFT-based gradient
        # g[k] = weights[k-1] * acf[k] for k=1..n-1, g[0]=0
        g = np.zeros(n)
        g[1:] = weights * acf[1:]
        
        # grad = 2 * ifft(fft(g) * conj(fft(x))) ... need to be careful
        # Actually, the operation sum_k g[k] * x[(i+k)%n] is a circular cross-correlation
        # which is ifft(fft(g) * conj(fft(x)))
        X = np.fft.fft(x)
        G = np.fft.fft(g)
        corr = np.fft.ifft(G * np.conj(X)).real
        grad = 2.0 * corr
        
        # Project gradient descent
        x_new = x - lr * grad
        # Project to [-1, 1]
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Check if it's better
        old_val, _ = soft_psl(x)
        new_val, _ = soft_psl(x_new)
        if new_val < old_val:
            x = x_new
        else:
            lr *= 0.95
    
    # Round to +/- 1
    x_rounded = np.sign(x)
    x_rounded[x_rounded == 0] = 1.0
    return x_rounded

def _construct_barker_like():
    """
    Construct a Barker-like sequence using known structure.
    Barker sequences have low autocorrelation. We'll use a Barker polynomial 
    construction or a Golay-like pair approach.
    """
    # Barker sequence of length 13: +1,-1,-1,+1,-1,+1,+1,+1,-1,+1,-1,-1,+1
    # For length 400, we can tile or use a different construction.
    # Let's use a simple approach: start with a Barker-like base and extend.
    
    # Actually, let's just use a structured initialization:
    # Use a chirp-like sequence: x[i] = sign(cos(2*pi*f*i/n))
    n = N
    # Multiple chirps
    candidates = []
    for f in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        t = np.arange(n)
        x = np.sign(np.cos(2*np.pi*f*t/n))
        x[x == 0] = 1.0
        candidates.append(x)
    
    # Also try a few random ones
    np.random.seed(12345)
    for _ in range(5):
        x = np.random.choice([-1.0, 1.0], size=n)
        candidates.append(x)
    
    # Pick the best
    best_x = None
    best_psl = np.inf
    for x in candidates:
        p = _psl(x)
        if p < best_psl:
            best_psl = p
            best_x = x.copy()
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    np.random.seed(777)
    
    best_x = None
    best_peak = np.inf
    
    # Phase 1: Get a good starting point
    x0 = _construct_barker_like()
    best_peak = _psl(x0)
    best_x = x0.copy()
    
    # Phase 2: Continuous relaxation
    if time.time() < deadline - 60:
        x_cont = _continuous_relax(x0, deadline - 60, max_iters=2000)
        p_cont = _psl(x_cont)
        if p_cont < best_peak:
            best_peak = p_cont
            best_x = x_cont.copy()
    
    # Phase 3: Greedy single-flip optimization
    if time.time() < deadline - 30:
        x_g, p_g = _greedy_flip(best_x, deadline - 30, max_rounds=1000)
        if p_g < best_peak:
            best_peak = p_g
            best_x = x_g.copy()
    
    # Phase 4: Pair-flip optimization
    if time.time() < deadline - 10:
        x_p, p_p = _pair_flip(best_x, deadline - 10, max_rounds=200)
        if p_p < best_peak:
            best_peak = p_p
            best_x = x_p.copy()
    
    # Phase 5: Final greedy polish
    if time.time() < deadline:
        x_f, p_f = _greedy_flip(best_x, deadline, max_rounds=500)
        if p_f < best_peak:
            best_peak = p_f
            best_x = x_f.copy()
    
    if best_x is None:
        best_x = np.random.choice([-1.0, 1.0], size=N)
    
    # Ensure output is a proper sequence of signed values
    best_x = np.sign(best_x)
    best_x[best_x == 0] = 1.0
    
    return best_x
# EVOLVE-BLOCK-END
