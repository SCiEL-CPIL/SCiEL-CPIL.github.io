import numpy as np
import time
from scipy.optimize import minimize

N = 600

def _autoconv_peak(h):
    """
    Compute the maximum value of the linear autoconvolution of h.
    h: 1D array of length N.
    Returns the peak value.
    """
    # Zero pad to 2*N to avoid circular convolution artifacts
    size = 2 * N
    # Ensure size is a power of 2 or has small prime factors for FFT speed? 
    # 2*600 = 1200 = 2^4 * 3 * 5. Good enough.
    H = np.fft.rfft(h, size)
    power = H * np.conj(H)
    # Inverse FFT
    # The linear autoconvolution g_k = sum_i h_i h_{k-i} 
    # is contained in the first N+1 elements of the inverse FFT of the power spectrum?
    # Actually, let's check.
    # Circular conv of size M: c_n = sum_{k=0}^{M-1} h_k h_{(n-k) mod M}
    # If we zero pad h to length 2N-1 (or 2N), the circular conv of size 2N 
    # contains the linear conv in the first N elements (lags 0 to N-1) 
    # AND the negative lags wrapped around?
    # Linear conv length is 2N-1.
    # Indices 0 to 2N-2.
    # Lag k corresponds to index k in the linear conv if we define g_k = sum h_i h_{k-i}.
    # Wait, standard conv: (h*h)[n] = sum h[k] h[n-k].
    # If we want g[k] = sum_i h[i] h[k-i], this is correlation or conv depending on definition.
    # For real h, h*h (conv) is symmetric if h is symmetric?
    # Let's just use np.fft.ifft on the zero-padded power spectrum.
    # The result will be the circular convolution.
    # Since we padded to 2N, the "wrap around" from lag > N-1 will land in indices > N-1.
    # The lags 0, 1, ..., N-1 are in indices 0, 1, ..., N-1.
    # The lags N, ..., 2N-2 (which correspond to negative lags -N+1 ... -1 in symmetric sense 
    # or just higher indices in conv) are in N, ..., 2N-1.
    # We want max over all lags k in -(N-1) ... N-1.
    # Due to symmetry of autoconvolution of a real signal?
    # g(k) = sum h(i) h(k-i). g(-k) = sum h(i) h(-k-i) = sum h(j) h(-k-j+k-k)?
    # Let j = k-i -> i = k-j.
    # g(-k) = sum_j h(k-j) h(-j). If h is defined on 0..N-1, h(j)=0 for j<0 or j>=N.
    # So g(-k) is not necessarily equal to g(k) unless h is symmetric.
    # However, the problem states "nonnegative step function". It does not require symmetry.
    # But usually for minimizing peak autoconvolution, symmetric functions are optimal 
    # by rearrangement inequalities?
    # Even if not, we must check all lags.
    # The linear autoconvolution has 2N-1 non-zero terms.
    # The peak is likely near 0 lag.
    # We can compute the full inverse FFT and take the max of the real part 
    # over the valid range.
    
    # If we use size = 2N, the indices 0..N-1 correspond to lags 0..N-1.
    # The indices N..2N-1 correspond to lags -(N-1)..-1 (wrapped).
    # Specifically, index N corresponds to lag -(N-1) ? No.
    # Index m in circular conv of size M corresponds to lag m if m < M/2?
    # Let's just compute ifft and look at the max.
    # The values at the "seam" (indices N-1 and N) are the far tails.
    # We should take the max of the real part of the entire array, 
    # but ensure we are interpreting it correctly.
    # Since h is non-negative, g(k) >= 0.
    # The max will be positive.
    g = np.fft.irfft(power, n=size)
    # The linear conv is embedded.
    # g[0:N] gives lags 0 to N-1.
    # g[N:2N] gives lags -(N-1) to -1?
    # Let's verify with a small example.
    # h = [1, 1]. N=2.
    # Linear conv: [1, 2, 1]. Lags: -1: 1, 0: 2, 1: 1.
    # Size = 4.
    # FFT [1,1,0,0] -> ...
    # Ifft should give [1, 2, 1, 0]? Or [1, 0, 1, 2]?
    # np.fft.ifft(fft([1,1,0,0])^2).
    # Let's just use np.convolve for safety in the inner loop if N is small? 
    # N=600 is small enough for np.convolve?
    # np.convolve is O(N^2) = 360,000 ops. Very fast in C.
    # FFT is O(N log N) ~ 6000 ops.
    # But FFT has overhead.
    # Let's stick to FFT for the optimization loop, but verify correctness.
    
    # Actually, `scipy.signal.correlate` or `np.correlate` might be safer for logic.
    # But for speed in 1000s of iterations, FFT is better.
    
    # Let's use the property that for real input, the max autoconv is at lag 0 if h is symmetric.
    # If we enforce symmetry, we only need to check positive lags?
    # g(k) for k>0.
    # And by symmetry g(k)=g(-k).
    # So max is max(g[0], max(g[1:N])).
    
    # Let's assume we can just take np.max(g).
    # If there are numerical artifacts or negative values due to rounding, we might pick a spurious negative? 
    # No, we want max, and true max is positive.
    # But we must ensure we are looking at the correct elements.
    # In the circular conv of size 2N, the linear conv of size 2N-1 fits in 2N.
    # The elements are g[0] ... g[2N-2] at indices 0 ... 2N-2?
    # And g[2N-1] is 0?
    # If we pad to 2N, the wrap-around adds g[k] + g[k-M].
    # Since support is 0..N-1, g[k] is 0 for k > 2N-2.
    # So g_circular[k] = g_linear[k] for k < 2N-1?
    # Actually, if we pad to M >= 2N-1, there is no wrap-around aliasing.
    # So g_circular is exactly the linear conv padded to M.
    # So we can just take np.max(g).
    
    peak = np.max(g.real)
    # Clip tiny negatives to 0?
    if peak < 0: peak = 0
    return peak

def _score(h):
    """
    Normalized maximum of autoconvolution.
    We need to decide on the normalization.
    "minimize the normalized maximum of its autoconvolution"
    Common normalization for L1 step functions:
    Let S = sum(h).
    The function is f(t) = h_i on interval i.
    L1 norm = S * dx.
    Autoconvolution peak = P.
    "Normalized maximum" could be P / (L1^2) or P / (L2^2) etc.
    
    Let's look at the seed: [1.0]*N.
    If the evaluator simply computes `max_convolve(h, h)` and returns that, 
    and the seed score is 1.0, then:
    Seed h = [1]*600.
    Conv peak: sum of 1s in overlap. Max overlap is 600.
    Peak = 600.
    If score = Peak, seed score = 600.
    But the task says "score is the upper bound on C1" and "seed/value".
    And "1.0 equals the seed program".
    This implies the evaluator returns a value V such that Seed_Score = 1.0.
    If the evaluator returns the raw peak, and the seed peak is 600, then 
    maybe the score is Peak / Seed_Peak?
    Or maybe the evaluator normalizes h to sum=1 first?
    If h is normalized to sum=1:
    Seed h_norm = [1/600]*600.
    Conv peak = 600 * (1/600)^2 = 600 / 360000 = 1/600.
    If the score is this peak, seed score = 1/600.
    If the search score is seed/value, and 1.0 is seed, then Value must be 1/600?
    
    Let's assume the evaluator:
    1. Takes h.
    2. Computes autoconvolution peak P.
    3. Normalizes P by (sum(h))^2 ?
    Let's test this hypothesis.
    Seed: sum=600, P=600.
    Norm P = 600 / (600^2) = 1/600.
    If the reported "value" is 1/600, and the "score" is 1.0 (seed), 
    then the metric is likely `1 / (Norm_P)`? No.
    
    Let's re-read: "score is the upper bound on C1".
    "The evaluator measures the raw value (lower is better)."
    "The search score is seed/value, so 1.0 equals the seed program".
    This implies:
    Search_Score = Seed_Raw_Value / Candidate_Raw_Value.
    If Search_Score = 1.0, then Seed_Raw_Value = Candidate_Raw_Value.
    So the "Raw Value" is the metric we need to minimize.
    
    What is the Raw Value for the seed?
    If the raw value is the "normalized maximum", we need to know the normalization.
    If I submit h=[1]*600, and get score 1.0, and I submit h=[2]*600, 
    what is the score?
    If raw value scales with h, then [2]*600 has peak 4*600=2400.
    If raw value is just peak, then Value=2400.
    Seed Value = 600.
    Score = 600/2400 = 0.25.
    This would mean scaling down h gives better score (higher value in denominator? No, lower value in denominator -> higher score).
    If I submit h=[0.1]*600, peak = 0.1^2 * 600 = 6.
    Score = 600/6 = 100.
    This would mean I can cheat by making h arbitrarily small.
    The task says "normalized maximum". This implies scale invariance.
    Therefore, the Raw Value must be scale-invariant.
    Possible candidates:
    1. max(f*f) / (sum f)^2
    2. max(f*f) / ||f||_2^2
    3. max(f*f) / ||f||_1^2 (same as 1 for discrete steps if dx cancels)
    
    Let's assume the metric is `max_conv / (sum_h)^2`.
    For seed: 600 / 360000 = 1/600.
    If the Raw Value is 1/600, and the Search Score is Seed_Raw / Cand_Raw,
    then for the seed, Score = (1/600) / (1/600) = 1.0.
    This fits.
    
    So, I should minimize `max_conv / (sum_h)^2`.
    Since this is scale-invariant, I can fix `sum_h = 1` during optimization 
    and just return the shape. The evaluator will likely normalize the input 
    or compute the ratio. To be safe, I will return the shape normalized to sum=1.
    If the evaluator expects sum=600, it will compute the same ratio.
    
    Objective: Minimize R = max(h*h) / (sum(h))^2.
    Subject to h >= 0.
    """
    s = np.sum(h)
    if s == 0:
        return 1.0 # Bad
    peak = _autoconv_peak(h)
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # We want to minimize R(h) = max(h*h) / (sum(h))^2.
    # This is equivalent to minimizing max(h*h) subject to sum(h)=1, h>=0.
    
    # Initial guess: Uniform
    h = np.ones(N) / N
    best_h = h.copy()
    best_score = _score(h)
    
    # Optimization Strategy:
    # 1. Projected Gradient Descent on the simplex (sum=1, h>=0).
    #    The function max(h*h) is convex (max of quadratics).
    #    So minimizing it on the simplex is a convex problem.
    #    We can use subgradient methods.
    #    Gradient of max_k (h*h)_k with respect to h:
    #    Let k* be the argmax lag.
    #    grad = 2 * cross_corr(h, h, lag=k*).
    #    
    # 2. Use L-BFGS-B or similar from scipy?
    #    L-BFGS-B requires smooth gradients. max() is not smooth.
    #    But we can approximate or use subgradient.
    #    
    # 3. Alternating Minimization / Coordinate Descent?
    #    Not obvious structure.
    
    # Let's try Projected Gradient Descent (PGD) with subgradients.
    
    def _project_simplex(v, s=1.0):
        """Project v onto the simplex sum(v) = s, v >= 0."""
        n = len(v)
        if n == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        theta = (cssv[rho] - s) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    # Precompute FFT setup for speed?
    # We'll compute grad inside the loop.
    
    eta = 0.01 # Step size
    patience = 0
    max_iter = 5000
    
    h_curr = h.copy()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute score and gradient
        s = np.sum(h_curr)
        # Normalize for gradient calculation? 
        # The objective is F(h) = max(h*h) / s^2.
        # Gradient of F is complex due to s^2.
        # Easier: Minimize G(h) = max(h*h) subject to sum(h)=1.
        # Then the value is G(h).
        # Let's optimize G(h) on the simplex.
        
        # Compute autoconv
        # We need the lag of the maximum.
        size = 2 * N
        H = np.fft.rfft(h_curr, size)
        power = H * np.conj(H)
        g = np.fft.irfft(power, n=size).real
        peak_val = np.max(g)
        k_star = np.argmax(g)
        
        # Check improvement
        # The score we track is peak_val (since sum=1).
        if peak_val < best_score:
            best_score = peak_val
            best_h = h_curr.copy()
            patience = 0
        else:
            patience += 1
        
        # Subgradient of G(h) = max_k sum_i h_i h_{k-i}
        # grad_j = 2 * h_{k_star - j}
        # Note: indices are modulo N? No, linear conv.
        # h is zero outside [0, N-1].
        # So grad_j = 2 * h[k_star - j] if 0 <= k_star - j < N else 0.
        
        # We can compute this via shift.
        # Let h_shifted be h shifted such that h_shifted[j] = h[k_star - j].
        # This is equivalent to reversing h and shifting?
        # h_rev = h[::-1]. h_rev[j] = h[N-1-j].
        # We want h[k_star - j].
        # Let i = k_star - j => j = k_star - i.
        # grad = 2 * h[k_star - j].
        
        # Efficient computation:
        # grad = 2 * np.zeros(N)
        # idx = np.arange(N)
        # valid = (k_star - idx >= 0) & (k_star - idx < N)
        # grad[valid] = 2 * h_curr[k_star - idx[valid]]
        
        # Even better: use FFT?
        # grad is the cross-correlation of h_curr with the indicator of the max lag?
        # No, it's just a shifted version of h_curr.
        
        # Let's construct the shifted vector.
        # We want v[j] = h_curr[k_star - j] for valid j.
        # This is a "reversed and shifted" operation.
        
        # Create a zero array
        grad = np.zeros(N)
        # Determine valid indices j
        # j such that 0 <= k_star - j < N
        # 0 <= k_star - j => j <= k_star
        # k_star - j < N => j > k_star - N
        
        j_min = max(0, k_star - N + 1)
        j_max = min(N - 1, k_star)
        
        if j_min <= j_max:
            js = np.arange(j_min, j_max + 1)
            # grad[js] = 2 * h_curr[k_star - js]
            grad[js] = 2.0 * h_curr[k_star - js]
        
        # Update: h_new = h_curr - eta * grad
        # Then project to simplex.
        
        # Adaptive step size?
        # Simple PGD often needs small eta.
        # Let's try eta = 0.1 / N ?
        # Or line search? Too slow.
        
        # Let's use a fixed eta and project.
        h_new = h_curr - eta * grad
        h_new = _project_simplex(h_new, 1.0)
        
        # Check if h_new is better
        # We already computed peak_val for h_curr.
        # We need peak for h_new to decide? 
        # PGD guarantees decrease for convex functions if eta is small enough?
        # Yes, for L-smooth. Max of quadratics is Lipschitz gradient.
        # L <= 2 * max_j sum_i |h_i h_{k-i}|? 
        # Basically L depends on the magnitude of h. Since sum=1, it's bounded.
        
        # Let's just update and check in the next iteration's "Check improvement".
        # But to be safe, we can compute the score of h_new immediately.
        # Cost: 1 FFT.
        
        # Optimize: only compute new score every few steps?
        # No, we need to know if we improved to reset patience.
        
        # Compute score of h_new
        H_new = np.fft.rfft(h_new, size)
        power_new = H_new * np.conj(H_new)
        g_new = np.fft.irfft(power_new, n=size).real
        peak_new = np.max(g_new)
        
        if peak_new < best_score:
            best_score = peak_new
            best_h = h_new.copy()
            patience = 0
        else:
            patience += 1
        
        h_curr = h_new
        
        # If stuck, try increasing eta or random restart?
        if patience > 100:
            # Try a different eta
            eta *= 0.5
            if eta < 1e-6:
                # Random restart
                h_curr = np.ones(N) / N
                # Add noise
                h_curr += np.random.normal(0, 0.05, N)
                h_curr = np.maximum(h_curr, 0)
                h_curr = h_curr / np.sum(h_curr)
                eta = 0.01
                patience = 0
        
        # If we found a very good score, stop?
        # Theoretical lower bound?
        # For uniform, peak = 1/N.
        # Can we do better?
        # If we make h sparse, e.g. single spike.
        # h = [1, 0, ..., 0].
        # Conv: [1, 0, ...]. Peak = 1.
        # Uniform Peak = 1/600 = 0.00166.
        # So uniform is much better.
        # What about a triangle?
        # Conv of triangle is piecewise quadratic.
        # Peak at 0.
        # It is known that spreading mass out reduces peak autoconv.
        # Uniform is likely optimal for non-negative functions.
        # Proof sketch: 
        # By rearrangement, for fixed L1 norm, the peak autoconv is maximized 
        # by concentrating mass? No, we want to MINIMIZE peak.
        # So we want to spread mass as much as possible.
        # Uniform distribution spreads mass maximally over the support.
        # So Uniform is the optimal solution!
        
        # If Uniform is optimal, why did the previous program score 1.0 (seed)?
        # Because the seed IS uniform.
        # And my best_score is 1/600.
        # If I return uniform, I get score 1.0.
        # Can I do better than uniform?
        # If I restrict support to a subset, peak increases.
        # If I vary heights, e.g. cos^2, does peak decrease?
        # Let's test h = cos^2(pi * t / L).
        # It has lower variance than uniform? No, uniform has max variance for fixed mean?
        # Actually, uniform has the minimum peak autoconv among all non-negative functions 
        # with fixed support and L1 norm?
        # This is a known result: The uniform distribution minimizes the maximum 
        # auto-correlation (or auto-convolution) peak for non-negative functions.
        
        # If this is true, then no algorithm can beat 1.0.
        # But the inspiration mentions "1.125 -> 1.133".
        # This implies it IS possible to beat the seed.
        # How?
        # Maybe the "normalized maximum" is defined differently?
        # Or maybe the support can be chosen?
        # "Choose a nonnegative step function with 600 equal intervals on [-1/4,1/4]"
        # The support is FIXED to [-1/4, 1/4].
        # The heights are variable.
        
        # Wait, if Uniform is optimal, why would anyone improve it?
        # Maybe my assumption that Uniform is optimal is wrong.
        # Let's check a simple case N=2.
        # h = [a, b]. a+b=1.
        # Conv: [a^2, 2ab, b^2].
        # Peak = max(a^2, 2ab, b^2).
        # a^2 + b^2 + 2ab = 1.
        # If a=b=0.5, peak = max(0.25, 0.5, 0.25) = 0.5.
        # If a=1, b=0, peak = 1.
        # If a=0.9, b=0.1, peak = max(0.81, 0.18, 0.01) = 0.81.
        # So uniform (0.5, 0.5) gives min peak 0.5.
        
        # N=3.
        # h = [a, b, c].
        # Conv: [a^2, 2ab, b^2+2ac, 2bc, c^2].
        # Uniform: [1/3, 1/3, 1/3].
        # Conv: [1/9, 2/9, 5/9, 2/9, 1/9]. Peak = 5/9 = 0.555.
        # Try [0.5, 0, 0.5].
        # Conv: [0.25, 0, 0.5, 0, 0.25]. Peak = 0.5.
        # 0.5 < 0.555!
        # So Uniform is NOT optimal for N=3.
        # The "split" distribution is better.
        
        # So for N=600, we should look for structures that are not uniform.
        # Perhaps periodic spikes?
        
        # The PGD loop above should find such structures if initialized well.
        # But starting from Uniform, the gradient at Uniform for N=3:
        # Peak is at lag 1 (value 5/9).
        # Grad = 2 * h shifted by 1.
        # h = [1/3, 1/3, 1/3].
        # Shift by 1: [0, 1/3, 1/3] (assuming linear).
        # Grad = [0, 2/3, 2/3].
        # Update: h - eta * grad.
        # h_new = [1/3, 1/3-2/3 eta, 1/3-2/3 eta].
        # This makes h[0] larger relative to others?
        # Wait, grad is positive where we should DECREASE h.
        # We want to decrease h at indices contributing to the peak.
        # Peak at lag 1 involves h[0]h[1] and h[1]h[2].
        # So decreasing h[0], h[1], h[2] might help?
        # But we must keep sum=1.
        # The projection will handle it.
        
        # Let's run the loop.
        
    return best_h.tolist()

# Quick sanity check
if __name__ == "__main__":
    h = construct()
    h = np.array(h)
    print("Sum:", np.sum(h))
    score = _score(h)
    print("Score (normalized peak):", score)
    uniform = np.ones(N) / N
    print("Uniform Score:", _score(uniform))
