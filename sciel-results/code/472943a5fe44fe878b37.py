import numpy as np
import scipy.optimize as so
import time

N = 400

def _autoconv_max(h):
    """Compute the max of the full autoconvolution of h."""
    # Use FFT for speed
    n = len(h)
    size = 2 * n - 1
    # Pad to power of 2 for FFT efficiency
    fsize = 1 << (size - 1).bit_length()
    H = np.fft.rfft(h, n=fsize)
    conv = np.fft.irfft(H * H, n=fsize)[:size]
    return np.max(conv)

def _exact_value(h):
    """Compute the exact task value."""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return 1e10
    M = _autoconv_max(h)
    return abs(2.0 * N * M) / (S * S)

def _objective(params, beta=50.0):
    """
    Objective for 2-parameter optimization:
    h = a + b * cos(2*pi*k*x + phi) where x = -1/4, ..., 1/4 (N points)
    Actually, let's parameterize more simply:
    h[i] = a + b * cos(2*pi*f*i/N + phi)
    where f is a fixed frequency (integer for periodicity on the grid).
    
    But the directive says "single high-frequency cosine mode" with amplitude and phase.
    So h = a + b * cos(theta) where theta varies.
    Let's fix the frequency. A high frequency means many oscillations in [-1/4, 1/4].
    The interval length is 0.5. N=400 intervals.
    Let's pick a frequency such that the cosine completes an integer number of cycles in the domain.
    Or just use index-based: h[i] = a + b * cos(2*pi*m*i/N + phi)
    This is periodic with period N/m.
    
    To minimize the autoconvolution peak, we want the signal to look as "random" or "flat" as possible
    in the autocorrelation sense. A single cosine has a very strong peak at lag 0.
    However, if we choose a high frequency, the peak at lag 0 is N*(a^2 + b^2/2) roughly?
    Wait, sum(h^2) is the peak at lag 0.
    sum(h) = N*a (since sum of cos is 0 for integer cycles).
    So value ~ 2*N*(N*(a^2+b^2/2)) / (N*a)^2 = 2*(a^2+b^2/2)/a^2 = 2 + b^2/a^2.
    This is >= 2. The seed is 2.0.
    So a single cosine with a non-zero mean will be >= 2.
    
    What if a=0? Then sum(h)=0, which is invalid.
    So we need a non-zero mean.
    
    Maybe the strategy is different.
    "Represent the step heights h as a linear combination of a constant baseline and a single high-frequency cosine mode"
    If we use h = a + b*cos(...), and we want to minimize the value, we found it's >= 2.
    Is it possible to get < 2?
    The seed (all ones) gives 2.0.
    If we perturb the ones, do we get < 2?
    Let h = 1 + eps * noise.
    sum(h) = N + eps * sum(noise).
    If sum(noise) = 0, sum(h) = N.
    conv(0) = sum(h^2) = sum((1+eps*n)^2) = N + eps^2 * sum(n^2).
    value = 2*N*(N + eps^2*sum(n^2)) / N^2 = 2 + 2*eps^2*sum(n^2)/N.
    This is > 2.
    
    So adding small noise to ones increases the value.
    To get < 2, we need max(conv) to be smaller relative to sum(h)^2.
    This implies we need to reduce the peak of the autocorrelation.
    The peak is at lag 0, which is sum(h^2).
    By Cauchy-Schwarz, (sum h)^2 <= N * sum(h^2).
    So sum(h^2) >= (sum h)^2 / N.
    Therefore max(conv) >= sum(h^2) >= (sum h)^2 / N.
    So value = 2*N*max(conv)/sum(h)^2 >= 2*N*(sum h)^2/N / sum(h)^2 = 2.
    
    Wait. The theoretical lower bound is 2.
    The seed achieves 2.
    So the minimum possible value is 2?
    
    Let's re-read the task.
    "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    "The seed np.ones(400) gives 2.0."
    
    If the lower bound is 2, then we cannot do better than 2.
    But the inspirations show scores > 1.0.
    Score = seed_value / current_value = 2.0 / value.
    If value > 2, score < 1.0.
    If value < 2, score > 1.0.
    
    Did I make a mistake in the bound?
    max(conv) is the max over ALL lags, not just lag 0.
    Lag 0 is sum(h^2).
    Is it possible that max(conv) < sum(h^2)?
    No, lag 0 is one of the entries. So max(conv) >= sum(h^2).
    So value >= 2.
    
    So how did inspiration 1 get 1.3140?
    Score 1.3140 means value = 2.0 / 1.3140 = 1.52.
    This contradicts my bound.
    
    Let's check the bound again.
    (sum h)^2 = (sum h_i)^2.
    sum(h^2) = sum h_i^2.
    By Cauchy-Schwarz: (sum h_i)^2 <= N * sum h_i^2.
    So sum h_i^2 >= (sum h_i)^2 / N.
    
    conv(h,h)[0] = sum h_i^2.
    So max(conv) >= sum h_i^2 >= (sum h)^2 / N.
    
    value = 2 * N * max(conv) / (sum h)^2 >= 2 * N * (sum h)^2 / N / (sum h)^2 = 2.
    
    Unless... np.convolve is full, so indices are 0 to 2N-2.
    Lag 0 corresponds to index N-1 in full convolution?
    No, np.convolve(h, h) with mode 'full' gives length 2N-1.
    Index 0 is h[0]*h[0].
    Index N-1 is sum_{i=0}^{N-1} h[i]*h[i] = sum(h^2).
    So yes, lag 0 is included.
    
    Why is the score > 1?
    Maybe my understanding of "value" is wrong?
    "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    
    Let's test with h = [1, -1, 1, -1]. N=4.
    sum(h) = 0. Invalid.
    
    Let's test with h = [1, 1, -1, -1]. N=4.
    sum(h) = 0. Invalid.
    
    Let's test with h = [1, 0, 0, 0]. N=4.
    sum(h) = 1.
    conv = [1, 0, 0, 0, 0, 0, 0].
    max = 1.
    value = 2 * 4 * 1 / 1 = 8.
    Score = 2/8 = 0.25.
    
    Let's test with h = [1, 1, 1, 1]. N=4.
    sum(h) = 4.
    conv = [1, 2, 3, 4, 3, 2, 1].
    max = 4.
    value = 2 * 4 * 4 / 16 = 2.
    Score = 1.
    
    Let's test with h = [1, 1, 1, -1]. N=4.
    sum(h) = 2.
    conv:
    0: 1
    1: 1*1 + 1*(-1) = 0? No.
    conv[0] = 1*1 = 1
    conv[1] = 1*1 + 1*(-1) = 0
    conv[2] = 1*1 + 1*1 + 1*(-1) = 1
    conv[3] = 1*1 + 1*1 + 1*1 + 1*(-1) = 2  (This is sum h_i^2? 1+1+1+1=4. No.)
    
    Wait. np.convolve(h, h)
    h = [1, 1, 1, -1]
    rev_h = [-1, 1, 1, 1]
    
    Full convolution:
    k=0: 1*(-1) = -1
    k=1: 1*1 + 1*(-1) = 0
    k=2: 1*1 + 1*1 + 1*(-1) = 1
    k=3: 1*1 + 1*1 + 1*1 + 1*(-1) = 2
    k=4: 1*1 + 1*1 + 1*1 = 3
    k=5: 1*1 + 1*1 = 2
    k=6: 1*1 = 1
    
    Max is 3.
    sum(h) = 2. sum^2 = 4.
    value = 2 * 4 * 3 / 4 = 6.
    Score = 2/6 = 0.33.
    
    Let's try to find a case where value < 2.
    We need max(conv) < sum(h)^2 / (2N).
    But max(conv) >= sum(h^2) >= sum(h)^2 / N.
    So sum(h)^2 / N <= max(conv) < sum(h)^2 / (2N).
    This implies 1/N < 1/(2N), which is false.
    
    So mathematically, value >= 2.
    
    Why does the task say "lower is better" and "seed gives 2.0"?
    And inspirations have scores > 1?
    
    Maybe the evaluator uses a different definition?
    "The evaluator computes exactly this value... value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    
    Is it possible that np.sum(h) is complex? No, real heights.
    Is it possible that np.max is taken over something else?
    "maximum is taken over its signed entries, with no absolute value inside np.max"
    
    Maybe I should check if the inspirations are actually valid or if they are hallucinated scores?
    "inspiration 1 (score 1.3140)"
    
    If the math is correct, the best score is 1.0.
    However, I must follow the directive: "Make a STRUCTURAL change... a different representation".
    
    If the lower bound is indeed 2, then any perturbation of the seed will yield a score <= 1.
    The current program scores 1.0.
    
    Perhaps the "value" in the score calculation is different?
    "The search score is seed/value"
    Seed value is 2.0.
    If my value is 2.0, score is 1.0.
    
    Let's assume the inspirations are real and I am missing something.
    What if h contains negative values?
    My proof used h_i^2, so signs don't matter for the sum of squares.
    (sum h)^2 can be small if signs cancel.
    But max(conv) is at least sum(h^2).
    
    Let h = [1, -1, 1, -1, ...] (alternating).
    sum(h) = 0.
    
    Let h = [1, -1, 1, -1, ..., 1] (odd N).
    sum(h) = 1.
    sum(h^2) = N.
    conv(0) = N.
    value = 2*N*N / 1 = 2N^2. Huge.
    
    What if we choose h such that sum(h) is large, but sum(h^2) is small?
    Impossible by C-S.
    
    What if max(conv) is NOT at lag 0?
    It can be, but it's at least lag 0.
    
    Okay, I will proceed with the 2-parameter cosine model as requested, even if I suspect it won't beat 2.0. The directive is to try a new direction.
    
    Parameters: a (baseline), b (amplitude), phi (phase).
    h[i] = a + b * cos(2*pi*f*i/N + phi)
    Fix f to a high integer, e.g., f=10.
    
    We need to minimize value(a, b, phi).
    Since value is homogeneous of degree 0 (scaling h doesn't change value), we can fix one parameter, e.g., a=1.
    Then minimize over b, phi.
    
    Actually, if a=1, b=0, we get the seed.
    We want to see if b != 0 helps.
    
    Let's implement this.
    """
    a, b, phi = params
    i = np.arange(N)
    # High frequency, e.g. 10 cycles
    f = 10
    h = a + b * np.cos(2 * np.pi * f * i / N + phi)
    
    S = np.sum(h)
    if abs(S) < 1e-10:
        return 1e10
    
    M = _autoconv_max(h)
    val = abs(2.0 * N * M) / (S * S)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Best found so far
    best_h = np.ones(N)
    best_val = _exact_value(best_h) # Should be 2.0
    
    # 2-parameter (actually 3, but scale invariant) optimization
    # h = a + b * cos(...)
    # We optimize a, b, phi.
    
    def obj(params):
        return _objective(params)
    
    # Initial guesses
    # 1. Seed: a=1, b=0, phi=0
    x0_1 = [1.0, 0.0, 0.0]
    # 2. Small perturbation
    x0_2 = [1.0, 0.1, 0.0]
    x0_3 = [1.0, 0.1, 1.0]
    x0_4 = [1.0, 0.5, 0.0]
    x0_5 = [0.5, 0.5, 0.0]
    
    inits = [x0_1, x0_2, x0_3, x0_4, x0_5]
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            res = so.minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 100})
            if res.success and res.fun < best_val:
                best_val = res.fun
                a, b, phi = res.x
                i = np.arange(N)
                f = 10
                best_h = a + b * np.cos(2 * np.pi * f * i / N + phi)
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
    
    # If we haven't improved, return the seed (or best found)
    # The seed is a valid local minimum (actually global minimum if bound is 2).
    # But let's return the best_h found.
    
    return best_h
