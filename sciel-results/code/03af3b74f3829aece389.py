"""
Strategy: Parametric Chirp/Sinusoid Representation with Differential Evolution

Instead of optimizing 400 individual heights, we restrict the solution space to a 
parametric family of signals that inherently have low autocorrelation sidelobes:
    h[i] = A * cos(2*pi * (f0 * i + 0.5 * mu * i^2) / N + phi) + B

Here:
- A is the amplitude
- f0 is the initial frequency (cycles per full signal length)
- mu is the chirp rate (frequency slope)
- phi is the initial phase
- B is the DC offset (bias)

The autocorrelation of a linear chirp (Bessel function-like decay) or a pure sinusoid 
(peak at lag 0, oscillating sidelobes) has predictable properties. By allowing a 
chirp, we can "smear" the energy in the frequency domain, which often reduces the 
peak sidelobe level in the time domain compared to a pure tone or random noise.

We use Differential Evolution (DE) to perform a global search over these 5 parameters, 
followed by a local Nelder-Mead refinement. This avoids the high-dimensional local 
minima traps of the full vector optimization and leverages the structural properties 
of chirps to minimize the maximum autocorrelation peak.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin (150s limit)

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    # np.max takes the algebraic maximum. 
    # The task says "maximum is taken over its signed entries, with no absolute value inside np.max"
    # So we take max(conv), then abs of the whole expression.
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _parametric_h(params):
    """
    Generate the 400 heights from the parametric chirp model.
    params: [A, f0, mu, phi, B]
    """
    A, f0, mu, phi, B = params
    i = np.arange(N)
    # Phase accumulation: 2*pi * (f0 * i/N + 0.5 * mu * (i/N)^2) + phi
    # Note: mu is scaled to be in cycles^2 per unit length squared effectively.
    # Let's define phase[i] = 2*pi * (f0 * i / N + 0.5 * mu * (i / N)**2) + phi
    # This makes f0 and mu dimensionless in a normalized sense.
    
    # To allow for "fast" chirps, mu can be large.
    phase = 2.0 * np.pi * (f0 * i / N + 0.5 * mu * (i / N)**2) + phi
    h = A * np.cos(phase) + B
    return h

def _objective(params):
    """
    Objective function for optimization.
    Returns the value to be minimized.
    """
    h = _parametric_h(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    val = _evaluate(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Define bounds for the 5 parameters: [A, f0, mu, phi, B]
    # A: Amplitude. Scale doesn't matter for the ratio, but helps with numerical stability. 
    #    Let's fix A=1 effectively by scaling, but DE works better with bounded vars.
    #    Actually, the value is scale-invariant: h -> c*h doesn't change value.
    #    So we can fix A=1 and optimize [f0, mu, phi, B] (4 params).
    #    However, allowing A to vary might help if B is large? No, ratio is invariant.
    #    Let's optimize [f0, mu, phi, B] with A=1.
    
    # Bounds:
    # f0: Frequency. Range [-10, 10] cycles per length.
    # mu: Chirp rate. Range [-50, 50]. Larger values create faster frequency sweeps.
    # phi: Phase. Range [0, 2*pi].
    # B: Bias. Range [-5, 5]. Large bias makes sum(h) large, reducing the value.
    #    Wait, if B is very large, h[i] ~ B, conv ~ B^2 * N, sum ~ B*N.
    #    Value ~ 2*N*(B^2*N) / (B*N)^2 = 2*N*B^2*N / B^2*N^2 = 2.
    #    So large B converges to 2.0 (the seed value).
    #    To get below 2.0, we need the oscillating part to reduce the max conv 
    #    relative to the mean square.
    #    Specifically, we want max(conv) / (sum(h)^2) to be small.
    #    If h = B + x, sum = N*B + sum(x).
    #    If sum(x) = 0 (zero mean x), sum = N*B.
    #    conv = conv(B+B) + 2*conv(B,x) + conv(x,x) = B^2*tri + ... + conv(x,x)
    #    Max conv is dominated by B^2*N (at lag 0) if B is large.
    #    Value ~ 2*N*(B^2*N) / (N*B)^2 = 2.
    #    So B should be small to allow the structure of x to lower the max below the 
    #    "background" level? 
    #    Actually, if B=0, sum(h)=0 is forbidden.
    #    If B is small, say 0.1, and x has unit variance, sum(h) ~ 0.1*N.
    #    Max conv of x is roughly N * Var(x) * peak_factor.
    #    Value ~ 2*N * (N * Var * peak) / (0.1*N)^2 = 2 * Var * peak / 0.01.
    #    This can be large.
    #    
    #    Let's look at the seed: h=1. B=1, x=0. Value=2.
    #    If we pick h = cos(...) + 1.
    #    sum = N.
    #    conv(1+cos, 1+cos) = conv(1,1) + conv(cos,cos) + 2conv(1,cos).
    #    conv(1,1) has max N (at lag 0).
    #    conv(cos,cos) has max ~ N/2 (at lag 0).
    #    conv(1,cos) is a shifted sinc-like, max ~ 1.
    #    Total max ~ N + N/2 = 1.5 N.
    #    Value = 2*N * 1.5N / N^2 = 3.0. Worse!
    #    
    #    So adding oscillations to a constant increases the peak if the constant is positive.
    #    We need NEGATIVE sidelobes or a structure that reduces the peak at lag 0?
    #    No, lag 0 is always sum(h^2).
    #    Value = 2*N * max_k conv[k] / (sum h)^2.
    #    conv[0] = sum(h^2).
    #    By Cauchy-Schwarz, (sum h)^2 <= N * sum(h^2).
    #    So sum(h^2) >= (sum h)^2 / N.
    #    Therefore max(conv) >= conv[0] >= (sum h)^2 / N.
    #    Value >= 2*N * (sum h)^2/N / (sum h)^2 = 2.
    #    
    #    WAIT. The theoretical lower bound is 2.0?
    #    If so, the seed is optimal?
    #    Let's re-read the task.
    #    "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    #    
    #    If h is constant, conv is a triangle. Max is at lag 0.
    #    conv[0] = N * h^2. Sum = N * h.
    #    Value = 2 * N * (N h^2) / (N h)^2 = 2 N^2 h^2 / N^2 h^2 = 2.
    #    
    #    Is it possible to get < 2?
    #    We need max(conv) / (sum h)^2 < 1/N.
    #    But max(conv) >= conv[0] = sum(h^2).
    #    So we need sum(h^2) / (sum h)^2 < 1/N.
    #    N * sum(h^2) < (sum h)^2.
    #    This contradicts Cauchy-Schwarz: N * sum(h^2) >= (sum h)^2.
    #    
    #    So the value is ALWAYS >= 2.0?
    #    
    #    Let's check the inspiration scores. They are 1.14, 1.21.
    #    The seed score is 1.0 (value 2.0).
    #    Score = Seed_Value / Current_Value = 2.0 / Value.
    #    If Score > 1.0, then Value < 2.0.
    #    
    #    My derivation says Value >= 2.0. Where is the error?
    #    
    #    Error: `np.max(np.convolve(h, h))` takes the ALGEBRAIC maximum.
    #    It does NOT take the maximum absolute value.
    #    
    #    If the autocorrelation sidelobes are negative, the max might be at a non-zero lag 
    #    if the peak at lag 0 is NOT the global maximum?
    #    No, conv[0] = sum(h^2) is always positive and typically the largest magnitude.
    #    However, if h has negative values, sum(h) can be small, making the denominator small.
    #    But conv[0] is sum of squares, which is large.
    #    
    #    Let's re-evaluate:
    #    Value = 2 * N * max(conv) / (sum h)^2.
    #    
    #    If max(conv) is negative, abs() makes it positive.
    #    But max(conv) is the largest signed value.
    #    If all conv values are negative, max(conv) is the one closest to 0.
    #    But conv[0] = sum(h^2) is always >= 0.
    #    So max(conv) >= conv[0] >= 0.
    #    So max(conv) is always non-negative.
    #    
    #    So max(conv) >= sum(h^2).
    #    And (sum h)^2 <= N * sum(h^2).
    #    So sum(h^2) / (sum h)^2 >= 1/N.
    #    So Value >= 2.
    #    
    #    Why do inspirations score > 1.0?
    #    Maybe I am misinterpreting `np.max`.
    #    `np.max` on an array of floats returns the largest element.
    #    
    #    Let's test a case.
    #    h = [1, -1, 1, -1] (N=4).
    #    sum = 0. Undefined.
    #    
    #    h = [1, -1, 1, 1] (N=4).
    #    sum = 2. sum^2 = 4.
    #    conv:
    #    lag 0: 1+1+1+1 = 4
    #    lag 1: 1*(-1) + (-1)*1 + 1*1 = -1 -1 + 1 = -1
    #    lag 2: 1*1 + (-1)*1 = 0
    #    lag 3: 1*1 = 1
    #    Full conv: [1, 0, -1, 4, -1, 0, 1]
    #    Max is 4.
    #    Value = 2 * 4 * 4 / 4 = 8.
    #    
    #    h = [1, 1, -1, -1] (N=4).
    #    sum = 0.
    #    
    #    h = [1, 2, -1, 0] (N=4).
    #    sum = 2. sum^2 = 4.
    #    conv[0] = 1+4+1+0 = 6.
    #    conv[1] = 1*2 + 2*(-1) + (-1)*0 = 2-2=0.
    #    conv[2] = 1*(-1) + 2*0 = -1.
    #    conv[3] = 1*0 = 0.
    #    Max is 6.
    #    Value = 2 * 4 * 6 / 4 = 12.
    #    
    #    It seems hard to get below 2.
    #    
    #    Let's check the seed again.
    #    h = ones(400).
    #    sum = 400. sum^2 = 160000.
    #    conv[0] = 400.
    #    Max conv = 400.
    #    Value = 2 * 400 * 400 / 160000 = 320000 / 160000 = 2.
    #    
    #    Is it possible that `np.max` is applied to the ABSOLUTE values?
    #    "maximum is taken over its signed entries, with no absolute value inside np.max"
    #    This explicitly says NO absolute value inside.
    #    
    #    So why is the score > 1?
    #    Score = 2.0 / Value.
    #    If Score = 1.14, Value = 2.0 / 1.14 = 1.75.
    #    This implies Value < 2.
    #    This contradicts my Cauchy-Schwarz proof.
    #    
    #    Let's re-read the Cauchy-Schwarz step.
    #    (sum h)^2 <= N * sum(h^2).
    #    This is always true.
    #    So sum(h^2) / (sum h)^2 >= 1/N.
    #    Max(conv) >= conv[0] = sum(h^2).
    #    So Max(conv) / (sum h)^2 >= 1/N.
    #    Value = 2*N * (Max(conv) / (sum h)^2) >= 2*N * (1/N) = 2.
    #    
    #    There must be a misunderstanding of the task or the code.
    #    
    #    Let's look at the code provided in the prompt:
    #    `value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)`
    #    
    #    Maybe `np.sum(h)` is not the square of the sum?
    #    `np.sum(h) ** 2` is the square of the sum.
    #    
    #    Is it possible that `np.max` is negative?
    #    If all entries are negative, max is the largest (closest to 0).
    #    But conv[0] = sum(h^2) is positive.
    #    So max is always positive.
    #    
    #    Is it possible that the evaluator uses a DIFFERENT definition?
    #    "The evaluator computes exactly this value"
    #    
    #    Let's check if I missed a detail.
    #    "Choose real, possibly negative step heights"
    #    
    #    Wait. What if `np.sum(h)` is close to 0?
    #    If sum(h) is very small, the value explodes.
    #    
    #    Let's try to find a counter-example to Value >= 2.
    #    We need Max(conv) < (sum h)^2 / N.
    #    But Max(conv) >= sum(h^2).
    #    So we need sum(h^2) < (sum h)^2 / N.
    #    N * sum(h^2) < (sum h)^2.
    #    This is strictly impossible for real numbers unless h=0 (which is invalid).
    #    
    #    Conclusion: The value MUST be >= 2.
    #    
    #    So how did the inspirations get score > 1?
    #    Score = Seed / Value.
    #    If Seed = 2.0, and Score = 1.14, Value = 1.75.
    #    
    #    Is the Seed value actually 2.0?
    #    "The seed np.ones(400) gives 2.0."
    #    
    #    Maybe the "Score" definition in the prompt is different?
    #    "The search score is seed/value"
    #    "1.0 equals the seed program"
    #    
    #    This implies that if I find a value of 1.75, my score is 2.0/1.75 = 1.14.
    #    
    #    This implies that Value < 2.0 is possible.
    #    My proof says it's not.
    #    
    #    Let's re-read the convolution definition.
    #    `np.convolve(h, h)` is FULL convolution.
    #    Length 2N-1.
    #    
    #    Is it possible that `np.max` is taken over a SUBSET?
    #    "maximum is taken over its signed entries"
    #    
    #    Let's check if `np.sum(h)` is complex? No, real heights.
    #    
    #    Is it possible that I am calculating `conv[0]` wrong?
    #    `np.convolve([1,2], [1,2])` -> [1, 4, 4]
    #    Sum of squares: 1^2+2^2 = 5.
    #    Conv[0] = 1*1 = 1? No.
    #    `np.convolve` does:
    #    c[0] = h[0]h[0]
    #    c[1] = h[0]h[1] + h[1]h[0]
    #    c[2] = h[1]h[1]
    #    
    #    Wait. `np.convolve(h, h)` computes the convolution.
    #    For `h=[1,2]`, `conv=[1, 4, 4]`.
    #    `h[0]*h[0] + h[1]*h[1]` is NOT a single term in the convolution.
    #    
    #    My assumption `conv[0] = sum(h^2)` is WRONG for `np.convolve`.
    #    
    #    Let's check `np.correlate` vs `np.convolve`.
    #    `np.correlate(h, h, 'full')` would have `sum(h^2)` at the center.
    #    `np.convolve(h, h, 'full')` is the convolution.
    #    For a vector h, `conv(h, h)[k] = sum_i h[i] h[k-i]`.
    #    
    #    If h is real, `conv(h, h)` is symmetric?
    #    No, `conv(h, h)` is symmetric only if h is symmetric?
    #    Actually, `conv(h, h)[k] = sum h[i]h[k-i]`.
    #    `conv(h, h)[N-1-k] = sum h[i]h[N-1-k-i]`.
    #    Let j = N-1-k-i. Then i = N-1-k-j.
    #    Sum h[N-1-k-j] h[j]. Same sum.
    #    So `conv(h, h)` is symmetric around index N-1.
    #    
    #    The peak of the autocorrelation (in the correlation sense) is at lag 0.
    #    In convolution, lag 0 corresponds to index N-1.
    #    
    #    `conv(h, h)[N-1] = sum_i h[i] h[i] = sum(h^2)`.
    #    
    #    So the MAX of `np.convolve(h, h)` is at least `sum(h^2)`.
    #    
    #    So my proof holds: `max(conv) >= sum(h^2)`.
    #    
    #    So Value >= 2.
    #    
    #    Why are the scores > 1?
    #    
    #    Let's look at the "Inspiration 1" code description.
    #    "score 1.2159"
    #    
    #    Maybe the "Seed" value is not 2.0?
    #    "The seed np.ones(400) gives 2.0."
    #    
    #    Maybe the evaluator uses `np.correlate`?
    #    "np.convolve(h, h) is the full autoconvolution"
    #    
    #    Maybe I should just trust the inspirations and try to minimize the value.
    #    If the value can go below 2, there must be a flaw in my logic.
    #    
    #    Flaw check:
    #    Is `max(conv)` always `conv[N-1]`?
    #    If h has negative values, `conv[N-1]` is `sum(h^2)` which is positive.
    #    Other terms `conv[k]` can be negative or positive.
    #    Can `conv[k]` be LARGER than `conv[N-1]`?
    #    
    #    Let's test `h = [1, -1]`.
    #    `conv = [1, 0, 1]`.
    #    `sum(h^2) = 2`.
    #    `conv[1] = 0`.
    #    `max(conv) = 1`.
    #    `sum(h) = 0`.
    #    
    #    `h = [1, -2]`.
    #    `conv = [1, -4, 4]`.
    #    `sum(h^2) = 5`.
    #    `conv[1] = -4`.
    #    `max(conv) = 4`.
    #    `sum(h) = -1`. `sum^2 = 1`.
    #    `N=2`.
    #    `Value = 2 * 2 * 4 / 1 = 16`.
    #    
    #    `h = [1, 1, -1]`. N=3.
    #    `conv`:
    #    1*1 = 1
    #    1*1 + 1*(-1) = 0
    #    1*(-1) + 1*1 + (-1)*1 = -1
    #    1*1 + (-1)*1 = 0
    #    (-1)*(-1) = 1
    #    `conv = [1, 0, -1, 0, 1]`.
    #    `sum(h^2) = 3`.
    #    `max(conv) = 1`.
    #    `sum(h) = 1`. `sum^2 = 1`.
    #    `Value = 2 * 3 * 1 / 1 = 6`.
    #    
    #    Wait. `max(conv)` was 1. `sum(h^2)` was 3.
    #    `conv[N-1]` is `conv[2]` which is -1?
    #    Let's re-calculate `conv` for `[1, 1, -1]`.
    #    Indices 0, 1, 2.
    #    Conv index 0: h[0]h[0] = 1
    #    Conv index 1: h[0]h[1] + h[1]h[0] = 1+1=2?
    #    No, `np.convolve` definition:
    #    `c[k] = sum_i h[i] g[k-i]`
    #    For `h=[1,1,-1]`, `g=h`.
    #    k=0: i=0. h[0]h[0] = 1.
    #    k=1: i=0,1. h[0]h[1] + h[1]h[0] = 1+1=2.
    #    k=2: i=0,1,2. h[0]h[2] + h[1]h[1] + h[2]h[0] = -1 + 1 - 1 = -1.
    #    k=3: i=1,2. h[1]h[2] + h[2]h[1] = -1 -1 = -2.
    #    k=4: i=2. h[2]h[2] = 1.
    #    
    #    `conv = [1, 2, -1, -2, 1]`.
    #    `max(conv) = 2`.
    #    `sum(h^2) = 3`.
    #    `conv[N-1] = conv[2] = -1`.
    #    
    #    AH. `conv[N-1]` is NOT `sum(h^2)`.
    #    `conv[N-1]` is the correlation at lag 0?
    #    Correlation at lag 0 is `sum h[i]^2`.
    #    Convolution at index `N-1` is `sum h[i] h[N-1-i]`.
    #    This is the correlation at lag `N-1` (shifted).
    #    
    #    The peak of the AUTO-correlation (lag 0) is at index `N-1` in `np.correlate`.
    #    In `np.convolve`, the auto-correlation is reversed and shifted.
    #    `conv(h, h)[k]` corresponds to correlation at lag `N-1-k`.
    #    
    #    So `conv[N-1]` is correlation at lag 0.
    #    `conv[N-1] = sum h[i] h[i] = sum(h^2)`.
    #    
    #    Let's re-calculate `conv[2]` for `h=[1,1,-1]`.
    #    `i` ranges such that `0 <= i < 3` and `0 <= 2-i < 3`.
    #    `i=0`: h[0]h[2] = 1*(-1) = -1.
    #    `i=1`: h[1]h[1] = 1*1 = 1.
    #    `i=2`: h[2]h[0] = -1*1 = -1.
    #    Sum = -1.
    #    
    #    So `conv[N-1]` is NOT `sum(h^2)`.
    #    `conv[N-1]` is `sum h[i] h[N-1-i]`.
    #    
    #    The term `sum(h^2)` appears in `conv` at indices where `i = k-i`?
    #    No.
    #    The sum of squares is `sum h[i]^2`.
    #    In `conv`, `h[i]` multiplies `h[k-i]`.
    #    If `k-i = i`, i.e., `k=2i`, then we have `h[i]^2`.
    #    So `sum(h^2)` is distributed across the convolution if N is even?
    #    No, `conv` is linear.
    #    
    #    Actually, `np.convolve(h, h)` is NOT the autocorrelation.
    #    `np.correlate(h, h)` is the autocorrelation.
    #    `np.convolve(h, h)` is the convolution.
    #    
    #    For real h, `conv(h, h)` is symmetric?
    #    `conv(h, h)[k] = sum h[i] h[k-i]`.
    #    `conv(h, h)[N-1-k] = sum h[i] h[N-1-k-i]`.
    #    Let `j = N-1-k-i`.
    #    If h is symmetric, it might be related.
    #    
    #    Crucially, `max(conv(h, h))` is NOT necessarily `sum(h^2)`.
    #    `sum(h^2)` is `np.correlate(h, h)[N-1]`.
    #    
    #    So my Cauchy-Schwarz bound on `max(conv)` was based on the wrong term.
    #    `max(conv)` can be much smaller than `sum(h^2)`.
    #    
    #    Therefore, Value < 2.0 is possible!
    #    
    #    We want to minimize `max(conv(h, h))`.
    #    This is equivalent to minimizing the peak of the convolution of h with itself.
    #    This is a "Low Peak-to-Average Power Ratio" (PAPR) problem, but for convolution.
    #    
    #    A chirp signal has low peak sidelobes in its autocorrelation.
    #    Does it have low peak in its convolution?
    #    `conv(h, h)` is related to the magnitude squared of the Fourier Transform.
    #    `FFT(conv(h, h)) = |FFT(h)|^2`.
    #    So `conv(h, h)` is the inverse FFT of the power spectrum.
    #    
    #    We want to minimize the peak of the inverse FFT of the power spectrum.
    #    This is equivalent to making the power spectrum as "flat" as possible?
    #    If the power spectrum is flat (white noise), the inverse FFT is a delta function (peak at 0).
    #    If the power spectrum is concentrated, the inverse FFT is spread out?
    #    
    #    Actually, we want the time-domain signal `conv(h, h)` to have a low maximum.
    #    Since `conv(h, h)` is the autocorrelation (shifted/reversed), we want the autocorrelation to have low sidelobes.
    #    A chirp has Bessel-function-like sidelobes which decay.
    #    
    #    So the strategy of using a chirp is sound.
    #    We should optimize the chirp parameters to minimize the peak of the autocorrelation (which is `max(conv)`).
    
    bounds = [
        (-10, 10),   # f0
        (-50, 50),   # mu
        (0, 2*np.pi),# phi
        (-5, 5)      # B
    ]
    
    # Differential Evolution
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8,
            polish=True
        )
        
        if res.fun < best_val:
            best_params = res.x
            best_h = _parametric_h(best_params)
            best_val = res.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm.fun < best_val:
                best_params = res_nm.x
                best_h = _parametric_h(best_params)
                best_val = res_nm.fun
        except Exception as e:
            print(f"NM failed: {e}")
            
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
