"""Optimized program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: evaluator kills after 150s wall-clock.
    # We must return within 120s per task contract, so use a safety margin.
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin
    
    # Interval width
    # Domain is [-1/4, 1/4], length = 1/2. N=600 intervals.
    # Width of each interval: h = (1/2) / 600 = 1/1200
    h = 1.0 / (2.0 * N)
    
    # Initial guess: uniform distribution
    # f_i >= 0. The autoconvolution (correlation) at lag 0 is sum f_i^2 * h (approximately, depending on normalization).
    # Actually, let's look at the definition of "normalized maximum of its autoconvolution".
    # Usually for such problems, we consider the function f(x) on the interval.
    # The autoconvolution (or autocorrelation) C(t) = integral f(x) f(x+t) dx.
    # The "normalized maximum" likely refers to max_t C(t) / (integral f)^2 or similar.
    # Let's assume the standard normalization: C(0) = integral f^2.
    # If we scale f by c, C scales by c^2. So the ratio max C(t) / C(0) is scale-invariant.
    # Let's define the objective as max_t (C(t) / C(0)).
    # Note: C(0) = sum f_i^2 h.
    # C(t) for t = k*h is approx sum f_i f_{i+k} h.
    # We want to minimize max_k (sum f_i f_{i+k} h) / (sum f_i^2 h).
    # This is equivalent to minimizing max_k (sum f_i f_{i+k}) / (sum f_i^2).
    
    # Let x be the vector of heights.
    # Objective: min max_k (x @ circulant_k(x)) / (x @ x)
    # This is a non-smooth problem. We can use a soft-max approximation or iterative minimization.
    
    # Strategy:
    # 1. Use a smooth approximation for the max: soft_max(z) = log(sum(exp(z_i/s))) * s.
    # 2. Optimize x using scipy.optimize.minimize.
    # 3. Constraint: x >= 0. We can use a log-barrier or just clip negative values (but gradient gets messy).
    #    Better: parameterize x = exp(y) to ensure positivity? No, that might be hard to optimize if x can be 0.
    #    Let's try projecting onto non-negative orthant after each step, or using bounds.
    #    scipy.optimize.minimize supports bounds.
    
    # Let's use a simple iterative approach or gradient descent with projection.
    # Given the time limit, a robust optimizer like L-BFGS-B with bounds is good.
    
    # Define the objective function
    def objective(x):
        # x is shape (N,)
        # Compute autocorrelation coefficients r_k for k=0..N-1
        # r_k = sum_{i=0}^{N-1-k} x_i x_{i+k}
        # We can use FFT to compute this efficiently.
        # Zero-pad x to length 2N to avoid circular convolution artifacts for lags up to N-1.
        # Actually, linear convolution of x with x gives lags 0..2N-2.
        # We only need lags 0..N-1 (positive lags). Due to symmetry, r_{-k} = r_k.
        
        # FFT method:
        # F = FFT(x, n=2*N)
        # R = IFFT(|F|^2)
        # R[0:N] contains r_0, r_1, ..., r_{N-1}
        
        n_fft = 2 * N
        F = np.fft.fft(x, n=n_fft)
        R = np.fft.ifft(np.abs(F)**2).real[:N]
        
        # Normalize by r_0 = sum x^2
        r0 = R[0]
        if r0 < 1e-12:
            return 0.0 # Degenerate case, avoid division by zero
        
        # We want to minimize max(R / r0).
        # Since R[0] = r0, R[0]/r0 = 1.
        # The max is at least 1.
        # Wait, is the objective max C(t) / C(0)?
        # If so, the minimum possible value is 1 (if C(t) <= C(0) for all t, which is true for autocorrelation).
        # But we want to MINIMIZE the maximum.
        # If the max is always >= 1, then we are minimizing something that is >= 1.
        # The uniform distribution gives:
        # r_k = (N-k) * 1^2 = N-k for k=0..N-1.
        # r_0 = N.
        # r_k / r_0 = 1 - k/N.
        # Max is 1 (at k=0).
        # So the uniform distribution gives a score of 1.0?
        # The task says "score is the upper bound on C1".
        # And "1.0 equals the seed program".
        # The seed program is uniform [1.0]*N.
        # So the seed score is 1.0.
        # We want a score LOWER than 1.0?
        # "lower is better".
        # But max C(t)/C(0) is always >= 1 for autocorrelation (by Cauchy-Schwarz, C(0)^2 >= C(t)^2 is not generally true, but C(0) is the max of the autocorrelation for positive definite functions? No, autocorrelation peaks at 0).
        # Yes, for any real function, the autocorrelation C(0) is the maximum value of C(t).
        # So max C(t) / C(0) is always 1.
        # This implies the score would always be 1.0.
        
        # Let's re-read the task carefully.
        # "minimize the normalized maximum of its autoconvolution"
        # Autoconvolution vs Autocorrelation?
        # Autoconvolution: (f * f)(t) = integral f(x) f(t-x) dx.
        # Autocorrelation: (f * f~)(t) = integral f(x) f(x+t) dx.
        # For real even functions, they are similar but autoconvolution is symmetric around 0 and peaks at 0.
        # For autoconvolution g(t) = integral f(x) f(t-x) dx.
        # g(0) = integral f(x) f(-x) dx. If f is non-negative, this is just integral f(x)^2 dx? No.
        # If f is supported on [-1/4, 1/4], and we consider t=0, g(0) = integral f(x) f(-x) dx.
        # If f is not even, this is not sum f^2.
        # However, usually in these "step function" problems on a symmetric interval, we might assume f is non-negative.
        # If f is non-negative, g(0) = integral f(x) f(-x) dx.
        # The maximum of g(t) is not necessarily at t=0 if f is not even.
        # But for non-negative f, the maximum of the autoconvolution is usually at t=0 if f is "concentrated", but can be elsewhere?
        # Actually, for non-negative functions, the autoconvolution is non-negative.
        # Is it true that max_t g(t) >= g(0)?
        # If f is a delta function at x0, g is a delta at 2x0. Max is at 2x0.
        # If f is uniform on [-1/4, 1/4], g(t) is a triangle peaking at 0.
        
        # Let's check the normalization.
        # "normalized maximum".
        # Maybe normalized by the integral?
        # Score = max_t g(t) / (integral f)^2 ?
        # For uniform f on [0,1] (scaled to N=1 for a moment), integral = 1.
        # g(t) is triangle with height 1 at t=0. Score = 1.
        # If we make f a spike, integral is small, but max g is large?
        # Let's use the discrete definition.
        # Let x be the heights.
        # Integral approx = sum x_i h.
        # Autoconvolution at lag k (t = k*h):
        # g_k = sum_{i} x_i x_{i+k} h^2 ?
        # The continuous autoconvolution G(t) = integral f(x) f(t-x) dx.
        # Discrete approximation: G(k*h) approx sum_{i} x_i x_{i+k} h^2.
        # Normalization: (Integral f)^2 = (sum x_i h)^2 = (sum x_i)^2 h^2.
        # So the normalized value is (sum x_i x_{i+k} h^2) / ((sum x_i h)^2) = (sum x_i x_{i+k}) / (sum x_i)^2.
        
        # Let S = sum x_i.
        # Objective: minimize max_k (sum_{i=0}^{N-1-k} x_i x_{i+k}) / S^2.
        
        # For uniform x_i = 1:
        # S = N.
        # Numerator for k=0: sum 1*1 = N.
        # Value = N / N^2 = 1/N.
        # For k=1: sum 1*1 = N-1.
        # Value = (N-1)/N^2.
        # Max is 1/N.
        # So the seed score is 1/N?
        # But the task says "1.0 equals the seed program".
        # This implies the score returned by the evaluator is 1.0 for the seed.
        # So the score is NOT 1/N.
        # Maybe the normalization is different.
        
        # Alternative: "normalized maximum of its autoconvolution"
        # Maybe it's max G(t) / G(0)?
        # For uniform, G(t) peaks at 0, so ratio is 1.
        # If the score is 1.0 for seed, and we want to lower it, we need a function where the peak of the autoconvolution is NOT at 0, and the peak value is LOWER than the value at 0?
        # No, if we normalize by G(0), the max is always >= 1?
        # Wait, if the max is at t!=0, and G(t_max) < G(0), then the max of the normalized function (divided by G(0)) is still 1 (at t=0).
        # So if the normalization is by G(0), the score is always 1.
        
        # Therefore, the normalization must be by something else, likely the total mass (integral) squared.
        # If Score = max G(t) / (Integral f)^2.
        # Seed: Score = 1/N.
        # But the prompt says "1.0 equals the seed program".
        # This is a contradiction unless the "score" reported is scaled.
        # "score is the upper bound on C1".
        # "The evaluator measures the raw value (lower is better). The search score is seed/value..."
        # This implies:
        # Search Score = Seed_Value / Current_Value.
        # If Seed_Value = 1.0 (in search score terms), then:
        # 1.0 = Seed_Raw / Seed_Raw.
        # So the "Seed_Value" in the search score formula is the raw value of the seed.
        # If the seed raw value is V_seed, and we return V_new, the search score is V_seed / V_new.
        # The prompt says "1.0 equals the seed program". This means if we return the seed, the search score is 1.0.
        # This is consistent.
        # The question is: what is V_seed?
        # If V_seed = 1/N, then to get a search score > 1.0, we need V_new < 1/N.
        # Can we get V_new < 1/N?
        # V = max_k (sum x_i x_{i+k}) / (sum x_i)^2.
        # Let's try a "bipolar" distribution? No, x >= 0.
        # Let's try two spikes at the ends?
        # x_0 = 1, x_{N-1} = 1, others 0.
        # S = 2.
        # k=0: sum x_i^2 = 2. Val = 2/4 = 0.5.
        # k=1: sum x_i x_{i+1} = 0.
        # ...
        # k=N-1: x_0 x_{N-1} = 1. Val = 1/4 = 0.25.
        # Max is 0.5.
        # For N=600, 1/N is ~0.0016.
        # 0.5 is much larger than 0.0016. So spikes are worse.
        
        # We want to MINIMIZE the max.
        # Uniform gives 1/N.
        # Can we go lower?
        # Note that sum_k (sum x_i x_{i+k}) is not simple.
        # However, note that for k=0, term is sum x_i^2.
        # By Cauchy-Schwarz, (sum x_i)^2 <= N * sum x_i^2.
        # So sum x_i^2 / (sum x_i)^2 >= 1/N.
        # The term for k=0 is ALWAYS >= 1/N.
        # Therefore, the max is ALWAYS >= 1/N.
        # So the minimum possible raw value is 1/N.
        # The uniform distribution achieves 1/N.
        # So the uniform distribution is optimal!
        
        # Why would the task exist then?
        # "Choose a nonnegative step function ... to minimize the normalized maximum of its autoconvolution".
        # Maybe the autoconvolution is defined differently?
        # Or maybe the normalization is different?
        
        # Re-read: "upper bound on C1".
        # C1 might be a constant in an inequality.
        # "First autocorrelation inequality".
        # Perhaps the quantity is max |C(t)| / C(0)?
        # If so, it's always 1.
        
        # Let's look at the "Inspiration" or "Lessons".
        # "Higher resolution FFT reduces discretization error... Measured: 1.212 -> 1.237".
        # This implies scores CAN be > 1.0.
        # If the seed is 1.0, and we can get 1.237, that means V_seed / V_new = 1.237 => V_new = V_seed / 1.237.
        # So V_new < V_seed.
        # This contradicts my derivation that V_min = 1/N (achieved by uniform).
        
        # Unless... the "autoconvolution" is not the standard one?
        # Or the "normalized maximum" is not max G(t) / (Int f)^2?
        
        # What if the normalization is by the max of the function?
        # Score = max G(t) / max f?
        # Uniform: max f = 1. max G(t) (discrete, unnormalized sum) = N.
        # Score = N.
        # Spikes: max f = 1. max G(t) = 1 (at lag 0, 1*1).
        # Score = 1.
        # 1 < N. So spikes are better.
        # If seed score is 1.0, and seed is uniform...
        # If Uniform Score = N, then Search Score = N / N = 1.
        # If Spikes Score = 1, then Search Score = N / 1 = N = 600.
        # This would be a huge improvement.
        
        # Let's check if this interpretation fits "upper bound on C1".
        # In harmonic analysis, C1 might relate to the L1 norm of the Fourier transform or similar.
        
        # Let's test the "max G(t) / max f" hypothesis.
        # Objective: minimize max_k (sum_{i} x_i x_{i+k}) / max(x).
        # Since x >= 0, we can scale x.
        # Let x = c * y, where y >= 0, max(y) = 1.
        # G(t) scales with c^2.
        # Score = c^2 * max_k (sum y_i y_{i+k}) / (c * 1) = c * max_k (sum y_i y_{i+k}).
        # To minimize this, we should make c as small as possible?
        # But x is the step function. Is there a constraint on the integral or max?
        # "Choose a nonnegative step function".
        # If there is no constraint, we can make c -> 0, and the score -> 0.
        # That would be trivial.
        
        # So there MUST be a normalization that makes the score scale-invariant.
        # Common normalizations:
        # 1. max G(t) / G(0)  (Always >= 1, min 1)
        # 2. max G(t) / (Int f)^2 (Min 1/N, achieved by uniform)
        # 3. max G(t) / (Int f) * (something)
        
        # Let's reconsider the "1.212 -> 1.237" clue.
        # If the seed is uniform, and the optimal is uniform, we can't improve.
        # So the seed is NOT optimal.
        # This means my derivation that Uniform is optimal for case (2) must be wrong, OR the objective is different.
        
        # What if the autoconvolution is circular?
        # "600 equal intervals on [-1/4, 1/4]".
        # If it's a step function on a compact interval, autoconvolution is usually linear.
        # But if it's treated as periodic (Fourier series context?), maybe circular?
        # If circular autoconvolution:
        # r_k = sum_{i=0}^{N-1} x_i x_{(i+k) mod N}.
        # For uniform x=1:
        # r_k = N for all k.
        # Normalization by (Int f)^2 = N^2 h^2?
        # Discrete: (sum x)^2 = N^2.
        # r_0 = N.
        # Score = N / N^2 = 1/N.
        # Still 1/N.
        
        # What if the normalization is by sum x^2 (Energy)?
        # Score = max r_k / r_0.
        # For uniform, r_k = N, r_0 = N. Score = 1.
        # For spikes (two 1s at distance d), r_0 = 2, r_d = 1, r_{-d}=1.
        # Max r_k = 2 (at k=0).
        # Score = 2/2 = 1.
        # Still 1.
        
        # Is it possible that the "autoconvolution" refers to the Fourier transform magnitude squared?
        # |F|^2?
        # "Autoconvolution" in time domain is |F|^2 in frequency domain.
        # Maybe the score is max |F(w)|^2 / (Int f)^2?
        # For uniform, F(w) is sinc-like. Max at w=0 is N.
        # |F(0)|^2 = N^2.
        # (Int f)^2 = N^2.
        # Score = 1.
        
        # If we use spikes, F(w) is sum of cosines.
        # Max |F|^2 can be large.
        
        # Let's look at the "First autocorrelation inequality".
        # This likely refers to the inequality relating the L1 norm of the autocorrelation to the L2 norm of the function, or similar.
        # A common inequality: ||f * f||_1 <= ||f||_1^2.
        # Or: max |f*f| <= ||f||_1^2.
        
        # If the score is max |G(t)| / ||f||_1^2.
        # We established min is 1/N for uniform.
        
        # Is it possible that the intervals are NOT the support?
        # "step function with 600 equal intervals on [-1/4, 1/4]".
        
        # Let's try a different interpretation of "normalized maximum".
        # Maybe normalized by the MAX of the function?
        # And maybe the function must have a fixed Integral?
        # "Choose a nonnegative step function ... to minimize ...".
        # If no constraint, we can scale to 0.
        # So there must be an implicit constraint, e.g., Integral = 1, or Max = 1.
        
        # If Max = 1:
        # Score = max G(t). (Since max f = 1).
        # Uniform: G(t) peaks at N*h^2?
        # Discrete sum: max r_k = N.
        # Spikes: max r_k = 1.
        # 1 < N. Spikes are better.
        # Seed (Uniform) score = N.
        # Search Score = Seed_Raw / New_Raw.
        # If Seed_Raw = N, and we return Spikes (Raw=1), Search Score = N/1 = 600.
        # This fits the "1.212" improvement magnitude? No, 600 is huge.
        # But maybe the "Raw" value is normalized differently in the evaluator.
        
        # However, if the seed is uniform, and we can get 600x improvement, the task would be trivial.
        # The fact that there are scores like 1.212 suggests small improvements.
        
        # This implies the seed is CLOSE to optimal.
        # Uniform is optimal for max G(t) / (Int f)^2.
        # So maybe the objective IS max G(t) / (Int f)^2, and the seed is NOT uniform?
        # But the seed code is `return [1.0] * N`. That IS uniform.
        
        # Contradiction:
        # 1. Seed is Uniform.
        # 2. Uniform is optimal for standard normalization.
        # 3. Improvements exist (1.212).
        
        # Resolution: The standard normalization I assumed is wrong.
        
        # What if the "autoconvolution" is the autocorrelation, and the normalization is by the MEAN?
        
        # Let's try: Score = max_k r_k / (1/N sum r_k)?
        
        # Let's try: Score = max_k r_k / r_0.
        # We know this is >= 1.
        # If the seed score is 1.0, and we can get 1.212, that means V_seed / V_new = 1.212.
        # So V_new = 1 / 1.212 = 0.82.
        # But V_new = max r_k / r_0 >= 1.
        # So V_new cannot be 0.82.
        
        # Therefore, the score CANNOT be max r_k / r_0.
        
        # It must be a quantity that can be LESS than the seed value.
        # Seed Value = V_seed.
        # We need V_new < V_seed.
        
        # If V_seed = 1.0 (in the context of the search score being 1.0), then we need V_new < 1.0.
        
        # What quantity associated with autoconvolution is <= 1 for uniform, and can be smaller?
        
        # Consider: max G(t) / G(0) is >= 1.
        # Consider: min G(t) / G(0)?
        # "minimize the normalized maximum". No, it says maximize.
        
        # Maybe the "autoconvolution" is negative somewhere?
        # No, f >= 0 implies G >= 0.
        
        # What if the normalization is by the L2 norm?
        # Score = max G(t) / ||f||_2^2.
        # For uniform: G(0) = N (discrete sum). ||f||_2^2 = N.
        # Score = 1.
        # For spikes: G(0) = 2. ||f||_2^2 = 2.
        # Score = 1.
        # Always 1?
        # G(0) = sum x_i^2 = ||f||_2^2.
        # So G(0) / ||f||_2^2 = 1.
        # Max G(t) >= G(0).
        # So Score >= 1.
        
        # It seems almost all standard normalizations give a minimum of 1 (or 1/N) achieved by uniform or similar.
        
        # Let's look at the "1.212" again.
        # If the search score is 1.212, and seed is 1.0, then V_seed / V_new = 1.212.
        # V_new = 0.82 V_seed.
        
        # If V_seed = 1.0 (Raw), then V_new = 0.82.
        
        # What raw value can be 0.82?
        
        # Maybe the "normalized maximum" is:
        # max_t G(t) / (max_t G(t) + something)?
        
        # Let's guess the evaluator uses:
        # Score = max(G) / (Integral f)^2.
        # And the seed is NOT uniform?
        # But the code is `return [1.0] * N`.
        
        # Is it possible that the "intervals" imply a specific scaling?
        
        # Let's assume the task is to minimize the L-infinity norm of the autoconvolution, normalized by the L1 norm squared.
        # And assume that for some reason, the uniform distribution is NOT optimal.
        # This happens if the autoconvolution is CIRCULAR and we are looking at the max over the circle?
        # No, uniform is still optimal for circular.
        
        # What if we are allowed to choose the WIDTH of the intervals?
        # "600 equal intervals". Fixed.
        
        # Let's try to find a function that beats uniform for max G(t) / (Int f)^2.
        # We proved max >= 1/N.
        # So uniform is optimal.
        
        # There is only one possibility left: The "autoconvolution" is defined with a NEGATIVE sign or difference?
        # Or "autoconvolution" means (f - c) * (f - c)?
        
        # Given the ambiguity and the "1.212" clue, let's assume the objective is:
        # Minimize max_k (r_k) subject to sum x_i = 1 and x_i >= 0.
        # (Normalization by Int f = 1).
        # Then Score = max r_k.
        # Uniform: r_k = (N-k)/N^2. Max = 1/N.
        # Spikes: r_0 = 1 (if one spike of height 1). Max = 1.
        # 1/N < 1. Uniform is better.
        
        # So Uniform is robustly optimal for standard metrics.
        
        # Why is the seed score 1.0 and we can improve?
        # Maybe the seed is NOT the uniform distribution in the evaluator's eyes?
        # Or maybe the "1.0" is just the baseline, and the raw value is something else.
        
        # Let's try a different strategy:
        # If I can't determine the exact metric, I should try to minimize the "peakedness" of the autoconvolution.
        # This means making the autoconvolution as flat as possible.
        # Uniform gives a triangular autoconvolution (linear decay).
        # Can we make it flatter?
        # A "box" function gives a triangle.
        # A "raised cosine" or smoother function might give a smoother decay?
        # But we are restricted to step functions.
        
        # If we make the function non-monotonic, e.g., high in the middle, low at edges?
        # This might reduce the tails of the autoconvolution.
        
        # Let's try to optimize x to minimize max(r_k / S^2).
        # Even if uniform is theoretically optimal, maybe numerical issues or a slightly different metric (e.g. circular vs linear) makes a slight variation better.
        
        # I will implement an optimizer that minimizes max(r_k) / (sum x)^2.
        # I will start from uniform and perturb it.
        
    # Define objective for optimizer
    # x: array of N non-negative numbers.
    def obj(x):
        S = np.sum(x)
        if S == 0:
            return 0
        # Compute linear autocorrelation r_k for k=0..N-1
        # Use FFT
        n_fft = 2 * N
        F = np.fft.fft(x, n=n_fft)
        R = np.fft.ifft(np.abs(F)**2).real[:N]
        # Normalize
        norm = R / (S * S)
        # We want to minimize the max
        # Use soft max for gradient
        # max(norm)
        # Since norm[0] is always 1 (if we normalize by S^2, R[0]=sum x^2, so R[0]/S^2 <= 1? No.
        # R[0] = sum x^2. S^2 = (sum x)^2.
        # sum x^2 <= (sum x)^2. So R[0]/S^2 <= 1.
        # Wait, for uniform, R[0]/S^2 = N / N^2 = 1/N.
        # So the max is 1/N.
        
        # If we minimize max(norm), we are minimizing 1/N?
        # It's constant for uniform.
        
        # Let's just return the max.
        return np.max(norm)
    
    # Initial guess
    x0 = np.ones(N)
    
    # Bounds
    bounds = [(0, None)] * N
    
    # Optimization
    # Use L-BFGS-B
    # Note: This might be slow.
    
    # Check time
    if time.time() > deadline:
        return list(x0)
        
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    # Ensure non-negative
    x_opt = np.clip(x_opt, 0, None)
    
    # Final check
    if time.time() > deadline:
        return list(x0)
        
    return list(x_opt)

# EVOLVE-BLOCK-END
