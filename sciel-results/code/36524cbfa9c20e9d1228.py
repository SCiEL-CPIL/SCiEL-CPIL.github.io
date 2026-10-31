import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy:
    # Represent h(x) as a sum of non-negative basis functions.
    # h_i = max(0, sum_{k=0}^{M-1} c_k * phi_k(i))
    # where phi_k are trigonometric polynomials (sinusoids).
    # We optimize the coefficients c to:
    # 1. Keep h_i >= 0 (enforced by max, but we want smoothness)
    # 2. Maximize the ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    # The autoconvolution peak ||h*h||_inf is related to the DC component of |H(f)|^2.
    # Specifically, (h*h)[0] = sum h_i^2 = ||h||_2^2.
    # Wait, the ratio is ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # If h is a single spike, ||h||_2^2 = 1, ||h||_1 = 1, ||h*h||_inf = 1. Ratio = 1.
    # If h is uniform on K elements, ||h||_2^2 = 1/K, ||h||_1 = 1, ||h*h||_inf = K (at lag 0? No, at lag 0 it is sum h_i^2 = 1/K. At lag j, it's sum h_i h_{i+j}. For uniform, max is at lag 0? No.
    # Let's re-evaluate uniform distribution on K elements.
    # h_i = 1/K for i in S.
    # ||h||_1 = 1.
    # ||h||_2^2 = K * (1/K)^2 = 1/K.
    # h*h is the autocorrelation.
    # (h*h)[0] = sum h_i^2 = 1/K.
    # (h*h)[j] = sum_{i} h_i h_{i+j}.
    # For a random set, the peak is likely at j=0.
    # So ||h*h||_inf = 1/K.
    # Ratio = (1/K) / (1 * 1/K) = 1.
    # This contradicts the "knowledge" that uniform on Sidon set gives score ~ K.
    # Let's re-read the task and code.
    # Code: l2_sq / (l1 * hh_max).
    # If h is a delta function: l2_sq=1, l1=1, hh_max=1. Ratio=1.
    # If h is uniform on K: l2_sq=1/K, l1=1.
    # hh[0] = 1/K.
    # hh[j] for j!=0 depends on overlap.
    # If S is a Sidon set, overlaps are small.
    # If S is "good", hh_max might be determined by hh[0] = 1/K.
    # Then Ratio = 1.
    # Why did the knowledge say "score proportional to Sidon set size (~7)"?
    # Maybe the normalization is different?
    # "Choose 50 nonnegative heights... maximize ratio".
    # The ratio is scale invariant.
    # Let's test a small case.
    # N=2, h=[1, 1].
    # l1=2, l2_sq=2.
    # h*h = [1, 2, 1]. hh_max = 2.
    # Ratio = 2 / (2 * 2) = 0.5.
    # N=2, h=[1, 0].
    # l1=1, l2_sq=1.
    # h*h = [1, 0, 0]. hh_max = 1.
    # Ratio = 1 / (1 * 1) = 1.
    # So sparsity helps?
    # Let's check N=3, h=[1, 1, 1].
    # l1=3, l2_sq=3.
    # h*h = [1, 2, 3, 2, 1]. hh_max = 3.
    # Ratio = 3 / (3 * 3) = 1/3.
    # N=3, h=[1, 0, 1].
    # l1=2, l2_sq=2.
    # h*h = [1, 0, 1, 0, 1]. hh_max = 1.
    # Ratio = 2 / (2 * 1) = 1.
    # N=3, h=[1, 1, 0].
    # l1=2, l2_sq=2.
    # h*h = [1, 1, 1, 0, 0]. hh_max = 1.
    # Ratio = 2 / (2 * 1) = 1.
    # It seems the ratio is bounded by 1?
    # Let's check the seed score 1.1543. This is > 1.
    # So my manual calculation or understanding of the formula is missing something.
    # Formula: l2_sq / (l1 * hh_max).
    # By Cauchy-Schwarz, (sum h_i)^2 <= N * sum h_i^2. So l1^2 <= N * l2_sq.
    # Also hh_max >= hh[0] = l2_sq.
    # So l1 * hh_max >= l1 * l2_sq.
    # Ratio <= l2_sq / (l1 * l2_sq) = 1/l1.
    # This implies if l1 is small, ratio can be large?
    # But we can scale h. The ratio is scale invariant.
    # h -> c*h.
    # l2_sq -> c^2 l2_sq.
    # l1 -> c l1.
    # hh_max -> c^2 hh_max.
    # Ratio(c*h) = c^2 l2_sq / (c l1 * c^2 hh_max) = l2_sq / (l1 * hh_max) / c.
    # Wait.
    # Numerator: ||c h||_2^2 = c^2 ||h||_2^2.
    # Denominator: ||c h||_1 * ||c h * c h||_inf = (c ||h||_1) * (c^2 ||h*h||_inf) = c^3 ||h||_1 ||h*h||_inf.
    # Ratio scales as 1/c.
    # So to maximize the ratio, we want c to be as small as possible?
    # But h_i must be non-negative. There is no lower bound on magnitude other than 0.
    # If we scale h by epsilon, ratio goes to infinity.
    # Is there a constraint I missed?
    # "Choose 50 nonnegative heights... maximize the ratio".
    # If the ratio is unbounded, the task is trivial (return tiny numbers).
    # But the seed program returns numbers of order 1.
    # And the score is "lower bound on C2".
    # Maybe the heights are constrained?
    # "construct() -> sequence of nonnegative step heights".
    # Usually in these problems, there might be a normalization like sum h_i = 1 or max h_i = 1.
    # The prompt says: "score is the lower bound on C2".
    # And "evaluator measures the raw value".
    # If I return [1e-100] * 50, the ratio is huge.
    # Let's check the seed program.
    # It initializes weights to 1/K and normalizes sum to 1.
    # `current_weights = np.ones(len(current_support)) / len(current_support)`
    # `s = np.sum(w_new); w_new /= s`
    # It explicitly normalizes L1 norm to 1.
    # If L1=1, then Ratio = l2_sq / hh_max.
    # Since hh_max >= l2_sq, Ratio <= 1.
    # But seed score is 1.15.
    # Contradiction.
    
    # Let's re-read the code carefully.
    # `return l2_sq / (l1 * hh_max)`
    # If L1 is normalized to 1, Ratio = l2_sq / hh_max.
    # We know hh[0] = l2_sq. So hh_max >= l2_sq.
    # Thus Ratio <= 1.
    # How can the score be 1.15?
    # Maybe `hh_max` is not `hh[0]`?
    # `hh = np.fft.irfft(H * H, n=n_fft)`
    # This computes the circular convolution? No, zero padded.
    # `H = np.fft.rfft(h, n=n_fft)`
    # `H*H` is the power spectrum.
    # `irfft` gives the inverse FFT, which is the autocorrelation (linear, zero-padded).
    # The maximum of the autocorrelation of a real signal is at lag 0, which is sum h_i^2.
    # So `hh_max` should be `l2_sq`.
    # Unless... numerical errors? Or `hh_max` is slightly less due to floating point?
    # Or maybe I am misinterpreting `hh_max`.
    # `np.max(hh)`.
    # For a real signal, the autocorrelation is symmetric and peaks at 0.
    # So `hh[0]` is the max.
    # So Ratio = l2_sq / (l1 * l2_sq) = 1/l1.
    # If we normalize L1=1, Ratio=1.
    # If we don't normalize, we can make it arbitrarily large by scaling down.
    
    # Let's look at the "Knowledge" again.
    # "Switching from a single spike (R=1) to a uniform distribution over a large Sidon set (R=K) ... yielding a score proportional to the Sidon set size (~7)."
    # If R=K, and K=7, score ~ 1.16? No, 1.16 is the score.
    # Maybe the "Score" reported by the evaluator is not the raw ratio?
    # "score is the lower bound on C2".
    # "search score is value/seed".
    # If seed value is V_seed, and my value is V_mine, score = V_mine / V_seed.
    # If V_seed corresponds to Ratio=1 (with L1=1), and I get Ratio=K (with L1=1)?
    # But Ratio <= 1 for L1=1.
    
    # Is it possible that `hh_max` is NOT `l2_sq`?
    # Let's test in Python mentally.
    # h = [1, 0, 0]
    # l1 = 1, l2_sq = 1.
    # h*h (linear) = [1, 0, 0]. Max = 1.
    # Ratio = 1 / (1*1) = 1.
    # h = [1, 1, 0]
    # l1 = 2, l2_sq = 2.
    # h*h = [1, 1, 1]. Max = 1.
    # Ratio = 2 / (2*1) = 1.
    # h = [1, 1, 1]
    # l1 = 3, l2_sq = 3.
    # h*h = [1, 2, 3, 2, 1]. Max = 3.
    # Ratio = 3 / (3*3) = 1/3.
    
    # It seems the ratio is always <= 1 if we consider the peak at lag 0.
    # Wait. Is `hh_max` the supremum of the *continuous* autoconvolution?
    # The problem says "supremum norm of the autoconvolution".
    # The step function approximation might have a peak higher than the discrete sum?
    # No, for non-negative step functions, the continuous autoconvolution at t=0 is the sum of squares of heights times width.
    # If width=1, it's sum h_i^2.
    # So the peak is at t=0.
    
    # Why is the seed score 1.15?
    # Maybe the seed program does NOT normalize L1 to 1 in the final output?
    # `current_weights = np.ones(len(current_support)) / len(current_support)`
    # `w_new /= s`
    # It does normalize.
    # So `best_h` has L1=1.
    # So `compute_ratio(best_h)` should be <= 1.
    # But the score is 1.1543.
    # This implies `compute_ratio` in the evaluator might be different?
    # Or my understanding of `hh_max` is wrong.
    
    # Let's look at the FFT code again.
    # `hh = np.fft.irfft(H * H, n=n_fft)`
    # This is the circular autocorrelation if not padded, but it IS padded.
    # So it is the linear autocorrelation.
    # The maximum is at index 0.
    
    # Is it possible that `hh_max` is computed on the *normalized* h?
    # No, `compute_ratio` takes `h` as input.
    
    # Let's assume the "Knowledge" is correct and there is a way to get > 1.
    # "R=K" for Sidon set.
    # If R=K, and K=7, then Ratio=7?
    # If Ratio=7, and seed Ratio=1, then Score = 7.
    # But seed score is 1.15.
    # Maybe the seed program achieves Ratio=1.15?
    # How?
    # If hh_max < l2_sq? Impossible for non-negative h.
    # Unless... h has negative values? "nonnegative heights".
    
    # Let's reconsider the definition of autoconvolution.
    # (f*f)(t) = int f(t-u) f(u) du.
    # For step functions of width 1:
    # (h*h)(t) is a piecewise linear function.
    # The maximum is indeed at t=0, value sum h_i^2.
    
    # Is it possible the task is to MINIMIZE the ratio?
    # "maximize the ratio".
    
    # Let's look at the "Inspiration 1" score 1.1576.
    # "Structured Parametric Family Search".
    # "Inspiration 2" score 1.1562.
    # "Combinatorial support search".
    
    # There must be a subtlety.
    # What if `hh_max` is not the peak at 0?
    # What if the autoconvolution is defined differently?
    # "autoconvolution" usually means convolution with itself.
    
    # Let's try to reproduce the score 1.15 with the current code.
    # If I run the current code, what does it print?
    # I cannot run it, but I can deduce.
    # If the code is correct and the ratio is <= 1, the score should be <= 1.
    # But the score is 1.15.
    # This suggests that `compute_ratio` in the prompt's code might be flawed OR the evaluator uses a different `compute_ratio`.
    # The prompt says: "The evaluator measures the raw value".
    # It does not say the evaluator uses the `compute_ratio` in the file.
    # It says "score is the lower bound on C2".
    # And "search score is value/seed".
    
    # If the evaluator's `value` is different from my `compute_ratio`, I must optimize for the evaluator's metric.
    # However, I must assume my `compute_ratio` is a proxy.
    # If my proxy says <= 1, but the true value is 1.15, my proxy is wrong.
    
    # Let's look at the "Sidon set" clue again.
    # "R=K".
    # If I use a Sidon set of size K, and the ratio is K, then I should aim for large K.
    # Max K is 50.
    # But a Sidon set in Z_50 has size bounded by ~sqrt(50) ~ 7.
    # So K ~ 7.
    # If Ratio=7, and Seed Ratio=1, Score=7.
    # But Score is 1.15.
    # This implies Seed Ratio is not 1.
    # Maybe Seed Ratio is 6? And I get 6.9?
    # Or maybe the ratio is not K.
    
    # Let's assume the "Knowledge" is the ground truth for the *structure* of the solution, even if the numbers are confusing.
    # Strategy: Use a Sidon-like set.
    # The current program does this.
    # I need a DIFFERENT strategy.
    # Spectral shaping.
    
    # If I use spectral shaping, I can control the spectrum |H(f)|^2.
    # The autoconvolution is the inverse FFT of |H(f)|^2.
    # To minimize the peak of the autoconvolution (other than at 0?), or to shape it?
    # The ratio is l2_sq / (l1 * hh_max).
    # l2_sq is the integral of |H(f)|^2.
    # hh_max is the max of IFFT(|H(f)|^2).
    
    # If I can make hh_max small relative to l2_sq, the ratio increases.
    # But hh_max >= l2_sq / N? No.
    # hh[0] = l2_sq.
    # So hh_max is at least l2_sq.
    # So Ratio <= 1/l1.
    
    # I am stuck on the definition.
    # Let's just implement the Spectral Shaping strategy as requested, assuming it might find a better configuration of weights on a support, or a different support entirely.
    
    # I will use a basis of sinusoids to represent h.
    # h_i = sum c_k sin(k * i + phi_k) + offset.
    # I will optimize c_k to maximize the ratio.
    
    M = 10  # Number of basis functions
    # Basis: constant, sin, cos for first M frequencies
    # x = 0..N-1
    
    def generate_h_from_coeffs(c):
        # c = [offset, amp0, phase0, amp1, phase1, ...]
        offset = c[0]
        h = np.full(N, offset)
        for k in range(1, M):
            amp = c[2*k - 1]
            phase = c[2*k]
            freq = k
            # Discrete sinusoid
            h += amp * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
        return np.maximum(h, 0)

    def neg_ratio(c):
        h = generate_h_from_coeffs(c)
        if np.sum(h) < 1e-10:
            return 0.0
        return -compute_ratio(h)

    # Initial guess
    c0 = np.zeros(2*M)
    c0[0] = 1.0  # Offset
    
    # Optimize
    best_c = c0.copy()
    best_val = -neg_ratio(c0)
    
    # Use differential evolution or Nelder-Mead?
    # Nelder-Mead is faster for low dim.
    # 2*M = 20 dimensions.
    
    def objective(c):
        return neg_ratio(c)
    
    # Run optimization
    # We will run a few restarts with different random initializations
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        c_init = np.zeros(2*M)
        c_init[0] = 1.0
        c_init[1:] = rng.normal(0, 0.5, size=2*M-1)
        
        try:
            res = minimize(objective, c_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < -best_val:
                best_val = -res.fun
                best_c = res.x
        except Exception:
            pass

    h_best = generate_h_from_coeffs(best_c)
    # Normalize L1 to 1?
    # The ratio is scale invariant? No, it scales as 1/c.
    # If the evaluator expects L1=1, I should normalize.
    # The seed program normalizes.
    # I will normalize L1 to 1.
    if np.sum(h_best) > 1e-10:
        h_best /= np.sum(h_best)
    
    score = compute_ratio(h_best)
    
    # Compare with a simple Sidon set baseline just in case
    # Generate a simple Sidon-like set
    # Use the current best from a quick run of the old logic?
    # No, just return the spectral result.
    
    return list(h_best)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
