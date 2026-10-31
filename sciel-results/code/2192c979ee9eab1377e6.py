import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Strategy: Optimize a smooth "bump" function defined by continuous parameters.
    # The function is a sum of Gaussians or a single smoothed box.
    # Let's try a single parameterized function: f(x) = exp(-alpha * (x-x0)^2) * H(x-x0+width/2) * H(x0+width/2-x)
    # Or simpler: A smooth window.
    # Let's use a function defined by a few parameters:
    # 1. Amplitude scale (normalized later)
    # 2. Width of the core
    # 3. Tapering exponent (for a polynomial taper or Gaussian-like)
    
    # Actually, let's try optimizing the shape directly in a low-dimensional parameter space
    # that allows for asymmetric or non-trivial shapes, then project to the 600 bins.
    
    # Let's try a function of the form:
    # f(x) = (1 - (2|x|/W)^p)^(1/q) for |x| < W/2, else 0
    # where W is width, p is shape, q is exponent.
    # This generalizes the triangle (p=1, q=1) and box (p->inf, q=1).
    
    def score_params(params):
        W, p, q = params
        # W is in units of the interval [-1/4, 1/4], so max W is 0.5
        # p is shape parameter (1 is triangle, high is box-like)
        # q is height taper
        
        # Discretize
        # The domain is [-1/4, 1/4].
        # We have 600 bins.
        # Bin width dx = 0.5 / 600
        dx = 0.5 / N
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2
        
        # Calculate f(x)
        # Support is |x| < W/2
        # W is relative to full domain? Let's say W is absolute width.
        # Max W is 0.5.
        # Let's constrain W in [0.05, 0.5]
        
        vals = np.zeros(N)
        mask = np.abs(x) < W/2.0
        # Avoid division by zero or negative base
        # f(x) = (1 - (2|x|/W)^p)^(1/q)
        # At x=0, f=1. At x=W/2, f=0.
        
        if W <= 0:
            return 1.0 # Penalty
        
        x_abs = np.abs(x[mask])
        # Clip to avoid numerical issues outside range due to float precision
        u = np.clip(x_abs / (W/2.0), 0.0, 1.0 - 1e-12)
        
        # Calculate shape
        # If p is small, it's flatter? No, (1-u^p). If p=1, linear. If p=2, parabolic.
        # We want to optimize the peak of autoconvolution.
        # The autoconvolution of a smooth bump is also a smooth bump.
        # We want to minimize the max of the convolution.
        # Note: The "normalized maximum" likely means:
        # max( (f*f)(tau) ) / ( max(f) * something )?
        # Or max( (f*f)(tau) ) / ( (int f)^2 )?
        # The task says "normalized maximum of its autoconvolution".
        # Standard C1 bound context: C1 is often related to ||f||_2^4 / ||f*f||_infty?
        # Or ||f*f||_infty / ||f||_2^4?
        # If we scale f by c, f*f scales by c^2.
        # If the metric is scale-invariant, we can fix one norm.
        # Let's assume the metric is:
        # Score = max(f*f) / (sum(f^2 * dx)^2) ??
        # Or maybe:
        # The "first autocorrelation inequality" usually bounds C1.
        # If f is non-negative, C1 <= (max f*f) / (||f||_2^4) ?
        # Let's test with the constant function.
        # f(x) = 1 on [-1/4, 1/4].
        # ||f||_2^2 = 0.5.
        # ||f||_2^4 = 0.25.
        # f*f(0) = 0.5.
        # Ratio = 0.5 / 0.25 = 2.
        # But the seed score is 1.0.
        
        # Maybe the normalization is different.
        # "normalized maximum of its autoconvolution"
        # If the seed [1.0]*600 scores 1.0, and the value is "upper bound on C1".
        # Let's look at the definition of C1.
        # In the context of the uncertainty principle or signal processing, C1 might be:
        # C1 = sup_f ( ||f||_1^2 / ||f||_2^2 )? No.
        
        # Let's re-read: "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
        # If the seed is constant, let's calculate common ratios.
        # 1. max(f*f) / max(f^2)?
        # max(f*f) = 0.5. max(f^2) = 1. Ratio = 0.5. Not 1.
        # 2. max(f*f) / (int f)^2 ?
        # int f = 0.5. (int f)^2 = 0.25. Ratio = 2. Not 1.
        # 3. max(f*f) / (int f^2) ?
        # int f^2 = 0.5. Ratio = 0.5 / 0.5 = 1.
        # This matches the seed score of 1.0!
        # So the metric is likely: Score = max_tau (f*f)(tau) / \int f(x)^2 dx.
        
        # Let's verify if this makes sense as an upper bound on C1.
        # If C1 is defined such that (f*f)(0) <= C1 * ||f||_2^2?
        # No, usually (f*f)(0) = ||f||_2^2.
        # Wait, if the metric is max(f*f) / ||f||_2^2, then for constant f, it is 1.
        # Can we get it lower than 1?
        # (f*f)(tau) <= ||f||_2^2 ||f||_2^2 / ||f||_2^2 = ||f||_2^2?
        # By Cauchy-Schwarz, (f*f)(tau) = \int f(x) f(x+tau) dx <= sqrt(\int f^2 \int f^2) = ||f||_2^2.
        # So max(f*f) <= ||f||_2^2.
        # Therefore max(f*f) / ||f||_2^2 <= 1.
        # The maximum possible value is 1 (achieved by delta-like functions or constants? No, constants achieve 1 at tau=0).
        # Wait, if the bound is 1, how can we minimize it?
        # The question says "minimize ... gives an upper bound on C1".
        # If the minimum is 1, we can't do better?
        # Unless I have the metric backwards.
        # Maybe the metric is ||f||_2^2 / max(f*f)?
        # For constant: 0.5 / 0.5 = 1.
        # If we use a spike, max(f*f) is large, ||f||_2^2 is small?
        # If f is a spike of height H, width w.
        # ||f||_2^2 ~ H^2 w.
        # max(f*f) ~ H^2 w.
        # Ratio ~ 1.
        
        # Is it possible the metric is different?
        # "normalized maximum"
        # Maybe normalized by max(f)?
        # For constant: max(f*f)=0.5, max(f)=1. Ratio 0.5.
        # If the seed score is 1.0, and the value is 1.0, then the metric must evaluate to 1.0 for the constant.
        
        # Let's reconsider the "Upper bound on C1".
        # If C1 is a constant in an inequality like ||f||_1^2 <= C1 ||f||_2^2 (||f||_inf?)
        
        # Let's assume the evaluator calculates:
        # value = max_conv / norm_sq
        # and we want to MINIMIZE this value?
        # But CS says max_conv <= norm_sq. So value <= 1.
        # The constant function achieves 1.
        # If we want to minimize, we want it to be as small as possible.
        # Can we make max_conv much smaller than norm_sq?
        # max_conv is at least norm_sq (at tau=0? No, (f*f)(0) = ||f||_2^2).
        # Wait. (f*f)(0) = \int f(x)^2 dx = ||f||_2^2.
        # So max_tau (f*f)(tau) >= (f*f)(0) = ||f||_2^2.
        # Therefore max_conv / norm_sq >= 1.
        # The minimum possible value is 1.
        # The constant function achieves 1.
        # So the seed is already optimal??
        
        # This contradicts the task implying there is an improvement to be found.
        # Let's re-read carefully.
        # "minimize the normalized maximum of its autoconvolution"
        
        # Possibility 2: The metric is max_conv / max_f^2?
        # For constant: 0.5 / 1 = 0.5.
        # If the seed score is 1.0, maybe the evaluator multiplies by 2?
        # Or maybe the metric is 2 * max_conv / max_f^2?
        #
        # Possibility 3: The metric is max_conv / (max_f * min_f)? No, min_f is 0 for support functions.
        #
        # Possibility 4: The metric is related to the "sidelobes".
        # The peak at 0 is fixed relative to the norm.
        # Maybe the "normalized maximum" refers to the maximum EXCLUDING the peak at 0?
        # i.e. max_{tau != 0} (f*f)(tau) / ||f||_2^2.
        # For constant function, (f*f)(tau) is a triangle.
        # At tau = 0.25 (edge), it is 0.
        # At tau = 0.1, it is (0.5 - 0.1) * 1 * 1 = 0.4?
        # Let's calculate for constant f=1 on [-0.25, 0.25].
        # (f*f)(tau) = length of intersection of [-0.25, 0.25] and [-0.25-tau, 0.25-tau].
        # For tau in [0, 0.5], length = 0.5 - tau.
        # Max is at tau=0, value 0.5.
        # If we exclude tau=0, the sup is 0.5 (as tau->0).
        # So even excluding 0, it's 1.0 ratio.
        
        # Is it possible the function is not required to be supported on [-1/4, 1/4]?
        # "step function with 600 equal intervals on [-1/4,1/4]"
        # This implies the support is within this interval.
        
        # Let's look at the "Low-rank Gaussian mixture" inspiration.
        # "Measured: 1 -> 1.155"
        # If the score IMPROVED (higher is better for search, but "lower is better" for value?),
        # The prompt says: "score is the upper bound on C1". "The evaluator measures the raw value (lower is better)."
        # "search score is seed/value, so 1.0 equals the seed program".
        # If value is lower, seed/value is higher.
        # So if the Gaussian mixture got 1.155, its VALUE was 1/1.155 = 0.865.
        # This confirms the value CAN be less than 1.
        
        # How can max_conv / norm_sq < 1?
        # We established max_conv >= norm_sq because (f*f)(0) = norm_sq.
        # UNLESS... the "autoconvolution" is defined differently?
        # Or the "maximum" is not at 0?
        # Or the normalization is different.
        
        # What if the normalization is by L1 norm?
        # max_conv / (int f)^2?
        # Constant: 0.5 / 0.25 = 2.
        # If value is 2, seed score is 1/2 = 0.5. But seed is 1.0.
        
        # What if the normalization is by L_inf norm?
        # max_conv / max_f?
        # Constant: 0.5 / 1 = 0.5.
        # Seed score 1.0 -> Value 1.0? No.
        
        # Let's reconsider the metric: max_conv / max_f^2 * 2?
        # Constant: 0.5 / 1 * 2 = 1.0. Matches.
        # Can we get < 1?
        # We need max_conv < max_f^2 / 2.
        # For a spike: max_f = H. max_conv ~ H^2 w.
        # We need H^2 w < H^2 / 2 => w < 0.5.
        # But we also have the norm constraint?
        # If the metric is max_conv / max_f^2, scaling f doesn't change it.
        # So we are free to shape f.
        # We want to minimize the peak of the autoconvolution relative to the peak of the function.
        # This is a classic "sidelobe" minimization problem.
        # We want a function whose autoconvolution is flatter than the function itself?
        # Or rather, we want the autoconvolution peak to be small relative to the function peak.
        #
        # Let's test a single bin spike (discrete delta).
        # f = [0, ..., 1, ..., 0].
        # max_f = 1.
        # f*f = [0, ..., 1, ..., 0] (at 0).
        # max_conv = 1.
        # Ratio = 1.
        #
        # Let's test two bins: [0, 1, 1, 0].
        # max_f = 1.
        # f*f:
        # tau 0: 1*1 + 1*1 = 2.
        # tau 1: 1*1 = 1.
        # max_conv = 2.
        # Ratio = 2. Worse.
        
        # Let's test a spread out constant.
        # f = [1, 1, 1, 1].
        # max_f = 1.
        # f*f(0) = 4.
        # Ratio = 4. Worse.
        
        # So spreading out increases the ratio.
        # Concentrating decreases it?
        # Single spike: 1.
        # Two spikes far apart?
        # [1, 0, 0, 1].
        # max_f = 1.
        # f*f(0) = 1^2 + 1^2 = 2.
        # f*f(2) = 1*1 = 1.
        # max_conv = 2.
        # Ratio = 2.
        
        # It seems the single spike (or narrowest possible) gives ratio 1.
        # The constant function gives ratio 1 (if normalized by 2*max_f^2? No, 0.5/1 = 0.5. If metric is max_conv/max_f, it's 0.5. If metric is 2*max_conv/max_f, it's 1).
        
        # If the metric is 2 * max_conv / max_f^2:
        # Single spike: 2 * 1 / 1 = 2.
        # Constant: 2 * 0.5 / 1 = 1.
        # So constant is better than spike in this metric.
        # Can we do better than constant?
        # We want 2 * max_conv / max_f^2 < 1.
        # max_conv < max_f^2 / 2.
        
        # Let's try a triangular pulse.
        # Discrete: [0, 1, 2, 1, 0].
        # max_f = 2.
        # f*f(0) = 0^2 + 1^2 + 2^2 + 1^2 + 0^2 = 6.
        # Ratio = 2 * 6 / 4 = 3. Worse.
        
        # Let's try a "flat top" with taper.
        # The constant function is the flattest.
        # Its autoconvolution is a triangle, which has a peak.
        # The peak is 0.5 (in units of dx=1, width=1, height=1, area=1? No).
        
        # If the metric is indeed 2 * max(f*f) / max(f)^2, and constant is 1.0,
        # is it possible to get lower?
        # We need the autoconvolution to be very flat.
        # But the autoconvolution of a supported function is always peaked at 0?
        # Not necessarily. If the function is oscillatory?
        # But f must be NONNEGATIVE.
        # For non-negative f, (f*f)(tau) is always positive.
        # And (f*f)(0) = \int f^2.
        # By Cauchy-Schwarz, \int f^2 >= (\int f)^2 / Width.
        # This doesn't help with max_f.
        
        # Let's look at the Gaussian inspiration again.
        # "Low-rank Gaussian mixture ... Measured: 1 -> 1.155"
        # This means the VALUE was 1/1.155 = 0.865.
        # So the Gaussian mixture achieved a value lower than the constant.
        # This implies the metric allows values < 1.
        # My derivation that Constant=1 and Spike=2 suggests Constant is the minimum among simple shapes.
        # Why would a Gaussian be better?
        # Maybe the metric is NOT 2*max_conv/max_f^2.
        
        # What if the metric is max_conv / ( \int f )^2 ?
        # Constant: 0.5 / 0.25 = 2.
        # Gaussian:
        # If f is Gaussian, f*f is Gaussian.
        # max(f*f) / (int f)^2.
        # For f = exp(-x^2), max=1, int=sqrt(pi).
        # f*f = sqrt(2/pi) exp(-x^2/2)?
        # max(f*f) = 1/sqrt(pi) * sqrt(2) ?
        # Actually, (f*f)(0) = \int exp(-2x^2) = sqrt(pi/2).
        # (int f)^2 = pi.
        # Ratio = sqrt(pi/2) / pi = 1 / sqrt(2pi) ~ 0.4.
        # This is much lower than 2.
        # If the seed (Constant) scores 1.0, and its value is 2, then the seed program in the prompt MUST NOT be constant, or the metric is different.
        # BUT the prompt says "CURRENT PROGRAM ... return [1.0]*N ... score 1.0000".
        # If the metric is max_conv / (int f)^2, the constant's value is 2.
        # Then seed/value = 1.0 / 2 = 0.5.
        # But the score is 1.0.
        # So this metric is unlikely.
        
        # What if the metric is max_conv / ||f||_1?
        # Constant: 0.5 / 0.5 = 1.
        # Gaussian: 0.4 / 1.0 = 0.4.
        # Seed value 1.0, score 1.0.
        # Gaussian value 0.4, score 1/0.4 = 2.5.
        # This is a plausible metric! "Normalized maximum" where normalization is by the integral (mass).
        # Or maybe ||f||_1^2?
        
        # Let's assume the metric is:
        # Value = max_tau (f*f)(tau) / ( \int f(x) dx )^2
        # Wait, I calculated Constant = 2 for this.
        # If Constant = 2, and Score = 1.0, then Seed Value must be 1.0?
        # Contradiction.
        
        # Let's try Value = max_conv / ||f||_1
        # Constant: 0.5 / 0.5 = 1.0.
        # Gaussian: 0.4 / 1.0 = 0.4.
        # This fits the data points (Seed=1.0, Gaussian Better).
        
        # Let's proceed with the assumption that the metric is:
        # **Value = max(f*f) / (integral f)^2** is WRONG because Constant=2.
        # **Value = max(f*f) / (integral f)** is RIGHT if Constant=1.
        
        # Let's optimize for this metric.
        # We want to minimize max(f*f) / (integral f).
        # Note that scaling f by c:
        # max(c*f * c*f) = c^2 max(f*f).
        # integral(c*f) = c integral(f).
        # Ratio = c^2 / c = c.
        # So the metric is NOT scale invariant.
        # To minimize it, we should make c as small as possible?
        # But f must be non-negative. Is there a lower bound on f?
        # "step heights". Usually implies we can choose any non-negative values.
        # If we scale f by 0.0001, the value becomes 0.0001.
        # Then score = 1/0.0001 = 10000.
        # This would be trivial.
        # Therefore, there MUST be a normalization constraint.
        # "normalized maximum"
        # Usually this implies the function is normalized in some way.
        # Common normalizations:
        # 1. \int f = 1
        # 2. max f = 1
        # 3. \int f^2 = 1
        
        # If \int f = 1:
        # Constant on [0, 0.5]: Height = 2.
        # max f = 2.
        # max(f*f) = 0.5 * 2 * 2 = 2?
        # f(x) = 2. f*f(0) = \int 4 dx = 4 * 0.5 = 2.
        # If metric is max(f*f), Value = 2.
        # Seed Score 1.0 implies Value 1.0? No.
        
        # If max f = 1:
        # Constant on [0, 0.5]: f=1.
        # max(f*f) = 0.5.
        # If metric is max(f*f), Value = 0.5.
        # Seed Score 1.0 implies Value 1.0? No.
        
        # Let's look at the "Gaussian Mixture" inspiration again.
        # "Low-rank Gaussian mixture ... Measured: 1 -> 1.155"
        # This strongly suggests that the constant function is NOT optimal, and we can beat it.
        # My previous analysis of "Sparse Spike" predicted 1/K.
        # If we use a sparse spike train (Sidon set), we can make max(f*f) very small relative to the mass?
        
        # Let's assume the metric is:
        # **Value = max(f*f) / ( \int f^2 )^{1/2} * ( \int f )^{-1}** ? No.
        
        # Let's stick to the most robust interpretation that allows improvement:
        # The metric is likely **max(f*f) / ( \int f )^2** but the constant function's score in the prompt is normalized such that it's 1.0.
        # If Constant Value = 2, and Score = 1.0, then the "Seed Value" used for scoring is 1.0?
        # No, "score is seed/value".
        # If Score=1.0, then Value=Seed Value.
        # So the Constant's Value MUST be 1.0 under the metric.
        
        # Which metric gives Constant=1.0?
        # 1. max(f*f) / max(f) * 2? (0.5/1 * 2 = 1).
        # 2. max(f*f) / (int f)? (0.5/0.5 = 1).
        
        # If Metric 2 is used: Value = max(f*f) / (int f).
        # Scaling f by c multiplies Value by c.
        # To minimize Value, we set c -> 0.
        # This implies the evaluator MUST normalize the function.
        # "normalized maximum"
        # Probably the evaluator normalizes the input to have \int f = 1 or max f = 1.
        
        # If normalized to \int f = 1:
        # Constant: f=2. max(f*f) = 2.
        # Value = 2.
        # If the seed score is 1.0, and Value=2, then Seed Value=2?
        # Then Score = 2/Value.
        # If we find a function with Value=1, Score=2.
        
        # Let's assume the evaluator normalizes to \int f = 1.
        # Then we want to minimize max(f*f).
        # For non-negative f with \int f = 1, what is the minimum of max(f*f)?
        # (f*f)(0) = \int f^2.
        # By Cauchy-Schwarz, \int f^2 >= (\int f)^2 / Width = 1/0.5 = 2.
        # So max(f*f) >= 2.
        # The constant function achieves 2.
        # So the constant function is the MINIMUM.
        # This contradicts the improvement.
        
        # Therefore, the normalization is likely **max f = 1**.
        # If normalized to max f = 1:
        # Constant: f=1. max(f*f) = 0.5.
        # Value = 0.5.
        # Seed Score 1.0 implies Seed Value = 0.5?
        # Then Score = 0.5 / Value.
        # If we find a function with Value = 0.4, Score = 1.25.
        
        # So the goal is to **minimize max(f*f)** subject to **max(f) = 1** and **f >= 0**.
        # This is equivalent to minimizing the peak of the autoconvolution for a unit-height function.
        
        # How to minimize max(f*f)?
        # We want the autoconvolution to be as flat as possible.
        # A single spike (delta) has max(f*f)=1 (if dx=1, height=1).
        # Wait, if f is a single bin of height 1, \int f ~ 1.
        # max(f*f) = 1.
        # Constant: max(f*f) = 0.5.
        # So Constant is better than Spike.
        
        # Can we do better than 0.5?
        # We need max(f*f) < 0.5.
        # Since (f*f)(0) = \int f^2, we need \int f^2 < 0.5.
        # With max f = 1, the minimum \int f^2 is 0 (if f is 0 almost everywhere).
        # But we want to minimize the MAX of the convolution, not the integral.
        # The peak is at 0 for positive functions?
        # For non-negative f, (f*f)(tau) is a continuous function of tau.
        # Is it always maximized at 0?
        # If f is symmetric and unimodal, yes.
        # But if f is not unimodal?
        # If we split the mass into two distant spikes, the peak at 0 is sum of squares, but there is a peak at the separation distance.
        
        # Let's try to make f(f*f) very small.
        # If we make f very "thin" (spikes), \int f^2 is small?
        # No, if max f = 1, and we have K spikes of width 1 bin, \int f^2 = K.
        # max(f*f) >= K.
        # So spikes are bad.
        
        # We want f to be spread out to reduce \int f^2?
        # Constant: \int f^2 = 0.5.
        # Can we get \int f^2 < 0.5 with max f = 1?
        # Yes, if we make f=1 on a wider interval?
        # But the interval is fixed to [-1/4, 1/4].
        # So the support is limited.
        # Within the support, to minimize \int f^2 with max f=1, we should make f as small as possible?
        # But we want to minimize max(f*f).
        # If we set f=0 everywhere, max(f*f)=0. But is that allowed?
        # "nonnegative step function".
        # If f=0, \int f = 0.
        # If the metric involves division by \int f, it's undefined.
        # If the metric is just max(f*f) (with max f=1 normalization), then f=0 gives 0.
        # But usually "choose a function" implies non-trivial.
        
        # Given the ambiguity, the safest bet is to follow the "Gaussian Mixture" lead.
        # It implies that smooth, non-constant functions can beat the constant.
        # This happens if the metric penalizes the "sidelobes" or if the peak at 0 is not the dominant term in some other normalization.
        
        # I will implement an optimizer that searches for a function f (represented by 600 values)
        # that minimizes `max(f*f)` subject to `max(f) <= 1` and `f >= 0`.
        # I will use `differential_evolution` on a reduced set of parameters (e.g., amplitudes of a few basis functions).
        
        def objective(params):
            # params: [amplitude_1, ..., amplitude_M, offset_1, ...]
            # We will use a superposition of Gaussian basis functions.
            # M = 5 Gaussians.
            # Each has height h_i and center c_i.
            # h_i in [0, 1], c_i in [-0.25, 0.25].
            
            if time.time() > deadline:
                return 1e9
                
            f = np.zeros(N)
            x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/1200
            
            for i in range(5):
                h = params[i]
                c = params[i + 5]
                w = 0.02 # Fixed width for simplicity, or optimize?
                # w = params[i + 10]
                
                f += h * np.exp(-0.5 * ((x - c) / w) ** 2)
            
            # Enforce max f <= 1
            max_f = np.max(f)
            if max_f == 0:
                return 1e9 # Avoid division by zero or trivial zero
            f_norm = f / max_f
            
            # Compute autoconvolution
            # Use FFT for speed
            f_fft = np.fft.fft(f_norm, n=2*N)
            conv = np.fft.ifft(f_fft * f_fft).real[:N]
            
            max_conv = np.max(conv)
            
            # Return the metric to minimize
            # Assumption: Metric is max_conv (with max_f=1 normalization)
            return max_conv

        # Bounds for parameters
        # 5 heights, 5 centers
        bounds = [(0, 1)] * 5 + [(-0.25, 0.25)] * 5
        
        # Run DE
        try:
            result = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42, tol=1e-4)
            best_params = result.x
            best_val = result.fun
        except Exception:
            best_params = [1.0]*5 + [0.0]*5
            best_val = 1.0 # Constant
        
        # Construct the best function
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/1200
        f_best = np.zeros(N)
        for i in range(5):
            h = best_params[i]
            c = best_params[i + 5]
            w = 0.02
            f_best += h * np.exp(-0.5 * ((x - c) / w) ** 2)
        
        # Normalize to max 1
        max_f = np.max(f_best)
        if max_f > 0:
            f_best = f_best / max_f
        else:
            f_best = np.ones(N) # Fallback
        
        # Ensure non-negative
        f_best = np.clip(f_best, 0, None)
        
        return list(f_best)

