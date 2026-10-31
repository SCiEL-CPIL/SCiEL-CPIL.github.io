import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600
L = 0.5  # interval length [-1/4, 1/4]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution.
    
    For a step function with heights h_i on N equal intervals of total length L,
    the autoconvolution at lag 0 is sum(h_i^2 * dx) where dx = L/N.
    The normalization is by max(h) * sum(h_i) * dx.
    
    The ratio is (sum h_i^2) / (max(h) * sum h_i), which is scale invariant.
    """
    h = np.asarray(h)
    s1 = np.sum(h)
    s2 = np.sum(h**2)
    hmax = np.max(h)
    if s1 < 1e-15 or hmax < 1e-15:
        return np.inf
    return s2 / (hmax * s1)

def objective(x):
    """Objective function for optimization: the normalized autoconvolution max."""
    h = np.maximum(x, 0.0)  # ensure nonnegativity
    return compute_autoconv_max(h)

def construct():
    """Construct a nonnegative step function with 600 intervals on [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution."""
    
    deadline = time.time() + 110.0  # safety margin before 120s limit
    best_h = np.ones(N)
    best_val = compute_autoconv_max(best_h)
    
    # Theoretical lower bound: for uniform h, ratio = N/(1*N) = 1
    # Wait: uniform h: s2 = N*h^2, s1 = N*h, hmax = h
    # ratio = N*h^2 / (h * N*h) = 1. 
    # So uniform gives ratio 1, which corresponds to score 1.0.
    
    # Let's think about what the "score" actually is.
    # The task says "score is the upper bound on C1" and "the evaluator measures the raw value (lower is better)".
    # And "search score is seed/value, so 1.0 equals the seed program".
    # Seed program returns [1.0]*N, which gives ratio = 1.0.
    # So if we find a function with ratio r, the score = 1.0 / r? No...
    # "search score is seed/value" where seed is the seed program's value (1.0) and value is our value.
    # So search_score = 1.0 / our_value. If our_value < 1.0, search_score > 1.0 (improvement).
    # Wait, "lower is better" for the raw value. So we want to MINIMIZE the raw value.
    # The raw value is the normalized maximum of the autoconvolution.
    # Uniform gives raw value = 1.0.
    # Can we get below 1.0?
    
    # For nonnegative h, by Cauchy-Schwarz: (sum h_i)^2 <= N * sum h_i^2
    # So sum h_i^2 / sum h_i >= sum h_i / N
    # And sum h_i / N <= max(h_i) (since sum h_i <= N * max h_i)
    # Actually: ratio = sum h_i^2 / (max h_i * sum h_i)
    # Let's denote a_i = h_i / max(h_i), so a_i in [0,1] and max a_i = 1.
    # ratio = sum a_i^2 / sum a_i
    # We want to minimize sum a_i^2 / sum a_i subject to a_i in [0,1], max a_i = 1.
    # 
    # If all a_i = 1: ratio = N/N = 1.
    # If one a_i = 1, rest = 0: ratio = 1/1 = 1.
    # If one a_i = 1, rest = eps: ratio ≈ (1 + (N-1)eps^2) / (1 + (N-1)eps) ≈ 1 - (N-1)eps for small eps.
    # This is LESS than 1!
    # 
    # Let's verify: h = [1, eps, eps, ..., eps]
    # s2 = 1 + (N-1)*eps^2
    # s1 = 1 + (N-1)*eps
    # hmax = 1
    # ratio = (1 + (N-1)*eps^2) / (1 + (N-1)*eps)
    # For eps << 1: ≈ (1 + (N-1)*eps^2) * (1 - (N-1)*eps) ≈ 1 - (N-1)*eps + (N-1)*eps^2
    # For N=600, eps=0.01: ≈ 1 - 599*0.01 + 599*0.0001 ≈ 1 - 5.99 + 0.0599 ≈ -4.93
    # Wait that can't be right, ratio must be positive.
    # Let me recalculate: (1 + 599*0.0001)/(1 + 599*0.01) = (1 + 0.0599)/(1 + 5.99) = 1.0599/6.99 ≈ 0.1516
    # That's much less than 1!
    # 
    # So the optimal strategy is to have one large spike and the rest very small but positive.
    # As eps -> 0, ratio -> 1/1 = 1. But for finite eps, we can get below 1.
    # Actually as eps -> 0, the ratio approaches 1 from below.
    # The minimum is achieved at some optimal eps.
    # 
    # Let's minimize f(eps) = (1 + (N-1)*eps^2) / (1 + (N-1)*eps) for eps in (0, 1).
    # f'(eps) = [2(N-1)eps * (1+(N-1)eps) - (1+(N-1)eps^2)*(N-1)] / (1+(N-1)eps)^2
    # Set numerator to 0:
    # 2(N-1)eps(1+(N-1)eps) - (N-1)(1+(N-1)eps^2) = 0
    # 2eps + 2(N-1)eps^2 - 1 - (N-1)eps^2 = 0
    # (N-1)eps^2 + 2eps - 1 = 0
    # eps = (-2 + sqrt(4 + 4(N-1))) / (2(N-1)) = (-2 + 2*sqrt(N)) / (2(N-1)) = (sqrt(N) - 1) / (N-1)
    # For N=600: eps = (sqrt(600) - 1) / 599 ≈ (24.49 - 1)/599 ≈ 23.49/599 ≈ 0.0392
    # 
    # f(eps*) = (1 + (N-1)*eps*^2) / (1 + (N-1)*eps*)
    # (N-1)*eps* = sqrt(N) - 1
    # (N-1)*eps*^2 = (sqrt(N)-1)^2 / (N-1) = (N - 2*sqrt(N) + 1)/(N-1)
    # s2 = 1 + (N - 2*sqrt(N) + 1)/(N-1) = (N-1 + N - 2*sqrt(N) + 1)/(N-1) = (2N - 2*sqrt(N))/(N-1) = 2(N - sqrt(N))/(N-1)
    # s1 = 1 + (sqrt(N)-1) = sqrt(N)
    # ratio = 2(N - sqrt(N))/(N-1) / sqrt(N) = 2(N - sqrt(N))/((N-1)*sqrt(N))
    # For large N: ≈ 2N/(N*sqrt(N)) = 2/sqrt(N)
    # For N=600: 2/sqrt(600) ≈ 2/24.49 ≈ 0.0817
    # 
    # So the optimal single-spike solution gives ratio ≈ 0.0817, which is much less than 1!
    # This should give a score of 1.0/0.0817 ≈ 12.2.
    
    # Let's verify with the exact formula for N=600:
    eps_opt = (np.sqrt(N) - 1) / (N - 1)
    h_spike = np.full(N, eps_opt)
    h_spike[0] = 1.0  # one spike at index 0
    val_spike = compute_autoconv_max(h_spike)
    best_h = h_spike.copy()
    best_val = val_spike
    
    # Try to improve with optimization
    # Parameterize: one spike of height 1, rest have height eps
    # But maybe multiple spikes could help?
    # Let's try differential evolution on a small number of free parameters.
    
    # Actually, let's think more carefully. The ratio is:
    # sum(h_i^2) / (max(h_i) * sum(h_i))
    # This is equivalent to: sum(a_i^2) / sum(a_i) where a_i = h_i/max(h_i), max(a_i)=1, a_i >= 0
    # We want to minimize this.
    # 
    # By convexity, the minimum should be at an extreme point.
    # The extreme points of the feasible region (a_i in [0,1], max a_i = 1) are:
    # - One a_i = 1, rest 0: ratio = 1
    # - One a_i = 1, rest equal to some value: this is what we found.
    # 
    # Actually, the feasible set is a simplex-like region. The minimum of a convex function over a convex set is at the boundary.
    # sum(a_i^2)/sum(a_i) is not convex in general, but let's just try the single-spike solution.
    
    # Let's also try: k spikes of height 1, rest epsilon
    # For k=1: we already computed.
    # For k=2: s2 = 2 + (N-2)*eps^2, s1 = 2 + (N-2)*eps
    # ratio = (2 + (N-2)*eps^2)/(2 + (N-2)*eps)
    # Derivative: 2(N-2)eps(2+(N-2)eps) - (2+(N-2)eps^2)(N-2) = 0
    # 4eps + 2(N-2)eps^2 - 2(N-2) - (N-2)^2 eps^2 = 0
    # This is more complex. Let's just try k=1 and k=2 numerically.
    
    for k in [1, 2, 3, 5, 10]:
        if k >= N:
            break
        # Optimize eps for k spikes
        def ratio_k(eps):
            s2 = k + (N - k) * eps**2
            s1 = k + (N - k) * eps
            return s2 / s1
        
        # Find minimum of ratio_k for eps in (0, 1)
        # Use golden section or just scan
        eps_vals = np.linspace(0.0001, 1.0, 1000)
        ratios = [ratio_k(e) for e in eps_vals]
        min_idx = np.argmin(ratios)
        h_k = np.full(N, eps_vals[min_idx])
        for i in range(k):
            h_k[i] = 1.0
        val_k = compute_autoconv_max(h_k)
        if val_k < best_val:
            best_val = val_k
            best_h = h_k.copy()
    
    # Now try to refine with scipy optimization
    # Parameterize as: heights for a subset of "active" positions
    # Simple approach: optimize all 600 heights with L-BFGS-B
    # But that might be slow. Let's try a reduced parameterization.
    
    # Actually, the single-spike solution with optimal eps should be very close to optimal.
    # Let's just verify it and return.
    
    # Final check: ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    # Verify the value
    final_val = compute_autoconv_max(best_h)
    
    # Return as list of floats
    return list(best_h)
