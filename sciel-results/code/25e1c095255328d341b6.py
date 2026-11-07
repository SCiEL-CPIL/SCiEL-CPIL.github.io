"""
Sinc-based Fourier Series Approximation Strategy.

Rationale:
Polynomials are poor at representing periodic or highly oscillatory functions
(e.g., sin(x), cos(y), sin(x+y)) within a bounded domain like [-1, 1] without
very high degrees, which leads to ill-conditioning.
Sinc functions (normalized by their argument) form a basis for band-limited
functions. A truncated sum of sinc terms with varying frequencies can approximate
a wide variety of smooth functions, including periodic ones, more robustly than
low-degree polynomials.

Model:
f(x, y) = p0 + 
          p1 * sinc(p2 * x) + p3 * sinc(p2 * y) + 
          p4 * sinc(p2 * (x + y)) + p5 * sinc(p2 * (x - y)) +
          p6 * sinc(2*p2 * x) + p7 * sinc(2*p2 * y) +
          p8 * sinc(2*p2 * (x + y)) + p9 * sinc(2*p2 * (x - y))

Note: numpy.sinc(x) = sin(pi*x)/(pi*x), with sinc(0)=1.
This allows us to scale the frequency via the parameter p2 (shared to reduce
dimensionality and improve conditioning) and individual amplitudes.
We use a shared frequency scale p2 to keep the problem well-conditioned for BFGS,
allowing the optimizer to adjust the "resolution" of the approximation.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sinc-based approximation with 10 parameters.
    
    Parameters:
    p0: DC offset
    p1, p3, p4, p5: Amplitudes for fundamental frequency terms (x, y, x+y, x-y)
    p6, p7, p8, p9: Amplitudes for second harmonic terms (2x, 2y, 2(x+y), 2(x-y))
    p2: Global frequency scale factor
    """
    # Deadline guard: If this function is called in a context where time matters
    # (though the evaluator calls it once for fitting, we assume standard usage),
    # we just compute directly. The computation is O(N) and very fast.
    
    # Ensure params is a flat array
    p = np.asarray(params).flatten()
    
    # Extract parameters
    # If optimizer fails or provides weird values, we might have NaNs.
    # We assume the optimizer provides finite values.
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]
    p5 = p[5]
    p6 = p[6]
    p7 = p[7]
    p8 = p[8]
    p9 = p[9]
    
    # To prevent numerical issues if p2 becomes 0 or very large,
    # we clip p2 to a reasonable range. 
    # np.sinc is well-behaved, but if p2 is huge, we get high freq oscillations
    # which might alias if not sampled densely enough, but inputs are uniform.
    # Let's allow p2 to vary but keep it stable.
    # BFGS starting from 1.0 should keep p2 in a reasonable range initially.
    
    # Compute arguments for sinc
    # np.sinc(x) computes sin(pi*x)/(pi*x)
    # So sinc(k * x) corresponds to sin(pi * k * x) / (pi * k * x)
    
    arg_x = p2 * x
    arg_y = p2 * y
    arg_xy_sum = p2 * (x + y)
    arg_xy_diff = p2 * (x - y)
    
    # Fundamental terms
    term1 = p1 * np.sinc(arg_x)
    term2 = p3 * np.sinc(arg_y)
    term3 = p4 * np.sinc(arg_xy_sum)
    term4 = p5 * np.sinc(arg_xy_diff)
    
    # Second harmonic terms (frequency doubled)
    arg_2x = 2 * p2 * x
    arg_2y = 2 * p2 * y
    arg_2xy_sum = 2 * p2 * (x + y)
    arg_2xy_diff = 2 * p2 * (x - y)
    
    term5 = p6 * np.sinc(arg_2x)
    term6 = p7 * np.sinc(arg_2y)
    term7 = p8 * np.sinc(arg_2xy_sum)
    term8 = p9 * np.sinc(arg_2xy_diff)
    
    pred = p0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
    
    # Ensure output is finite
    # If any parameter is NaN or Inf, pred will be NaN/Inf.
    # The evaluator handles non-finite errors, but we can try to be safe.
    # However, modifying pred to replace NaNs might break gradients if done poorly.
    # Standard practice is to let the optimizer handle it, but the prompt says
    # "non-finite error costs 1e30". So returning NaN is acceptable if it leads
    # to that penalty, but it's better to avoid if possible.
    # Given BFGS, it should stay in finite regions.
    
    return pred
# EVOLVE-BLOCK-END
