"""
Symbolic regression using a Gaussian RBF basis with learnable centers, widths, and weights.
This structural change moves from a sigmoid+polynomial representation to a local-basis
representation (RBFs) that can model arbitrary smooth functions on [-2, 2].
Five Gaussian kernels with learnable centers, widths, and amplitudes, plus a linear term.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = p[8] + p[9]*x + sum_{k=0}^{4} p[3k] * exp(-0.5 * ((x - p[3k+1]) / p[3k+2])^2)
    
    Parameters (10 total):
        p[0], p[1], p[2]: amplitude, center, width for RBF 0
        p[3], p[4], p[5]: amplitude, center, width for RBF 1
        p[6], p[7], p[8]: amplitude, center, width for RBF 2
        p[9]: linear coefficient
    
    Wait, that's only 10 params but I need 5 RBFs (15 params) + linear. Let me restructure.
    
    Actually with 10 params:
        p[0]: constant offset
        p[1]: linear coefficient
        p[2], p[3], p[4]: amplitude, center, width for RBF 0
        p[5], p[6], p[7]: amplitude, center, width for RBF 1
        p[8], p[9]: amplitude, center for RBF 2 (fixed width)
    
    Let me use a cleaner design: 3 RBFs with learnable center and amplitude, fixed width,
    plus constant and linear. That's 2 + 3*2 = 8 params. I have 10, so let me use
    4 RBFs with learnable center and amplitude (8 params) + constant + linear = 10.
    
    Model: y(x) = p[0] + p[1]*x + sum_{k=0}^{3} p[2+2k] * exp(-((x - p[3+2k])^2) / (2*sigma^2))
    where sigma is fixed at 0.5.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.ndim != 1:
        p = p.flatten()
    
    # Ensure we have at least 10 params
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), constant_values=1.0)
    
    p0 = p[0]  # constant
    p1 = p[1]  # linear
    
    # 4 RBFs with centers p[3], p[5], p[7], p[9] and amplitudes p[2], p[4], p[6], p[8]
    sigma = 0.5  # fixed width
    
    centers = np.array([p[3], p[5], p[7], p[9]])
    amplitudes = np.array([p[2], p[4], p[6], p[8]])
    
    # Compute RBF contributions
    # x shape: (N,), centers shape: (4,)
    # diff shape: (N, 4)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum of RBFs
    rbf_sum = rbf @ amplitudes
    
    return p0 + p1 * x + rbf_sum
# EVOLVE-BLOCK-END
