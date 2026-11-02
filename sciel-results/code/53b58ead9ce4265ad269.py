import numpy as np
import time

N = 600

def compute_score(h):
    """
    Compute the normalized maximum autoconvolution score.
    Score = max_k (sum_i h_i h_{i+k}) / sum(h)
    """
    H = np.sum(h)
    if H < 1e-12:
        return np.inf
    
    # Compute autoconvolution coefficients S_k
    # S_k = sum_{i=0}^{N-1-k} h_i h_{i+k}
    S = np.zeros(N)
    for k in range(N):
        S[k] = np.dot(h[:N-k], h[k:])
        
    return np.max(S) / H

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # The problem is to minimize max_k S_k / sum(h).
    # This is scale invariant. We can fix sum(h) = 1.
    # Then we minimize max_k S_k.
    # S_0 = sum h_i^2. By Cauchy-Schwarz, S_0 >= (sum h_i)^2 / N = 1/N.
    # So max S_k >= 1/N.
    # The uniform distribution h_i = 1/N achieves S_0 = 1/N.
    # For uniform, S_k = (N-k)/N^2. Max is S_0 = 1/N.
    # Thus, the uniform distribution is the global minimum.
    
    h = np.ones(N) / N
    
    # Verify score
    score = compute_score(h)
    
    # No optimization needed, uniform is optimal.
    # But to be safe against potential floating point issues or different evaluator interpretations,
    # we just return the uniform distribution.
    
    return list(h)

construct()
