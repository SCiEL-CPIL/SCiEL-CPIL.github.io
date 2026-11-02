import numpy as np
import time
import warnings
from itertools import combinations

N = 50

def _score(h):
    """Compute the score: L2^2 / (L1 * max_k conv_k) for nonnegative h.
    Here conv is the discrete autoconvolution (full mode).
    Note: for nonnegative h, conv[0] = sum h_i^2 = L2^2, which is the max.
    So score = L2^2 / (L1 * L2^2) = 1/L1.
    Wait, that can't be right if scores > 1 are observed.
    
    Let me re-read the task: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    This is ambiguous. It could be:
      (L2^2) / (L1 * ||conv||_inf)
    OR
      (L2^2) / (L1 * ||conv||_inf) where conv is the CONTINUOUS autoconvolution
    
    But for discrete h on 50 bins, the autoconvolution is discrete.
    
    Actually, looking at the inspirations:
    - inspiration 1: ratio = ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)
      This has an extra ||h||_inf in the denominator!
    - inspiration 3: parent score is 1.1944 for [2/3, 1/3, 0...]
    
    Let's check: h = [2/3, 1/3, 0, ...]
    L2^2 = (4/9 + 1/9) = 5/9
    L1 = 1
    h*h (full conv) = [4/9, 4/9, 1/9, 0, ...]
    max conv = 4/9
    ||h||_inf = 2/3
    
    If score = L2^2 / (L1 * ||h||_inf * max_conv) = (5/9) / (1 * 2/3 * 4/9) = (5/9) / (8/27) = (5/9)*(27/8) = 15/8 = 1.875
    
    That doesn't match 1.1944 either.
    
    Let me try: score = L2^2 / (L1 * max_conv) = (5/9) / (1 * 4/9) = 5/4 = 1.25
    
    Still not 1.1944.
    
    What if the convolution is defined differently? Maybe centered?
    Or maybe the score is L2^2 / (L1 * max_k conv_k) where conv is circular?
    
    For circular conv of length 50:
    h = [2/3, 1/3, 0, ..., 0]
    circular conv[0] = h[0]*h[0] + h[1]*h[1] = 4/9 + 1/9 = 5/9
    circular conv[1] = h[0]*h[1] + h[1]*h[0] = 2*(2/3)*(1/3) = 4/9
    So max is still 5/9.
    Score = (5/9)/(1*5/9) = 1.
    
    Hmm. Let me try another interpretation.
    
    What if "squared L2 norm" means (sum h_i)^2? No, that's L1^2.
    
    Let me reconsider. The parent program returns [2/3, 1/3, 0...] and scores 1.1944.
    1.1944 = 107.5/90 = 215/180 = 43/36 ≈ 1.1944
    
    What combination gives 43/36?
    
    L2^2 = 5/9
    L1 = 1
    max_conv (full) = 4/9
    
    (5/9) / (4/9) = 5/4 = 1.25
    (5/9) / (1 * 4/9) = 1.25
    (5/9) / (1 * 2/3 * 4/9) = 15/8 = 1.875
    
    None of these give 43/36.
    
    Let me try: what if the heights are not normalized to sum=1?
    The parent normalizes to L1=1. What if we don't?
    
    Actually, the score might be scale-dependent. Let's say h = [a, b, 0, ...]
    L2^2 = a^2 + b^2
    L1 = a + b
    conv (full) = [a^2, 2ab, b^2]
    max_conv = a^2 (assuming a >= b)
    
    Score = (a^2+b^2) / ((a+b) * a^2)
    
    For a=2/3, b=1/3: (4/9+1/9)/((1)*(4/9)) = (5/9)/(4/9) = 5/4 = 1.25
    
    Still not 1.1944.
    
    Let me try to see what value of a/b gives 43/36:
    Let r = a/b. Score = (r^2+1)/(r(r^2)) = (r^2+1)/r^3
    (r^2+1)/r^3 = 43/36
    36(r^2+1) = 43r^3
    43r^3 - 36r^2 - 36 = 0
    
    Let me solve this numerically... r≈1.5 gives 43*3.375-36*2.25-36 = 145.125-81-36=28.125>0
    r=1.3: 43*2.197-36*1.69-36 = 94.471-60.84-36=-2.369<0
    r=1.35: 43*2.4596-36*1.8225-36 = 105.76-65.61-36=4.15>0
    r=1.32: 43*2.299-36*1.7424-36 = 98.86-62.73-36=0.13>0
    r≈1.319
    
    So a/b ≈ 1.319, i.e., a≈0.569, b≈0.431. Not 2/3, 1/3.
    
    I'm confused. Let me just implement a general optimizer and see what happens.
    The key insight from the task is that we need to maximize some ratio involving L2, L1, and autoconvolution.
    
    Given the ambiguity, let me try a different structural approach:
    Use a genetic algorithm or simulated annealing to directly optimize the score.
    
    But first, let me define the score function based on the most likely interpretation:
    score = L2^2 / (L1 * max_k conv_k)
    
    For nonnegative h, max_k conv_k = conv[0] = L2^2 (in full convolution, index 0 corresponds to lag 0... wait, no).
    
    Actually, np.convolve(h, h, 'full') for h of length N gives length 2N-1.
    The element at index N-1 is the "zero lag" correlation = sum h_i^2.
    So max is at index N-1 for nonnegative h.
    
    Thus score = L2^2 / (L1 * L2^2) = 1/L1.
    
    This means to maximize score, we minimize L1. But heights are nonnegative, so we can make them arbitrarily small, giving infinite score. That can't be right.
    
    There must be a normalization constraint. The most natural is L1 = 1 or L2 = 1 or max(h) = 1.
    
    If L1 = 1, score = 1/L1 = 1. Always 1. That matches the seed score of 1.0.
    
    But inspirations show scores > 1. So the score formula must be different.
    
    Let me look at inspiration 1 more carefully:
    "ratio: ||h||_2^2 / (||h||_1 ||h||_inf ||h*h||_inf)"
    
    With L1=1 normalization:
    score = L2^2 / (1 * ||h||_inf * max_conv)
    
    For h=[2/3, 1/3, 0...]:
    L2^2 = 5/9
    ||h||_inf = 2/3
    max_conv = 4/9 (full conv)
    score = (5/9) / (2/3 * 4/9) = (5/9) / (8/27) = 15/8 = 1.875
    
    Still not 1.1944.
    
    I'll just implement a robust optimizer that tries to maximize a reasonable score function and see what happens. The key structural change: use a different representation (sparse support + continuous heights) and a different optimization method (simulated annealing with random restarts).
"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0 or l2_sq == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    best_score = 0.0
    best_h = np.zeros(N)
    best_h[0] = 1.0
    
    # Strategy: Try sparse vectors with few non-zero elements
    # The score for a vector with support S and heights h_i is:
    # score = (sum_{i in S} h_i^2) / ((sum_{i in S} h_i) * max_k conv_k)
    
    # For a single non-zero element: score = h^2 / (h * h^2) = 1/h. 
    # But if we normalize L1=1, h=1, score=1.
    
    # Let's try a different approach: optimize over a small number of non-zero elements
    # using a random search / hill climbing hybrid
    
    def evaluate(h):
        return _score(h)
    
    # Try all single-element vectors (normalized)
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        s = evaluate(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Try pairs of non-zero elements
    for i in range(N):
        for j in range(i+1, N):
            # Optimize the ratio of heights
            # h = [0,...,a,...,b,...,0]
            # Try various ratios
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]:
                a = 1.0 / (1.0 + ratio)
                b = ratio / (1.0 + ratio)
                h = np.zeros(N)
                h[i] = a
                h[j] = b
                s = evaluate(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
    
    if time.time() > deadline - 30:
        return list(best_h)
    
    # Try triples
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Try equal heights
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                s = evaluate(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
    
    if time.time() > deadline - 20:
        return list(best_h)
    
    # Random search with local improvement
    np.random.seed(42)
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
        
        # Random sparse vector
        n_nonzero = np.random.randint(1, 8)
        indices = np.random.choice(N, n_nonzero, replace=False)
        h = np.zeros(N)
        weights = np.random.dirichlet(np.ones(n_nonzero))
        h[indices] = weights
        
        s = evaluate(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # Local improvement: perturb the best vector
        if np.random.random() < 0.3:
            h = best_h.copy()
            # Add a small random perturbation
            idx = np.random.randint(N)
            h[idx] += np.random.uniform(-0.1, 0.1)
            h[idx] = max(0, h[idx])
            if np.sum(h) > 0:
                h = h / np.sum(h)
                s = evaluate(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
    
    # Final check
    final_score = evaluate(best_h)
    if final_score > best_score:
        best_score = final_score
    
    return list(best_h)
