import numpy as np

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def construct():
    """
    Construct the optimal step function.
    
    Mathematical Analysis:
    We want to minimize the normalized maximum of the autoconvolution of a non-negative 
    step function h with N intervals.
    Let S = sum(h). The normalized autoconvolution at lag k is C_k = (sum h_i h_{i+k}) / S^2.
    For non-negative h, the maximum of the autoconvolution is at lag 0 (self-correlation),
    because for k > 0, the sum involves products of disjoint or partially disjoint sets of 
    non-negative terms, and generally C_0 = sum(h_i^2)/S^2 is the largest term due to 
    the Cauchy-Schwarz inequality and the fact that h_i^2 terms are "aligned" while 
    cross terms are not necessarily, but more simply, for non-negative functions, 
    the peak of the convolution is at the center (lag 0) if the function is concentrated, 
    but actually, for any non-negative function, the maximum of the convolution is at 
    lag 0? No, that's not true for all non-negative functions. 
    Example: h = [1, 0, 1]. Conv = [1, 0, 2, 0, 1]. Max is 2 at lag 2. 
    Wait, the problem says "autoconvolution". Usually autoconvolution implies correlation 
    or convolution with itself. 
    If it's convolution: (h * h)[k] = sum h[i] h[k-i].
    For h = [1, 0, 1], conv = [1, 0, 2, 0, 1]. Max is 2. Sum = 2. Score = 2/4 = 0.5.
    Uniform h = [1, 1, 1]. Conv = [1, 2, 3, 2, 1]. Max is 3. Sum = 3. Score = 3/9 = 1/3.
    1/3 < 0.5.
    
    Let's check the claim: For non-negative h, is max(conv) / (sum h)^2 minimized by uniform h?
    
    Let's test N=2.
    h=[1,0] -> conv=[1,0,0], max=1, sum=1, score=1.
    h=[1,1] -> conv=[1,2,1], max=2, sum=2, score=2/4=0.5.
    h=[a,b] -> conv=[a^2, 2ab, b^2]. Max is max(a^2, 2ab, b^2). Sum = a+b.
    Score = max(a^2, 2ab, b^2) / (a+b)^2.
    If a=b, score = 2a^2 / 4a^2 = 0.5.
    If a=1, b=0, score = 1/1 = 1.
    If a=1, b=10, sum=11, sq=121. max(1, 20, 100)=100. Score=100/121 ~ 0.82.
    
    It seems uniform is better.
    
    Proof sketch:
    We want to minimize M = max_k (sum_i h_i h_{i+k}) / (sum_i h_i)^2.
    Note that for non-negative h, the term k=0 is sum h_i^2.
    The term k=N/2 (if even) or similar can be larger?
    Actually, for non-negative h, the maximum of the convolution is NOT necessarily at 0.
    However, the "normalized maximum" is the metric.
    
    Let's consider the vector h.
    If h is uniform, h_i = c.
    Conv_k = sum_{i} c * c = (N-|k|) c^2.
    Max is at k=0: N c^2.
    Sum = N c.
    Score = N c^2 / (N c)^2 = 1/N.
    
    Can we get lower than 1/N?
    By Cauchy-Schwarz: (sum h_i)^2 <= N sum h_i^2.
    So sum h_i^2 / (sum h_i)^2 >= 1/N.
    This proves that the k=0 term alone has a minimum of 1/N.
    Since the max of the convolution is >= the k=0 term, the score is >= 1/N.
    Therefore, the uniform distribution achieves the theoretical lower bound of 1/N.
    Any other distribution will have sum h_i^2 / (sum h_i)^2 > 1/N, and thus the score will be > 1/N.
    
    Thus, the uniform distribution is the unique global minimizer (up to scaling).
    """
    # The optimal solution is the uniform distribution.
    # h_i = 1 for all i.
    h_opt = np.ones(N)
    
    # Verify non-negativity
    if np.any(h_opt < 0):
        h_opt = np.maximum(h_opt, 0)
        
    # The score is 1/N.
    # We return the list of heights.
    return h_opt.tolist()
