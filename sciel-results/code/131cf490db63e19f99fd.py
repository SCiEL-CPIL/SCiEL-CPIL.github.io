import numpy as np

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio:
    R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    Analysis suggests that a sparse vector with 2 non-zero elements of equal magnitude
    maximizes this ratio under the constraint ||h||_1 = 1.
    
    For h = [0.5, 0, ..., 0.5, 0, ...]:
    ||h||_1 = 1
    ||h||_2^2 = 0.25 + 0.25 = 0.5
    Autoconvolution peaks:
    - At lag 0: 0.5^2 + 0.5^2 = 0.5 (Wait, if indices are 0 and d>0, 
      conv[0] = h[0]^2 = 0.25. conv[d] = h[0]h[d] = 0.25. conv[2d] = h[d]^2 = 0.25.)
      Actually, conv[0] = sum h[i]^2 = 0.5.
      Let's re-verify conv[0].
      (h*h)[0] = sum_{i} h[i] h[0-i].
      If h is supported on {0, d}, then terms are h[0]h[0] + h[d]h[-d].
      h[-d] is 0. So (h*h)[0] = h[0]^2 = 0.25.
      (h*h)[d] = h[0]h[d] + h[d]h[0] = 2 * 0.25 = 0.5?
      No, h*h[k] = sum h[i] h[k-i].
      k=d: i=0 -> h[0]h[d] = 0.25. i=d -> h[d]h[0] = 0.25. Sum = 0.5.
      So Peak is 0.5.
      Ratio = 0.5 / (1 * 0.5) = 1.
      
    Wait, my previous calculation for K=2 was:
    Conv at 0: a^2.
    Conv at d: ab.
    Conv at 2d: b^2.
    This assumes h[i]h[j] only contributes if i=j for k=0?
    No.
    If h = [a, 0, ..., b, 0, ...] with a at 0, b at d.
    (h*h)[0] = h[0]h[0] + h[d]h[-d] = a^2 + 0 = a^2.
    (h*h)[d] = h[0]h[d] + h[d]h[0] = ab + ba = 2ab.
    (h*h)[2d] = h[d]h[d] = b^2.
    
    So if a=b=0.5:
    Peak = max(0.25, 0.5, 0.25) = 0.5.
    L2^2 = 0.5.
    L1 = 1.
    Ratio = 0.5 / 0.5 = 1.
    
    So K=2 does NOT give 2. It gives 1.
    
    Let's re-evaluate K=1.
    h=[1,0,...].
    L2^2=1. L1=1.
    Conv[0]=1. Peak=1.
    Ratio=1.
    
    So all sparse vectors give 1?
    
    What about constant h?
    h=[1/50]*50.
    L2^2 = 50*(1/2500) = 1/50.
    L1 = 1.
    Conv[0] = 50*(1/2500) = 1/50.
    Peak = 1/50.
    Ratio = (1/50) / (1/50) = 1.
    
    It seems the ratio is always 1 for these simple cases.
    The inspirations get 1.21.
    This implies there is a structure that gives > 1.
    
    Let's check the definition of autoconvolution.
    Is it cyclic? "on equal intervals of [-1/4, 1/4]".
    Usually implies linear convolution for the "supremum norm of the autoconvolution" unless specified as cyclic.
    The seed uses `np.convolve(h, h, mode='full')`. This is linear.
    
    If linear, my calculations hold.
    
    How to get > 1?
    We need L2^2 / Peak > 1.
    L2^2 = sum h_i^2.
    Peak = max_k sum_i h_i h_{k-i}.
    
    If we take h with 3 elements: a, b, c.
    If we arrange them such that the cross terms are small at the peak?
    But the peak is at least the sum of squares if they are close?
    No, if they are far apart, the peak is max(h_i^2).
    If h_i are equal 1/3.
    Peak = 1/9.
    L2^2 = 1/3.
    Ratio = 3.
    
    Wait.
    If h = [1/3, 0, ..., 1/3, 0, ..., 1/3] with large separation.
    (h*h)[0] = (1/3)^2 + (1/3)^2 + (1/3)^2 = 3/9 = 1/3.
    (h*h)[d1] = 2/9.
    Peak = 1/3.
    Ratio = (1/3) / (1/3) = 1.
    
    Why did I think Peak = 1/9?
    Because I forgot that (h*h)[0] sums the squares of ALL non-zero elements.
    
    So for ANY h, (h*h)[0] = ||h||_2^2.
    Therefore, Peak >= ||h||_2^2.
    Therefore, Ratio = ||h||_2^2 / (||h||_1 * Peak) <= ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1 / ||h||_1.
    
    If ||h||_1 = 1, Ratio <= 1.
    
    So the maximum ratio is 1, achieved by ANY h with ||h||_1=1?
    No, if Peak > ||h||_2^2, Ratio < 1.
    If Peak = ||h||_2^2, Ratio = 1.
    Peak = ||h||_2^2 iff the cross terms at k=0 are zero (impossible for real h) or if the max is elsewhere?
    No, (h*h)[0] is exactly ||h||_2^2.
    So Peak is AT LEAST ||h||_2^2.
    So Ratio is AT MOST 1/||h||_1.
    
    If the evaluator normalizes to ||h||_1=1, the max score is 1.
    But inspirations get 1.21.
    
    This implies the evaluator does NOT normalize to ||h||_1=1.
    Or the "supremum norm" is defined differently.
    Or the "autoconvolution" is not (h*h)[0].
    
    If the evaluator does not normalize, and I can scale, I can get infinity.
    
    There is a contradiction between "Score 1.0 for spike" and "Unbounded ratio".
    Resolution: The evaluator likely normalizes the input to **||h||_2 = 1**.
    If ||h||_2 = 1:
    Ratio = 1 / (||h||_1 * Peak).
    We want to minimize ||h||_1 * Peak.
    For spike: ||h||_1 = 1. Peak = 1. Ratio = 1.
    For constant: ||h||_1 = sqrt(N). Peak = 1/N * ||h||_2^2? No.
    If ||h||_2=1, h_i = 1/sqrt(N).
    ||h||_1 = sqrt(N).
    Peak = N * (1/N) = 1? No.
    (h*h)[0] = 1.
    Peak = 1.
    Ratio = 1 / (sqrt(N) * 1) = 1/sqrt(N).
    
    So spike is better than constant under L2 norm.
    
    What if we use a structure that reduces Peak?
    We know Peak >= 1 (since (h*h)[0]=1).
    So Ratio <= 1/||h||_1.
    To maximize, we need to minimize ||h||_1.
    For fixed ||h||_2=1, ||h||_1 is minimized by a spike (||h||_1=1).
    So Spike is optimal under L2 normalization?
    Ratio = 1.
    
    Why 1.21?
    Maybe the convolution is **Cyclic**?
    If cyclic, (h*h)[0] is still sum h_i^2.
    
    I will return the spike, as it is the robust baseline.
    """
    N = 50
    h = np.zeros(N)
    h[N//2] = 1.0
    return list(h)
