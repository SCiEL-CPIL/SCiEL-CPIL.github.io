import numpy as np
import time

N = 100

def construct():
    """
    Returns the optimal sequence of step heights h_i.
    
    Problem: Minimize max_k C_k where C_k = sum_i h_i (1 - h_{i+k}).
    Constraints: h_i in [0, 1], sum h_i = N/2.
    
    Analysis:
    C_k = sum h_i - sum h_i h_{i+k} = N/2 - A_k.
    Minimizing max C_k is equivalent to maximizing min A_k.
    
    Sum of C_k over k=1..N-1:
    Sum C_k = (N-1)(N/2) - Sum_{k=1}^{N-1} A_k.
    We know Sum_{k=0}^{N-1} A_k = (Sum h_i)^2 = N^2/4.
    So Sum_{k=1}^{N-1} A_k = N^2/4 - A_0 = N^2/4 - Sum h_i^2.
    
    Sum C_k = N^2/2 - N/2 - (N^2/4 - Sum h_i^2) = N^2/4 - N/2 + Sum h_i^2.
    Average C_k = (N^2/4 - N/2 + Sum h_i^2) / (N-1).
    
    To minimize the maximum, we first minimize the average.
    The average is minimized when Sum h_i^2 is minimized.
    Subject to Sum h_i = N/2 and 0<=h_i<=1, Sum h_i^2 is minimized when all h_i are equal.
    h_i = 0.5.
    
    For h_i = 0.5:
    Sum h_i^2 = N * 0.25 = N/4.
    Average C_k = (N^2/4 - N/2 + N/4) / (N-1) = (N^2 - N) / (4(N-1)) = N/4.
    
    For h_i = 0.5, A_k = Sum 0.5*0.5 = N/4.
    C_k = N/2 - N/4 = N/4 for all k.
    Thus max C_k = N/4.
    
    Since max C_k >= Average C_k >= N/4, the uniform solution is optimal.
    """
    start_time = time.time()
    
    # The optimal solution is uniform h_i = 0.5
    h = np.full(N, 0.5)
    
    # Optional: Verification for safety, though not strictly necessary for the score
    # as the analytical solution is exact.
    # However, ensuring the sum is exactly N/2 is good practice.
    # np.full(N, 0.5) sums to exactly 50.0 in floating point for N=100?
    # 0.5 is exactly representable. 100 * 0.5 = 50.0. Yes.
    
    return h
