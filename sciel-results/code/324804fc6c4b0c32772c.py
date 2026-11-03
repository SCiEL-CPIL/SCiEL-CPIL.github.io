import numpy as np

N = 100

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The problem is to minimize the maximum overlap of h with shifted 1-h.
    Let O_k be the overlap for shift k.
    O_k = sum_i h_i * (1 - h_{(i+k)%N}) = N/2 - C_k
    where C_k is the circular autocorrelation of h at lag k.
    
    We want to minimize max_k O_k, which is equivalent to maximizing min_k C_k.
    The average autocorrelation is N/4.
    Thus min_k C_k <= N/4, so max_k O_k >= N/4.
    
    The uniform sequence h_i = 0.5 achieves C_k = N/4 for all k,
    resulting in max_k O_k = N/4.
    This is the theoretical lower bound, hence the global optimum.
    """
    return np.full(N, 0.5)
