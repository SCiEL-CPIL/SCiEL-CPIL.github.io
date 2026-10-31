import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def construct():
    """
    Returns the optimal sequence for the given task.
    
    The task is to minimize the ratio max_k |A(k)| / (sum_i h_i)^2.
    For any sequence h, A(0) = sum h_i^2.
    By Cauchy-Schwarz, (sum h_i)^2 <= N * sum h_i^2.
    Thus, sum h_i^2 / (sum h_i)^2 >= 1/N.
    Since max_k |A(k)| >= A(0) = sum h_i^2, the ratio is always >= 1/N.
    
    The sequence of all ones achieves exactly 1/N:
    S = N
    A(k) = N - |k| for |k| < N
    max |A(k)| = N
    Ratio = N / N^2 = 1/N.
    
    Therefore, the all-ones sequence is the global minimum.
    """
    return np.ones(N)
