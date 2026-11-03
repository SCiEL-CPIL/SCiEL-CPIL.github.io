"""
Rewritten program for the suite task.
The task is to minimize the maximum absolute autoconvolution divided by the squared sum of the sequence.
For a sequence of length N with elements x_i, the metric is:
    score_value = max_k |sum_i x_i * x_{i+k}| / (sum_i x_i)^2

Analysis:
1. The term sum_i x_i * x_i (k=0) is sum_i x_i^2.
2. For any real-valued sequence, max_k |c_k| >= |c_0| = sum_i x_i^2.
3. By Cauchy-Schwarz, (sum_i x_i)^2 <= N * sum_i x_i^2.
4. Therefore, (sum_i x_i)^2 / sum_i x_i^2 <= N.
5. The ratio R = max |c_k| / (sum x_i)^2 >= (sum x_i^2) / (sum x_i)^2 >= 1/N.
6. The minimum possible value is 1/N, achieved when x_i are all equal (e.g., all 1s).
7. Any deviation from equal values increases the ratio (worsens the raw value), leading to a score <= 1.0.
8. Since the seed program returns np.ones(N) and scores 1.0, it is optimal.

Strategy:
- Return the optimal constant sequence.
- Include a deadline guard as per directive, although the computation is O(1).
"""

import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 120.0  # 120 seconds wall-clock limit
    
    # The optimal sequence for this metric is a constant sequence.
    # We return an array of ones.
    result = np.ones(N)
    
    # Check deadline (trivial here, but required by directive)
    if time.time() > deadline:
        return result
        
    return result

# EVOLVE-BLOCK-END
