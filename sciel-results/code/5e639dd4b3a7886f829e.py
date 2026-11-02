import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Optimize epsilon for K=1
    # Minimize (1 + (N-1)*eps^2) / (1 + (N-1)*eps)
    # Derivative root: (N-1)*eps^2 + 2*eps - 1 = 0
    # eps = (-2 + sqrt(4 + 4*(N-1))) / (2*(N-1))
    # eps = (sqrt(N) - 1) / (N - 1)
    
    eps = (np.sqrt(N) - 1.0) / (N - 1.0)
    
    # Construct vector
    x = np.full(N, eps)
    x[0] = 1.0
    
    # Verify max is at lag 0
    # x*x[0] = 1 + (N-1)*eps^2
    # x*x[1] approx 2*eps + (N-2)*eps^2
    # 1 > 2*eps is true for N > 4
    
    return x.tolist()
# EVOLVE-BLOCK-END
