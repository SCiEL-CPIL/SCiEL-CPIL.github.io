import numpy as np

N = 600

def construct():
    # Analytical solution for minimizing sum(h^2) / (sum(h) * max(h))
    # with max(h)=1 and h >= 0.
    # Optimal shape: one bin at 1, others at 'a'.
    # a = (sqrt(N) - 1) / (N - 1)
    
    a = (np.sqrt(N) - 1) / (N - 1)
    
    h = np.full(N, a)
    h[0] = 1.0
    
    # Optional: Quick local optimization to ensure no discrete artifact
    # The analytical solution is very likely optimal, but let's verify
    # and potentially tweak if needed.
    # However, N=600 is large, so the continuous approximation is very good.
    
    return list(h)
