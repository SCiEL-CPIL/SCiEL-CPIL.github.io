import numpy as np

N = 600

def construct():
    # Create a zero array
    f = np.zeros(N)
    
    # We want to minimize the peak of the autoconvolution.
    # If the normalization is L_inf = 1, the peak is proportional to the support width.
    # Smaller support -> smaller peak.
    # Let's use a single bin or a small number of bins.
    # Using 1 bin:
    # f[N//2] = 1.0
    
    # Using 3 bins to be safe against discrete artifacts:
    center = N // 2
    f[center-1:center+2] = 1.0
    
    return list(f)
