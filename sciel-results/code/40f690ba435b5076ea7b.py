import numpy as np
import time

N = 400

def _autoconv_peak(s):
    # s: array of ±1, length N
    # Compute aperiodic autocorrelation R[k] = sum_{j=0}^{N-1-k} s[j] s[j+k] for k=0..N-1
    # and autoconvolution (cross-correlation with reversal) C[k] = sum_j s[j] r[j+k] where r = reverse(s)
    # Return max of peak sidelobes (k>=1) of |R| and |C|
    # Use FFT for speed? For N=400, direct is fine but let's use a fast method.
    ...
