import numpy as np
import time
from scipy.optimize import minimize
import scipy.signal

N = 600
TIME_LIMIT = 115

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def softmax(y):
    """Stable softmax."""
    y_max = np.max(y)
    exp_y = np.exp(y - y_max)
    return exp_y / np.sum(exp_y)

def linear_autocorr_fft(f):
    """Compute linear autocorrelation using FFT."""
    n = len(f)
    n_fft = 2 * n
    F = np.fft.rfft(f, n=n_fft)
    power = np.abs(F) ** 2
    corr = np.fft.irfft(power, n=n_fft)
    # The linear autocorrelation is in the first n elements and the last n-1 elements
    # Actually, irfft of |F|^2 gives circular autocorrelation of length n_fft.
    # For zero-padded f, the first n elements are c_{N-1} ... c_0?
    # Let's verify.
    # c_k = sum f_j f_{j+k}.
    # Standard way:
    # corr = np.correlate(f, f, mode='full')
    # This is O(N^2).
    # FFT way:
    # F = fft(f, 2*N)
    # P = |F|^2
    # c = ifft(P)
    # c is circular autocorrelation of length 2*N.
    # The linear autocorrelation c_lin[k] for k=-(N-1)..N-1 is contained in c.
    # c[N-1] corresponds to lag 0?
    # Let's test with small N.
    # f = [1, 2]. N=2.
    # c_lin = [2, 5, 2]. (lags -1, 0, 1)
    # fft([1, 2, 0, 0]) -> F.
    # |F|^2 -> P.
    # ifft(P) -> c_circ.
    # c_circ[0] = c_lin[0] = 5?
    # c_circ[1] = c_lin[-1] = 2?
    # c_circ[2] = c_lin[1] = 2?
    # c_circ[3] = 0?
    # So c_circ[0] is lag 0.
    # c_circ[k] for k=1..N-1 is lag -k.
    # c_circ[N..2N-1-k] is lag k?
    # Actually, c_circ is symmetric.
    # We want c_lin of length 2N-1.
    # c_lin[0] = c_circ[N-1]? No.
    # Let's just use np.convolve for simplicity if N=600 is fast enough?
    # 600*600 = 360k ops.
    # 1000 iterations * 360k = 3.6e8 ops.
    # In Python, this might be slow (10-20s).
    # FFT is safer.
    
    # Correct extraction:
    # c_circ = ifft(|fft(f, 2N)|^2)
    # c_lin = np.concatenate([c_circ[N-1::-1], c_circ[0:N]])
    # Let's check indices.
    # c_circ[0] is sum f_j f_j = sum f^2. This is lag 0.
    # c_circ[1] is sum f_j f_{j+1} (circular). For zero-padded, this is sum f_j f_{j+1} for j=0..N-2. This is lag 1?
    # No, circular shift by 1: f_{j+1}.
    # So c_circ[k] for k=0..N-1 corresponds to lags 0, 1, ..., N-1?
    # And c_circ[N..2N-1] corresponds to lags -1, ..., -(N-1)?
    # So c_lin (lags -(N-1)..N-1) can be formed by:
    # lags: -(N-1) ... -1, 0, 1 ... N-1
    # c_circ indices: N ... 2N-1, 0, 1 ... N-1
    # So c_lin = np.concatenate([c_circ[N:2*N-1], c_circ[0:N]])
    # Wait, c_circ has length 2N.
    # Indices 0 to 2N-1.
    # c_circ[N] is lag -1?
    # Let's assume this mapping.
    
    c_circ = np.fft.irfft(power, n=n_fft)
    # c_circ[0] is lag 0.
    # c_circ[k] is lag k for k=0..N-1.
    # c_circ[N+k] is lag -k-1?
    # Let's just use np.convolve for the gradient computation if it's not too slow, or implement the FFT extraction carefully.
    # For robustness, I will use np.convolve in the objective/gradient if N is small enough, or use the FFT method.
    # Given N=600, np.convolve is O(N^2).
    # Let's try to use FFT for speed.
    
    # Mapping:
    # c_circ[0] -> lag 0
    # c_circ[1] -> lag 1
    # ...
    # c_circ[N-1] -> lag N-1
    # c_circ[N] -> lag -(N-1) ??
    # Actually, c_circ[k] = sum f_j f_{(j+k) mod 2N}.
    # For k=1, sum f_j f_{j+1}. This is lag 1.
    # For k=N, sum f_j f_{j+N}. Since f is zero-padded to 2N, f_{j+N} is 0 for j<N?
    # No, f is [f_0..f_{N-1}, 0..0].
    # f_{j+N} is 0 for all j. So c_circ[N] = 0?
    # This is not right.
    # The autocorrelation of a zero-padded signal via FFT gives the linear autocorrelation wrapped around.
    # The linear autocorrelation c_lin[k] is non-zero for k in [-(N-1), N-1].
    # c_circ[k] = c_lin[k] for k in [0, N-1].
    # c_circ[k] = c_lin[k-N] for k in [N, 2N-1].
    # So c_lin[k-N] = c_circ[k].
    # Let m = k-N. m in [0, N-1].
    # c_lin[-(N-m)] = c_circ[N+m]?
    # Let's just extract:
    # lags: 0, 1, ..., N-1 are in c_circ[0:N].
    # lags: -1, -2, ..., -(N-1) are in c_circ[N:2N-1]?
    # c_circ[N] corresponds to lag -1?
    # Let's verify with N=2. f=[1,2].
    # c_lin = [2, 5, 2]. (lags -1, 0, 1)
    # c_circ = [5, 2, 2, 0]?
    # c_circ[0]=5 (lag 0).
    # c_circ[1]=2 (lag 1).
    # c_circ[2]=2 (lag -1).
    # c_circ[3]=0.
    # So c_lin[-1] = c_circ[2] = c_circ[N].
    # c_lin[-2] would be c_circ[3]?
    # So c_lin[k] for k<0 is c_circ[N+k]?
    # If k=-1, N-1 = 1? No, N=2, N-1=1. c_circ[1] is lag 1.
    # My previous check: c_circ[2] is lag -1. Index 2 = N.
    # So c_lin[k] for k<0 is c_circ[N+k]?
    # If k=-1, N-1? No, N+(-1) = 1. c_circ[1] is lag 1.
    # Contradiction.
    
    # Let's use a known library function or simple convolution to be safe.
    # N=600 is small.
    pass

def construct():
    # Placeholder
    pass
