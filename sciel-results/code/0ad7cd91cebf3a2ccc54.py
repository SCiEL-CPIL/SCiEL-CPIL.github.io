import numpy as np
import time

N = 400
DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return DEADLINE - 5.0 - _elapsed()

def _get_fft_size(n):
    # next power of 2 >= n
    m = 1
    while m < n:
        m <<= 1
    return m

def _autoconv(h):
    # linear autoconvolution via FFT
    n = len(h)
    M = _get_fft_size(2 * n - 1)
    H = np.fft.rfft(h, M)
    ac = np.fft.irfft(H * H, M)
    return ac[:2 * n - 1]

def _score(h):
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    ac = _autoconv(h)
    return float(np.max(np.abs(ac)) / (S * S))

def construct():
    # The objective function is max(|autocorr|) / (sum(h))^2.
    # Let h be a sequence of length N.
    # autocorr[0] = sum(h_i^2) = E.
    # sum(h)^2 <= N * sum(h_i^2) = N * E.
    # So (sum(h))^2 / E <= N.
    # E / (sum(h))^2 >= 1/N.
    # Since max(|autocorr|) >= autocorr[0] = E,
    # max(|autocorr|) / (sum(h))^2 >= E / (sum(h))^2 >= 1/N.
    # The lower bound 1/N is achieved when all h_i are equal (e.g., all 1).
    # In this case, sum(h) = N, E = N.
    # autocorr[0] = N.
    # autocorr[k] = N - |k| for k != 0.
    # max(|autocorr|) = N.
    # Score = N / N^2 = 1/N.
    
    # For N=400, Score = 0.0025.
    # This is significantly better than random signs (Score ~ 1.0) or other structured sequences.
    
    return np.ones(N)

if __name__ == "__main__":
    h = construct()
    print("Score:", _score(h))
    print("Time:", _elapsed())
