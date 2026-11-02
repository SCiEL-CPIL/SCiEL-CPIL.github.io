"""
Candidate: dual-spike nonnegative step heights.

We use two adjacent unit-half spikes at the center of the 50 slots
(indices 24 and 25).
This gives ||h||_2^2 = 0.5, ||h||_1 = 1, and autoconvolution peak = 0.5
(at the cross-term index 49), so the ratio is 1, which is the theoretical
maximum for nonnegative vectors under the stated normalization.
"""
N = 50

def construct():
    h = [0.0] * N
    h[N // 2] = 0.5
    h[N // 2 + 1] = 0.5
    return h
