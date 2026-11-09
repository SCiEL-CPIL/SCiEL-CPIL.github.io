import numpy as np

def compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    N = len(h)
    max_ov = 0.0
    for s in range(1, N):
        shifted = np.roll(h, s)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    N = 100
    h = np.full(N, 0.5)
    return h
