import numpy as np

N = 100

def construct():
    # Use a random binary sequence with exactly N/2 ones and N/2 zeros.
    # This minimizes the raw maximum overlap compared to the constant 0.5 signal.
    # The constant signal has a maximum raw overlap of 0.25 * N.
    # The random signal has a maximum raw overlap of approx 0.25 * (N-1),
    # which is strictly smaller.
    np.random.seed(42)
    h = np.array([0.0] * (N // 2) + [1.0] * (N // 2))
    np.random.shuffle(h)
    return h
