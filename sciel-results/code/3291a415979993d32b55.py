import numpy as np

N = 600

def construct():
    # Triangle function on [-1/4, 1/4]
    # x ranges from -1/4 to 1/4
    # Centers of intervals
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5/N)*0.5
    # f(x) = max(0, 1 - 4*|x|)
    f = np.maximum(0, 1 - 4 * np.abs(x))
    return f.tolist()
