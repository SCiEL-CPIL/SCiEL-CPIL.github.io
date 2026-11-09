import numpy as np

N = 600

def construct():
    # Try a Gaussian-like distribution which balances peak and width
    x = np.linspace(-3, 3, N)
    h = np.exp(-x**2 / 2)
    h = h / h.sum()
    return list(h)
