"""Return the constant vector, which achieves the theoretical lower bound for the objective."""
import numpy as np

N = 400

def construct():
    """Return a constant vector of +1s, which minimizes the objective."""
    return np.ones(N)
