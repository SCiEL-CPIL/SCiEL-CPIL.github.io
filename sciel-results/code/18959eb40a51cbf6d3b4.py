import numpy as np
from scipy.special import roots_laguerre

def construct():
    # Compute the roots of the Laguerre polynomial L_12(x)
    # These are the optimal nodes for the Laguerre weight e^{-x}
    nodes, weights = roots_laguerre(12)
    # The nodes are already sorted in increasing order
    return [float(x) for x in nodes]
