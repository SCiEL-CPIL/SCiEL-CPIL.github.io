import numpy as np

def construct():
    N = 50
    # The theoretical maximum for the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)
    # for a step function with bin width dx is 1/dx.
    # Here dx = 0.5 / 50 = 0.01, so max ratio is 100.
    # Any contiguous "box" function (constant height over a contiguous range)
    # achieves this bound, as its autoconvolution is a triangle with peak at 0.
    # A single bin is the simplest box.
    
    h = np.zeros(N)
    h[0] = 1.0
    
    return list(h)
