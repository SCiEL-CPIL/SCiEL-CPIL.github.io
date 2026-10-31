import numpy as np

N = 600

def construct():
    """
    Constructs a Gaussian step function.
    
    Rationale: The uniform distribution (seed) likely maximizes the peak sidelobe 
    ratio of the autoconvolution. By using a Gaussian window, we significantly 
    reduce the sidelobes, which may correspond to the 'normalized maximum' 
    if it refers to off-peak values. This is a structural change from flat 
    to concentrated/smooth.
    """
    centers = np.arange(N)
    center = N / 2.0
    sigma = N / 8.0
    h = np.exp(-((centers - center) ** 2) / (2 * sigma ** 2))
    return list(h)
