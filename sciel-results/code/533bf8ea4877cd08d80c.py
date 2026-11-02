"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    Mathematical Analysis:
    The autoconvolution $g(t) = (f*f)(t)$ of a non-negative step function $f$ 
    with heights $h$ on $N$ intervals of width $w$ has its maximum at $t=0$ 
    (or symmetric points) for distributions that are "unimodal" or uniform.
    Specifically, $g(0) = w \sum_{i=0}^{N-1} h_i^2$.
    Subject to the normalization $\sum_{i=0}^{N-1} h_i w = 1$ (i.e., $\sum h_i = 1/w = 2N$),
    the term $\sum h_i^2$ is minimized when $h_i$ is constant.
    By Cauchy-Schwarz, $\sum h_i^2 \ge \frac{(\sum h_i)^2}{N} = \frac{(2N)^2}{N} = 4N$.
    The minimum is achieved when $h_i = 2$ for all $i$.
    In this case, the autoconvolution is a triangle with peak $g(0) = w \cdot 4N = \frac{1}{2N} \cdot 4N = 2$.
    Any other distribution will have $\sum h_i^2 > 4N$, hence $g(0) > 2$, and thus a higher maximum.
    
    Therefore, the uniform distribution is the global minimizer.
    """
    # The optimal solution is the uniform distribution.
    # Normalization: integral of f is 1.
    # w = 1/(2N). sum(h) * w = 1 => sum(h) = 2N.
    # h_i = 2N / N = 2.
    
    h = np.ones(N) * 2.0
    
    return list(h)
# EVOLVE-BLOCK-END
