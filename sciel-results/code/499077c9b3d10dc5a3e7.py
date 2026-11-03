import numpy as np

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Strategy:
    Instead of using an arithmetic progression (seed) or geometric progression,
    we use the roots of the Laguerre polynomial L_12(x) scaled to match the 
    sum of the seed roots. This leverages the structural property that 
    orthogonal polynomial nodes provide optimal distribution for moment-based 
    uncertainty bounds, balancing the 'repulsion' between roots to minimize 
    higher-order moments (like the 4th moment) for a given total spread.
    """
    # Get the 12 roots of the Laguerre polynomial L_12(x)
    # laggauss returns the nodes (roots) and weights for Gaussian quadrature
    roots, _ = np.polynomial.laguerre.laggauss(12)
    
    # The sum of the roots of L_n(x) is n(n+1)
    # For n=12, sum = 12*13 = 156
    seed_sum = sum(4.0 * (i + 1) for i in range(12)) # 4 * (1+...+12) = 4 * 78 = 312
    
    # Scale the Laguerre roots to match the sum of the seed roots
    # This maintains the overall "scale" of the system while adopting the 
    # optimal internal distribution of the Laguerre nodes.
    scale_factor = seed_sum / np.sum(roots)
    
    scaled_roots = roots * scale_factor
    
    # Ensure the output is a list of floats as required
    return [float(x) for x in scaled_roots]
