import numpy as np

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination.
    Strategy: Use the roots of the 12th Laguerre polynomial (Gauss-Laguerre nodes)
    scaled to match the mean/sum of the seed arithmetic progression.
    The seed roots [4, 8, ..., 48] have a sum of 312.
    The roots of L_12(x) have a sum of 156.
    Scaling by 2 aligns the 'center of mass' of the roots with the seed,
    while adopting the optimal non-uniform spacing of the Laguerre basis.
    """
    # Calculate roots of the 12th Laguerre polynomial
    # laggauss returns (x, w) where x are the roots
    roots, _ = np.polynomial.laguerre.laggauss(12)
    
    # The sum of roots for L_n is n(n+1). For n=12, sum=156.
    # Seed sum: sum(4*i for i in 1..12) = 4 * 78 = 312.
    # Scale factor: 312 / 156 = 2.0
    scale = 2.0
    
    # Apply scaling
    optimized_roots = roots * scale
    
    # Ensure they are positive and sorted (laggauss returns sorted positive roots)
    # Convert to list of floats as required
    return [float(r) for r in optimized_roots]
