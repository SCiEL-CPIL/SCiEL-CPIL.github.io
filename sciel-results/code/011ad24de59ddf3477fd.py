# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 3 kernels in 3D, sharing a single width parameter.
    10 params: [w1, w2, w3, c1x, c1y, c1z, c2x, c2y, c2z, s]
    """
    p = params
    # Extract parameters
    w1, w2, w3 = p[0], p[1], p[2]
    c1x, c1y, c1z = p[3], p[4], p[5]
    c2x, c2y, c2z = p[6], p[7], p[8]
    s = p[9]
    
    # Smooth positive width
    s_safe = np.exp(s)
    
    # Compute squared distances
    d1_sq = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    d2_sq = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    
    # Wait, I need a 3rd kernel center. I only have 6 center params.
    # c1 (3) and c2 (3). That's 2 kernels.
    # To get 3 kernels with 10 params:
    # Weights: 3
    # Centers: 2 * 3 = 6
    # Width: 1
    # Total: 10.
    # This is 2 kernels.
    
    # To get 3 kernels, I need 3 * 3 = 9 center params + 3 weights = 12. Too many.
    # Unless I fix one center? Or use 2D centers?
    # Or use 1 kernel + 9 linear/quadratic terms?
    
    # Let's stick to 2 kernels but improve the width parameterization and maybe add a bias?
    # If I use 2 kernels (8 params for centers/widths) + 2 weights = 10.
    # Can I add a bias? No, no params left.
    # Can I make the weights include a bias?
    # Output = w1*k1 + w2*k2.
    # If k1 and k2 are far away, they go to 0.
    # If the function has a constant offset, we need a bias.
    # We can force one kernel to be "wide" and act as a bias?
    # Or we can use 1 RBF + 9 linear basis functions?
    # Basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz? (10 terms)
    # If we use a polynomial of degree 2, we have 10 coefficients.
    # This is a very different family (Polynomial vs RBF).
    # The directive says "KEEP this design and investigate improvements within its family".
    # So I should stick to RBF.
    
    # How to add bias to RBF with 10 params?
    # Use 1 RBF (4 params: w, cx, cy, cz, s -> 5 params)
    # And 5 linear terms?
    # This is a hybrid.
    
    # Let's try 2 RBFs with a shared width and one extra param for bias?
    # Params:
    # w1, w2 (2)
    # c1x, c1y, c1z (3)
    # c2x, c2y, c2z (3)
    # s (1)
    # bias (1)
    # Total: 10.
    # This is a strong candidate. 2 kernels + bias.
    
    p = params
    w1, w2 = p[0], p[1]
    c1x, c1y, c1z = p[2], p[3], p[4]
    c2x, c2y, c2z = p[5], p[6], p[7]
    s = p[8]
    bias = p[9]
    
    s_safe = np.exp(s)
    
    d1_sq = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    d2_sq = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    
    k1 = np.exp(-d1_sq / (2.0 * s_safe**2))
    k2 = np.exp(-d2_sq / (2.0 * s_safe**2))
    
    return w1 * k1 + w2 * k2 + bias
# EVOLVE-BLOCK-END
