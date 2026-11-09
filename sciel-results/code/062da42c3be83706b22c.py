"""
Strategy: Radial Basis Function (RBF) model with Gaussian kernels.
Instead of a global polynomial expansion, we use local basis functions centered
at the corners of the input domain. This allows the model to capture localized
nonlinearities and boundary behaviors of the oscillator more accurately than
polynomials, which tend to oscillate (Runge's phenomenon) or smooth out sharp
features. The centers are fixed at the corners of the training range to provide
a structured, physically grounded set of local "atoms" for the optimizer to weight.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts acceleration using a Radial Basis Function (RBF) model.
    
    Model:
    a = p[0] + p[1]*x + p[2]*v + sum_{i=1 to 8} p[i] * exp(-r_i^2 / (2 * sigma^2))
    
    Where r_i is the Euclidean distance from (x, v) to the i-th center.
    The centers are fixed at the corners of the input domain:
    [-0.5728, 0.5275] for x
    [-0.1754, 0.2821] for v
    
    Parameters:
    p[0]: Bias
    p[1]: Linear x coefficient
    p[2]: Linear v coefficient
    p[3]: Weight for RBF at (x_min, v_min)
    p[4]: Weight for RBF at (x_min, v_max)
    p[5]: Weight for RBF at (x_max, v_min)
    p[6]: Weight for RBF at (x_max, v_max)
    p[7]: Weight for RBF at (x_mid, v_min)
    p[8]: Weight for RBF at (x_mid, v_max)
    p[9]: Weight for RBF at (x_min, v_mid) -- using 4 corners + 4 edges for 8 RBFs + 3 linear = 11 params? 
    Wait, I have 10 params.
    Let's use 4 corner RBFs + 4 edge-mid RBFs + 2 linear terms + 1 bias = 11. Too many.
    
    Revised Plan:
    1 Bias
    1 Linear x
    1 Linear v
    7 RBF terms?
    
    Let's stick to 4 Corners + 1 Center + 1 Edge-Mid X + 1 Edge-Mid V = 7 RBFs.
    1 Bias + 1 Lin X + 1 Lin V + 7 RBFs = 10 Params.
    
    Centers:
    C1: (x_min, v_min)
    C2: (x_max, v_min)
    C3: (x_min, v_max)
    C4: (x_max, v_max)
    C5: (x_mid, v_mid)
    C6: (x_mid, v_min)
    C7: (x_mid, v_max)
    
    This provides coverage of the domain.
    """
    p = params
    
    # Fixed centers based on training range
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    x_mid = (x_min + x_max) / 2.0
    v_mid = (v_min + v_max) / 2.0
    
    # Define centers
    centers = np.array([
        [x_min, v_min],
        [x_max, v_min],
        [x_min, v_max],
        [x_max, v_max],
        [x_mid, v_mid],
        [x_mid, v_min],
        [x_mid, v_max]
    ])
    
    # Precompute distances
    # x and v are 1D arrays of same length
    # We need to compute distance from each point (x[i], v[i]) to each center
    # To avoid creating a huge 2D array of distances for every evaluation (memory/time),
    # we can compute the RBF terms iteratively or vectorize carefully.
    
    # Since N=10000 and K=7, a 10000x7 matrix is fine.
    # x shape: (N,)
    # centers shape: (7, 2)
    
    # dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (N, 7)
    # dv = v[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (N, 7)
    # r2 = dx**2 + dv**2                                   # (N, 7)
    
    # Using a fixed bandwidth sigma. 
    # The scale of x is ~1.1, v is ~0.45.
    # A sigma of 0.2-0.5 seems reasonable. 
    # Since we can't fit sigma easily with BFGS from 1s (it's a scale parameter, can be tricky),
    # we fix it. Let's try sigma = 0.3.
    sigma2 = 0.09 # sigma = 0.3
    
    # Calculate RBF values
    # To be efficient, we compute each term separately to avoid large temporary arrays if possible, 
    # but 10000*7*8 bytes is ~560KB, which is negligible.
    
    dx1 = x - x_min
    dv1 = v - v_min
    rbf1 = np.exp(-(dx1*dx1 + dv1*dv1) / sigma2)
    
    dx2 = x - x_max
    dv2 = v - v_min
    rbf2 = np.exp(-(dx2*dx2 + dv2*dv2) / sigma2)
    
    dx3 = x - x_min
    dv3 = v - v_max
    rbf3 = np.exp(-(dx3*dx3 + dv3*dv3) / sigma2)
    
    dx4 = x - x_max
    dv4 = v - v_max
    rbf4 = np.exp(-(dx4*dx4 + dv4*dv4) / sigma2)
    
    dx5 = x - x_mid
    dv5 = v - v_mid
    rbf5 = np.exp(-(dx5*dx5 + dv5*dv5) / sigma2)
    
    dx6 = x - x_mid
    dv6 = v - v_min
    rbf6 = np.exp(-(dx6*dx6 + dv6*dv6) / sigma2)
    
    dx7 = x - x_mid
    dv7 = v - v_max
    rbf7 = np.exp(-(dx7*dx7 + dv7*dv7) / sigma2)
    
    # Combine
    # p[0]: bias
    # p[1]: x
    # p[2]: v
    # p[3]: rbf1
    # p[4]: rbf2
    # p[5]: rbf3
    # p[6]: rbf4
    # p[7]: rbf5
    # p[8]: rbf6
    # p[9]: rbf7
    
    return (p[0] + 
            p[1] * x + 
            p[2] * v + 
            p[3] * rbf1 + 
            p[4] * rbf2 + 
            p[5] * rbf3 + 
            p[6] * rbf4 + 
            p[7] * rbf5 + 
            p[8] * rbf6 + 
            p[9] * rbf7)
# EVOLVE-BLOCK-END
