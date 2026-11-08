"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF model for nonlinear oscillator acceleration.
    
    Strategy:
    Instead of a global polynomial expansion, we use a linear combination of 
    Gaussian radial basis functions (RBFs). This allows the model to fit 
    highly nonlinear, local, and coupled behaviors in the (t, x, v) space.
    
    We use 10 fixed centers distributed in a 2x2x2 grid across the normalized 
    input ranges. The 10 parameters correspond to the weights of these centers.
    
    Input Ranges (from task description):
    t: [30, 50]
    x: [-0.2056, 0.1749]
    v: [-0.3391, 0.3624]
    
    Normalization:
    Map inputs to [-1, 1] or [0, 1] to ensure Gaussian kernels have reasonable 
    scale. We use [0, 1] scaling for simplicity in center placement.
    
    Centers:
    t_centers: [30, 50] -> normalized [0, 1]
    x_centers: [-0.2056, 0.1749] -> normalized [0, 1]
    v_centers: [-0.3391, 0.3624] -> normalized [0, 1]
    
    Grid points (2 per dimension = 8 points? No, we need 10 params).
    Let's use a 2x2x2 grid (8 centers) + 2 extra centers? 
    Or just 10 centers from a 2x2x3 grid? 
    2 (t) * 2 (x) * 2 (v) = 8. 
    Let's add 2 more centers to make 10.
    Actually, a 2x2x2 grid gives 8. 
    Let's use 10 centers: 
    4 in the "low t" half, 4 in the "high t" half? 
    Simpler: Just define 10 specific centers.
    
    Let's use a 2x2x2 grid (8 centers) plus the center of the domain and one more?
    Or just use 10 centers from a 2x2x3 grid (12 points, pick 10)?
    
    Let's stick to a clean 2x2x2 grid (8 centers) and add 2 centers at the 
    midpoints of the t-axis for each (x,v) corner? That might be redundant.
    
    Better: Use 10 centers from a 2x2x3 grid:
    t: [0.2, 0.8] (2 points)
    x: [0.2, 0.8] (2 points)
    v: [0.1, 0.5, 0.9] (3 points)
    Total = 2 * 2 * 3 = 12 centers. We only have 10 params.
    
    Let's use 2x2x2 = 8 centers and add 2 centers at the center of the domain?
    No, let's just use 10 centers from a 2x2x3 grid by dropping 2?
    
    Actually, the simplest robust approach: 
    Use 10 centers. 
    t: [0.2, 0.8]
    x: [0.2, 0.8]
    v: [0.2, 0.8]
    This is 8 centers. 
    Add 2 more: (0.5, 0.5, 0.2) and (0.5, 0.5, 0.8)?
    
    Let's define the centers explicitly in normalized space [0,1]^3.
    
    Centers:
    1. (0.2, 0.2, 0.2)
    2. (0.2, 0.2, 0.8)
    3. (0.2, 0.8, 0.2)
    4. (0.2, 0.8, 0.8)
    5. (0.8, 0.2, 0.2)
    6. (0.8, 0.2, 0.8)
    7. (0.8, 0.8, 0.2)
    8. (0.8, 0.8, 0.8)
    9. (0.5, 0.5, 0.5)  # Center
    10. (0.5, 0.2, 0.8) # Extra
    
    Kernel Width:
    The distance in normalized space is up to sqrt(3) ~ 1.73.
    A width (sigma) of 0.5 seems reasonable to capture local variations 
    without being too narrow (which would make the fit hard) or too wide 
    (which reduces to a constant).
    
    Calculation:
    1. Normalize t, x, v to [0, 1].
    2. Compute squared Euclidean distance to each of the 10 centers.
    3. Compute Gaussian kernel values: exp(-d^2 / (2 * sigma^2)).
    4. Return dot product of kernel matrix and params.
    """
    
    # Normalization constants
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    t_norm = (t - t_min) / (t_max - t_min)
    x_norm = (x - x_min) / (x_max - x_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    # Define 10 centers in normalized space
    # 2x2x2 grid corners
    c1 = np.array([0.2, 0.2, 0.2])
    c2 = np.array([0.2, 0.2, 0.8])
    c3 = np.array([0.2, 0.8, 0.2])
    c4 = np.array([0.2, 0.8, 0.8])
    c5 = np.array([0.8, 0.2, 0.2])
    c6 = np.array([0.8, 0.2, 0.8])
    c7 = np.array([0.8, 0.8, 0.2])
    c8 = np.array([0.8, 0.8, 0.8])
    # Center and one extra
    c9 = np.array([0.5, 0.5, 0.5])
    c10 = np.array([0.5, 0.2, 0.8])
    
    centers = np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    # Input vector shape: (N, 3)
    inputs = np.column_stack((t_norm, x_norm, v_norm))
    
    # Compute squared distances: (N, 10)
    # inputs: (N, 3), centers: (10, 3)
    # diff: (N, 10, 3)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    
    # Gaussian kernel
    sigma = 0.5
    kernel_vals = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    # Linear combination
    # kernel_vals: (N, 10), params: (10,)
    return kernel_vals @ params

# EVOLVE-BLOCK-END
