import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed Gaussian Radial Basis Function Basis.
    
    This program uses a structural change from the previous polynomial/sine hybrid
    to a fixed RBF basis. This allows the model to capture non-linear, non-separable
    interactions without explicitly defining interaction terms like x*y, as the
    Gaussian kernels naturally couple the dimensions.
    
    We use 10 Gaussian basis functions centered at key points in the domain [0.5, 2]^3:
    8 corners and 2 interior points.
    The parameters are the amplitudes of these Gaussians.
    
    This is a linear model in the parameters, so BFGS will converge quickly to the
    optimal amplitudes for the fixed centers.
    """
    p = params
    
    # Centers for the Gaussian basis functions
    # Corners of the cube [0.5, 2]
    c1 = np.array([0.5, 0.5, 0.5])
    c2 = np.array([2.0, 0.5, 0.5])
    c3 = np.array([0.5, 2.0, 0.5])
    c4 = np.array([0.5, 0.5, 2.0])
    c5 = np.array([2.0, 2.0, 0.5])
    c6 = np.array([2.0, 0.5, 2.0])
    c7 = np.array([0.5, 2.0, 2.0])
    c8 = np.array([2.0, 2.0, 2.0])
    # Interior points
    c9 = np.array([1.25, 1.25, 1.25])
    c10 = np.array([1.25, 0.5, 1.25])
    
    # Stack inputs into a (N, 3) array
    X = np.column_stack((x, y, z))
    
    # Variance for the Gaussians. 
    # Domain size is 1.5. A sigma of 0.5 provides local but overlapping support.
    sigma2 = 0.25 # sigma = 0.5
    
    # Calculate squared distances to each center
    # dist_i^2 = ||X - c_i||^2
    # We can compute this efficiently
    
    # p[0] * G1 + ... + p[9] * G10
    
    # G1
    d1_sq = (x - c1[0])**2 + (y - c1[1])**2 + (z - c1[2])**2
    G1 = np.exp(-d1_sq / (2 * sigma2))
    
    # G2
    d2_sq = (x - c2[0])**2 + (y - c2[1])**2 + (z - c2[2])**2
    G2 = np.exp(-d2_sq / (2 * sigma2))
    
    # G3
    d3_sq = (x - c3[0])**2 + (y - c3[1])**2 + (z - c3[2])**2
    G3 = np.exp(-d3_sq / (2 * sigma2))
    
    # G4
    d4_sq = (x - c4[0])**2 + (y - c4[1])**2 + (z - c4[2])**2
    G4 = np.exp(-d4_sq / (2 * sigma2))
    
    # G5
    d5_sq = (x - c5[0])**2 + (y - c5[1])**2 + (z - c5[2])**2
    G5 = np.exp(-d5_sq / (2 * sigma2))
    
    # G6
    d6_sq = (x - c6[0])**2 + (y - c6[1])**2 + (z - c6[2])**2
    G6 = np.exp(-d6_sq / (2 * sigma2))
    
    # G7
    d7_sq = (x - c7[0])**2 + (y - c7[1])**2 + (z - c7[2])**2
    G7 = np.exp(-d7_sq / (2 * sigma2))
    
    # G8
    d8_sq = (x - c8[0])**2 + (y - c8[1])**2 + (z - c8[2])**2
    G8 = np.exp(-d8_sq / (2 * sigma2))
    
    # G9
    d9_sq = (x - c9[0])**2 + (y - c9[1])**2 + (z - c9[2])**2
    G9 = np.exp(-d9_sq / (2 * sigma2))
    
    # G10
    d10_sq = (x - c10[0])**2 + (y - c10[1])**2 + (z - c10[2])**2
    G10 = np.exp(-d10_sq / (2 * sigma2))
    
    return (p[0] * G1 + p[1] * G2 + p[2] * G3 + p[3] * G4 + p[4] * G5 +
            p[5] * G6 + p[6] * G7 + p[7] * G8 + p[8] * G9 + p[9] * G10)
