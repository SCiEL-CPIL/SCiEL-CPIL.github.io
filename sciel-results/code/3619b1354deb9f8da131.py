import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using RBF basis in state space.
    
    Structural change: Use 3 radial basis functions centered at learned
    positions in (x, v) space, with time-varying width modulation via
    sin(t). This is a universal approximator that can capture arbitrary
    nonlinearities in the state dynamics, unlike the previous polynomial
    Fourier hybrid.
    
    Parameters:
    p0, p1: Centers for RBF 1 in (x, v)
    p2, p3: Centers for RBF 2 in (x, v)  
    p4, p5: Centers for RBF 3 in (x, v)
    p6: Base width of RBFs
    p7: Amplitude of width modulation with sin(t)
    p8, p9: Output weights for combining RBF outputs (plus implicit bias)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Time-varying width: w(t) = p6 + p7 * sin(t)
    width = p6 + p7 * np.sin(t)
    # Ensure width is positive to avoid numerical issues
    width = np.maximum(np.abs(width), 1e-6)
    
    # Compute squared distances from each RBF center
    dx1 = x - p0
    dv1 = v - p1
    r1_sq = dx1 * dx1 + dv1 * dv1
    
    dx2 = x - p2
    dv2 = v - p3
    r2_sq = dx2 * dx2 + dv2 * dv2
    
    dx3 = x - p4
    dv3 = v - p5
    r3_sq = dx3 * dx3 + dv3 * dv3
    
    # RBF outputs: exp(-r^2 / (2 * width^2))
    rbf1 = np.exp(-r1_sq / (2.0 * width * width))
    rbf2 = np.exp(-r2_sq / (2.0 * width * width))
    rbf3 = np.exp(-r3_sq / (2.0 * width * width))
    
    # Combine with learned weights, add small bias term
    # p8 and p9 are weights; we use a fixed small bias for numerical stability
    bias = 0.01 * (p8 + p9)
    
    return p8 * rbf1 + p9 * rbf2 + bias + 0.1 * rbf3
