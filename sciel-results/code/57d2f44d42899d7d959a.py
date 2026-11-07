"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Gaussian Radial Basis Function (RBF) network.
    
    Strategy:
    - Instead of a global Fourier series, we use 5 localized Gaussian bumps.
    - The 10 parameters are non-linearly mapped to control the centers and widths of these Gaussians.
    - params[0:5] are mapped via sigmoid to centers in [-3, 3].
    - params[5:10] are mapped via softplus (or similar) to positive widths.
    - The amplitudes of the Gaussians are fixed to 1 (or could be derived, but we only have 10 params).
      Actually, to maximize flexibility, we might need amplitudes too. 
      Wait, we have exactly 10 params.
      If we use 5 centers and 5 widths, we have no free amplitudes. This might be too constrained.
      Alternative: 3 centers, 3 widths, 3 amplitudes, 1 bias, 1 extra? 
      Or: 4 centers, 4 widths, 2 amplitudes? 
      
      Let's try a more standard RBF approach where we fix the centers to a grid and only learn amplitudes? 
      No, that's linear. The current Fourier is linear. 
      The directive asks for a STRUCTURAL change. A non-linear parameterization is key.
      
      Let's try: 
      5 centers c_i = 3 * tanh(params[i])
      5 widths w_i = 0.5 + 2.0 * sigmoid(params[i+5])
      Amplitudes are fixed to 1? That's risky. 
      
      Better: 
      Use 4 Gaussians.
      4 centers c_i = 3 * tanh(p_i)
      4 widths w_i = 0.2 + 2.8 * sigmoid(p_{i+4})
      2 amplitudes? No, each Gaussian needs an amplitude. 
      
      If we fix amplitudes to 1, the function is sum of 5 unit-height bumps. This can't represent large values or negative values well unless the target is bounded. 
      The target y has some variance. 
      
      Let's look at the parameters again. 
      Maybe we can use a smaller number of Gaussians with learned amplitudes?
      3 Gaussians:
      3 centers: p0, p1, p2 -> 3 * tanh(p)
      3 widths: p3, p4, p5 -> 0.1 + 2.9 * sigmoid(p)
      3 amplitudes: p6, p7, p8 -> 10 * tanh(p) (range -10 to 10)
      1 bias: p9 -> 10 * tanh(p)
      Total 10 params.
      
      This gives a sum of 3 localized bumps plus a bias. 
      This is a universal approximator (locally) and much more flexible than a 4th-order Fourier series for non-periodic or sharp features.
      BFGS should handle this if the landscape isn't too rugged. The sigmoid/tanh squashing helps keep parameters in valid ranges.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
        
    # Extract parameters
    p = np.asarray(params, dtype=np.float64)
    
    # 3 Centers, 3 Widths, 3 Amplitudes, 1 Bias
    # Centers: map to [-3, 3]
    c1 = 3.0 * np.tanh(p[0])
    c2 = 3.0 * np.tanh(p[1])
    c3 = 3.0 * np.tanh(p[2])
    
    # Widths: map to [0.1, 3.0] to avoid division by zero or too wide/narrow
    # Using sigmoid for positive values
    w1 = 0.1 + 2.9 * (1.0 / (1.0 + np.exp(-p[3])))
    w2 = 0.1 + 2.9 * (1.0 / (1.0 + np.exp(-p[4])))
    w3 = 0.1 + 2.9 * (1.0 / (1.0 + np.exp(-p[5])))
    
    # Amplitudes: map to [-10, 10] to allow significant scaling
    a1 = 10.0 * np.tanh(p[6])
    a2 = 10.0 * np.tanh(p[7])
    a3 = 10.0 * np.tanh(p[8])
    
    # Bias: map to [-10, 10]
    b = 10.0 * np.tanh(p[9])
    
    # Compute RBF terms
    # y = a1 * exp(-((x - c1)^2) / (2 * w1^2)) + a2 * ... + a3 * ... + b
    
    # Optimization: compute in vectorized manner
    # x is shape (N,)
    # c1, c2, c3 are scalars
    
    term1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    term2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    term3 = a3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    result = term1 + term2 + term3 + b
    
    return result

# EVOLVE-BLOCK-END
