"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier Series Model.
    
    Structural change: Switch from RBF (non-linear parameters) to a Fourier
    series (linear parameters). The MSE landscape is convex with respect to
    the amplitudes, allowing BFGS to converge reliably and quickly to the
    global optimum.
    
    Model:
      y(x) = sum_{k=1}^{5} a_k * sin(k * pi/2 * x) + sum_{k=1}^{5} b_k * cos(k * pi/2 * x)
    
    Params layout (length 10):
      - params[0:5] : Sine amplitudes a_1 to a_5
      - params[5:10] : Cosine amplitudes b_1 to b_5
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure valid shapes
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Frequencies: k * pi/2 for k=1..5
    # This gives periods of 4, 2, 4/3, 1, 0.8
    # Covering a good range of scales on [-2, 2]
    k = np.arange(1, 6)
    freqs = k * (np.pi / 2.0)
    
    # Precompute basis matrices
    # x is (N,), freqs is (5,)
    # We want (N, 5) for sin and cos
    X_freq = x[:, np.newaxis] * freqs[np.newaxis, :]
    
    sin_basis = np.sin(X_freq)
    cos_basis = np.cos(X_freq)
    
    a = params[0:5]
    b = params[5:10]
    
    # Linear combination
    result = sin_basis @ a + cos_basis @ b
    
    return result
# EVOLVE-BLOCK-END
