# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid orthogonal basis with fixed normalization.
    
    The basis consists of low-degree Chebyshev polynomials (scaled to [-1, 1])
    and low-frequency trigonometric functions orthogonal on [-2, 2].
    Each basis function is scaled by its standard deviation over the uniform
    distribution on [-2, 2] to improve the condition number for BFGS.
    """
    # 1. Define raw basis functions
    
    # Chebyshev polynomials of the first kind T_n(t) where t = x/2
    t = x / 2.0
    f0 = np.ones_like(x)
    f1 = t
    f2 = 2.0 * t**2 - 1.0
    f3 = 4.0 * t**3 - 3.0 * t
    
    # Trigonometric functions orthogonal on [-2, 2]
    f4 = np.sin(0.5 * np.pi * x)
    f5 = np.cos(0.5 * np.pi * x)
    f6 = np.sin(np.pi * x)
    f7 = np.cos(np.pi * x)
    f8 = np.sin(1.5 * np.pi * x)
    f9 = np.cos(1.5 * np.pi * x)
    
    # 2. Normalize basis functions
    # Pre-computed standard deviations for x ~ Uniform[-2, 2]
    # f0: 1 (keep as is)
    # f1: sqrt(1/3) -> scale by sqrt(3)
    # f2: sqrt(16/45) -> scale by 4/(3*sqrt(5))
    # f3: sqrt(17/35) -> scale by sqrt(35/17)
    # f4-f9: sqrt(0.5) -> scale by sqrt(2)
    
    scale = np.array([
        1.0,
        np.sqrt(3.0),
        4.0 / (3.0 * np.sqrt(5.0)),
        np.sqrt(35.0 / 17.0),
        np.sqrt(2.0),
        np.sqrt(2.0),
        np.sqrt(2.0),
        np.sqrt(2.0),
        np.sqrt(2.0),
        np.sqrt(2.0)
    ])
    
    basis = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # Apply scaling
    basis_scaled = basis * scale
    
    # 3. Compute prediction
    return basis_scaled @ params
# EVOLVE-BLOCK-END
