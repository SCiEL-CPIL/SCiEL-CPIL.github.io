# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 basis functions to approximate an unknown function.
    Basis: orthogonal polynomials (shifted) + trigonometric + tanh.
    Features are standardized using analytical moments for Uniform[-3, 3] distribution
    to ensure consistency and improve optimizer conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # 1. Constant
    f0 = np.ones_like(x)
    # 2. Linear
    f1 = x
    # 3. Quadratic (shifted for orthogonality on [-3,3])
    f2 = x**2 - 3.0
    # 4. Cubic (shifted for orthogonality)
    f3 = x**3 - 2.7 * x
    # 5. Sine
    f4 = np.sin(x)
    # 6. Cosine
    f5 = np.cos(x)
    # 7. Sine 2x
    f6 = np.sin(2.0 * x)
    # 8. Cosine 2x
    f7 = np.cos(2.0 * x)
    # 9. Tanh
    f8 = np.tanh(x)
    # 10. Sine 3x
    f9 = np.sin(3.0 * x)

    # Analytical means and stds for U[-3, 3]
    # f0: 1, 0
    # f1: 0, sqrt(3)
    # f2: 0, sqrt(7.2)
    # f3: 0, sqrt(38.53)
    # f4: 0, sqrt(0.523)
    # f5: 0.047, sqrt(0.475)
    # f6: 0, sqrt(0.544)
    # f7: -0.0465, sqrt(0.455)
    # f8: 0, sqrt(0.668)
    # f9: 0, sqrt(0.562)

    stds = np.array([1.0, np.sqrt(3.0), np.sqrt(7.2), np.sqrt(38.53), np.sqrt(0.523), np.sqrt(0.475), np.sqrt(0.544), np.sqrt(0.455), np.sqrt(0.668), np.sqrt(0.562)])
    means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.047, 0.0, -0.0465, 0.0, 0.0])

    # Build feature matrix (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Standardize: (f - mean) / std
    # We keep f0 as 1 (bias term) or standardize it?
    # If we standardize f0: (1-0)/1 = 1. It stays 1.
    # So we can just apply the same operation to all.
    features = (features - means) / stds

    # Linear combination
    return features @ params

# EVOLVE-BLOCK-END
