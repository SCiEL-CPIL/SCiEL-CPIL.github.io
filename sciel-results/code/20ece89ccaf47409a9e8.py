"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Build a rich basis of candidate features from x and y.
    # We use a combination of polynomial, trigonometric, and rational-like terms
    # to cover a wide space of possible target functions.
    #
    # Features:
    # 1. x
    # 2. y
    # 3. x*y
    # 4. x**2
    # 5. y**2
    # 6. x**3
    # 7. y**3
    # 8. x**2*y
    # 9. x*y**2
    # 10. x**4
    # 11. y**4
    # 12. sin(x)
    # 13. sin(y)
    # 14. cos(x)
    # 15. cos(y)
    # 16. sin(x)*cos(y)
    # 17. cos(x)*sin(y)
    # 18. exp(-x**2)
    # 19. exp(-y**2)
    # 20. exp(-(x**2 + y**2))
    # 21. 1/(1 + x**2)
    # 22. 1/(1 + y**2)
    # 23. 1/(1 + x**2 + y**2)
    # 24. log(1 + x**2)
    # 25. log(1 + y**2)
    # 26. |x|
    # 27. |y|
    # 28. |x*y|
    # 29. sign(x)
    # 30. sign(y)
    # 31. 1 (bias)
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Ensure params has at least 31 elements; if not, pad or truncate
    p = np.asarray(params, dtype=float)
    n_params = len(p)
    
    # Build feature matrix
    # We'll construct up to 31 features
    f = []
    f.append(x)
    f.append(y)
    f.append(x * y)
    f.append(x**2)
    f.append(y**2)
    f.append(x**3)
    f.append(y**3)
    f.append(x**2 * y)
    f.append(x * y**2)
    f.append(x**4)
    f.append(y**4)
    f.append(np.sin(x))
    f.append(np.sin(y))
    f.append(np.cos(x))
    f.append(np.cos(y))
    f.append(np.sin(x) * np.cos(y))
    f.append(np.cos(x) * np.sin(y))
    f.append(np.exp(-x**2))
    f.append(np.exp(-y**2))
    f.append(np.exp(-(x**2 + y**2)))
    f.append(1.0 / (1.0 + x**2))
    f.append(1.0 / (1.0 + y**2))
    f.append(1.0 / (1.0 + x**2 + y**2))
    f.append(np.log1p(x**2))
    f.append(np.log1p(y**2))
    f.append(np.abs(x))
    f.append(np.abs(y))
    f.append(np.abs(x * y))
    f.append(np.sign(x))
    f.append(np.sign(y))
    f.append(np.ones_like(x))
    
    # If we have more params than features, just use the first len(f) params
    # If we have fewer params, pad with zeros
    if n_params >= len(f):
        coeffs = p[:len(f)]
    else:
        coeffs = np.zeros(len(f))
        coeffs[:n_params] = p[:n_params]
    
    # Compute prediction as linear combination
    pred = np.zeros_like(x)
    for i, feat in enumerate(f):
        pred += coeffs[i] * feat
    
    return pred
# EVOLVE-BLOCK-END
