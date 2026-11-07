import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a quadratic polynomial basis.
    
    Inputs are normalized to [0,1] range based on training ranges:
    b: [1, 4.998] -> b_norm = (b-1)/3.998
    s: [1, 4.999] -> s_norm = (s-1)/3.999
    temp: [1.002, 39.98] -> temp_norm = (temp-1.002)/38.978
    pH: [2, 12] -> pH_norm = (pH-2)/10
    
    The model is a quadratic polynomial in these normalized features,
    providing 10 coefficients for all terms up to degree 2.
    """
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 3.998
    s_norm = (s - 1.0) / 3.999
    temp_norm = (temp - 1.002) / 38.978
    pH_norm = (pH - 2.0) / 10.0
    
    # Build feature matrix: 1, b, s, t, p, b*s, b*t, b*p, s*t, s*p
    # This gives us 10 features for 10 parameters
    X = np.ones((b.shape[0], 10))
    X[:, 1] = b_norm
    X[:, 2] = s_norm
    X[:, 3] = temp_norm
    X[:, 4] = pH_norm
    X[:, 5] = b_norm * s_norm
    X[:, 6] = b_norm * temp_norm
    X[:, 7] = b_norm * pH_norm
    X[:, 8] = s_norm * temp_norm
    X[:, 9] = s_norm * pH_norm
    
    # Predict: linear combination of features with params as weights
    pred = X @ params
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
