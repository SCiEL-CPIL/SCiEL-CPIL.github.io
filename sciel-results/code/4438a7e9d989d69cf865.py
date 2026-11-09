import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    The optimal measure is derived to be a 2-point distribution on {0, M} with 
    probabilities 1-p and p, where p = (1 + sqrt(33)) / 16.
    The scale M does not affect the probability, so we choose M=1.
    """
    # Optimal probability for the point M
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    
    # Support points: 0 and 1
    support = np.array([0.0, 1.0], dtype=np.float64)
    
    # Weights: 1-p and p
    weights = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    
    return support, weights
