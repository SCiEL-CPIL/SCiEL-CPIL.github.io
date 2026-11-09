"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a single-layer neural network with ReLU activations.
    
    The model uses a fixed set of hidden units (ReLU features) computed from centered inputs.
    The 10 parameters are the linear output weights connecting these features to the output.
    This makes the model linear in its trainable parameters, ensuring a convex loss landscape
    for BFGS optimization, while retaining capacity to approximate complex non-linear interactions.
    """
    # Center inputs for numerical stability
    b_c = b - 3.0
    s_c = s - 10.0
    t_c = temp - 30.0
    p_c = pH - 7.0
    
    p = params
    
    # Create 10 ReLU features from linear combinations of centered inputs
    # Each feature is max(0, w_i . x + b_i) where x = [b_c, s_c, t_c, p_c]
    # We use fixed, diverse weight vectors to capture different aspects of the input space
    
    # Feature 1: Positive density effect
    f1 = np.maximum(0, 1.0 * b_c)
    
    # Feature 2: Positive substrate effect
    f2 = np.maximum(0, 1.0 * s_c)
    
    # Feature 3: Temperature deviation (positive side)
    f3 = np.maximum(0, 1.0 * t_c)
    
    # Feature 4: pH deviation (positive side)
    f4 = np.maximum(0, 1.0 * p_c)
    
    # Feature 5: Combined density and substrate
    f5 = np.maximum(0, 0.5 * b_c + 0.5 * s_c)
    
    # Feature 6: Combined temperature and pH
    f6 = np.maximum(0, 0.5 * t_c + 0.5 * p_c)
    
    # Feature 7: Density-temperature interaction
    f7 = np.maximum(0, 0.7 * b_c + 0.3 * t_c)
    
    # Feature 8: Substrate-pH interaction
    f8 = np.maximum(0, 0.3 * s_c + 0.7 * p_c)
    
    # Feature 9: Negative density effect (for inhibition)
    f9 = np.maximum(0, -1.0 * b_c)
    
    # Feature 10: Constant bias feature (always 1, allows for offset)
    f10 = np.ones_like(b_c)
    
    # Linear combination of features with learnable weights
    pred = (
        p[0] * f1 +
        p[1] * f2 +
        p[2] * f3 +
        p[3] * f4 +
        p[4] * f5 +
        p[5] * f6 +
        p[6] * f7 +
        p[7] * f8 +
        p[8] * f9 +
        p[9] * f10
    )
    
    return pred
# EVOLVE-BLOCK-END
