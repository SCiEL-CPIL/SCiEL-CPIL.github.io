import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale inputs to roughly [-1, 1] or [0, 1] range to improve conditioning
    # for the BFGS optimizer.
    # Ranges given: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Let's use max abs values for scaling to center around 0 roughly.
    scale_x = 0.6
    scale_v = 0.3
    
    X = x / scale_x
    V = v / scale_v
    
    # Compute powers of scaled variables
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Basis terms:
    # 1. Constant
    # 2. Linear X
    # 3. Linear V
    # 4. Quadratic X
    # 5. Cubic X (Duffing term)
    # 6. Quadratic V
    # 7. Cubic V (Cubic damping)
    # 8. Cross XV
    # 9. Cross X2V
    # 10. Cross XV2
    
    term1 = 1.0
    term2 = X
    term3 = V
    term4 = X2
    term5 = X3
    term6 = V2
    term7 = V3
    term8 = XV
    term9 = X2V
    term10 = XV2
    
    return (params[0] * term1 +
            params[1] * term2 +
            params[2] * term3 +
            params[3] * term4 +
            params[4] * term5 +
            params[5] * term6 +
            params[6] * term7 +
            params[7] * term8 +
            params[8] * term9 +
            params[9] * term10)
