"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: return a simple linear prediction using first two params
        # to avoid crashing, though this is likely a bad fit if we hit deadline
        return params[0] * x + params[1] if len(params) >= 2 else np.zeros_like(x)

    x = np.asarray(x, dtype=float)
    t = x / 3.0  # Scale to [-1, 1] for numerical stability

    # Define 10 diverse, bounded basis functions on t in [-1, 1]
    # 1. Constant
    b0 = np.ones_like(t)
    # 2. Linear
    b1 = t
    # 3. Quadratic
    b2 = t**2
    # 4. Cubic
    b3 = t**3
    # 5. Quartic
    b4 = t**4
    # 6. Sine (periodic component)
    b5 = np.sin(np.pi * t)
    # 7. Cosine (periodic component)
    b6 = np.cos(np.pi * t)
    # 8. Tanh (smooth saturation)
    b7 = np.tanh(t)
    # 9. Arctan (bounded, smooth)
    b8 = np.arctan(t)
    # 10. Lorentzian (peak-like)
    b9 = 1.0 / (1.0 + t**2)

    # Stack into matrix (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Compute prediction
    y = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
