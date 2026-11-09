import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Structural Change: Use a Tanh-based basis (Neural Network-like single layer)
    # This allows for non-linear, bounded features that can approximate a wide variety of functions
    # better than low-degree polynomials if the function has saturation or complex interactions.
    # We use linear combinations of inputs inside tanh to capture interactions.
    
    # Scale inputs to roughly [-1, 1] to keep tanh in sensitive region
    # Inputs are in [0.5, 2]. Center 1.25, Scale 1.5?
    # Let's just use the raw inputs or slightly scaled.
    # tanh(1) ~ 0.76, tanh(2) ~ 0.96.
    # To make it more sensitive, we can scale by 1 or 2.
    # Let's use scale 1.0 for simplicity, or maybe 1.5 to utilize the range.
    # Let's try scaling by 1.0 first.
    
    # Basis terms:
    # 0: 1 (bias)
    # 1: tanh(x)
    # 2: tanh(y)
    # 3: tanh(z)
    # 4: tanh(x + y)
    # 5: tanh(x + z)
    # 6: tanh(y + z)
    # 7: tanh(x - y)
    # 8: tanh(x - z)
    # 9: tanh(y - z)
    
    # Alternatively, include some polynomial terms if tanh is not enough?
    # Let's stick to a pure Tanh basis for a clear structural shift.
    
    basis = np.column_stack((
        np.ones_like(x),
        np.tanh(x),
        np.tanh(y),
        np.tanh(z),
        np.tanh(x + y),
        np.tanh(x + z),
        np.tanh(y + z),
        np.tanh(x - y),
        np.tanh(x - z),
        np.tanh(y - z)
    ))
    
    result = basis @ p
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
