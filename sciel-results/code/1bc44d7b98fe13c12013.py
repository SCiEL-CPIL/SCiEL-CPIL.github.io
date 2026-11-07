"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for safe operations, though it usually is
    # x shape: (N,)
    
    # Compute basis functions
    # We use 10 basis functions to match the 10 parameters
    
    # 1. Constant
    f0 = np.ones_like(x)
    # 2. Sine
    f1 = np.sin(x)
    # 3. Cosine
    f2 = np.cos(x)
    # 4. Sine 2x
    f3 = np.sin(2 * x)
    # 5. Cosine 2x
    f4 = np.cos(2 * x)
    # 6. Sine 3x
    f5 = np.sin(3 * x)
    # 7. Cosine 3x
    f6 = np.cos(3 * x)
    # 8. Sine 4x
    f7 = np.sin(4 * x)
    # 9. Cosine 4x
    f8 = np.cos(4 * x)
    # 10. Tanh
    f9 = np.tanh(x)
    
    # Stack features into a matrix of shape (N, 10)
    # Columns correspond to params[0] ... params[9]
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    # params shape: (10,)
    # features shape: (N, 10)
    # Result shape: (N,)
    return features.dot(params)

# EVOLVE-BLOCK-END
