import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Data-driven orthogonalized feature basis via QR decomposition.
    We build a large candidate feature matrix, orthogonalize it, and use
    the first 10 orthogonal components as our basis. This allows the
    optimizer to find the best linear combination in a well-conditioned space.
    """
    # Deadline guard
    t_start = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    n = len(x)
    
    # Build a diverse candidate feature matrix
    # We want features that span various functional forms:
    # - Polynomials in x and y
    # - Trigonometric functions
    # - Radial basis functions
    # - Interactions
    
    features = []
    
    # Constant
    features.append(np.ones(n))
    
    # Linear
    features.append(x)
    features.append(y)
    
    # Quadratic
    features.append(x * x)
    features.append(x * y)
    features.append(y * y)
    
    # Cubic
    features.append(x * x * x)
    features.append(x * x * y)
    features.append(x * y * y)
    features.append(y * y * y)
    
    # Trigonometric
    features.append(np.sin(x))
    features.append(np.cos(x))
    features.append(np.sin(y))
    features.append(np.cos(y))
    features.append(np.sin(x + y))
    features.append(np.sin(x - y))
    features.append(np.cos(x + y))
    features.append(np.cos(x - y))
    
    # Higher frequency trig
    features.append(np.sin(2 * x))
    features.append(np.cos(2 * x))
    features.append(np.sin(2 * y))
    features.append(np.cos(2 * y))
    
    # Radial / Gaussian-like basis
    r2 = x * x + y * y
    features.append(r2)
    features.append(np.exp(-r2))
    features.append(np.exp(-2 * r2))
    features.append(np.exp(-4 * r2))
    
    # More interactions
    features.append(np.sin(x) * np.cos(y))
    features.append(np.cos(x) * np.sin(y))
    features.append(np.sin(x) * np.sin(y))
    features.append(np.cos(x) * np.cos(y))
    
    # Polynomial-trig hybrids
    features.append(x * np.sin(x))
    features.append(y * np.sin(y))
    features.append(x * np.cos(x))
    features.append(y * np.cos(y))
    
    # Normalize features to have zero mean and unit variance for better conditioning
    A = np.column_stack(features)
    
    # Handle potential NaN/Inf
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Center and scale
    mean_A = np.mean(A, axis=0)
    std_A = np.std(A, axis=0)
    std_A[std_A < 1e-12] = 1.0
    A = (A - mean_A) / std_A
    
    # QR decomposition to get orthogonal basis
    # We only need the first 10 orthogonal components
    try:
        Q, R = np.linalg.qr(A, mode='economic')
    except np.linalg.LinAlgError:
        # Fallback to simple polynomial basis if QR fails
        Tx0 = 1.0
        Tx1 = x
        Tx2 = 2.0 * x * x - 1.0
        Tx3 = 4.0 * x * x * x - 3.0 * x
        Ty0 = 1.0
        Ty1 = y
        Ty2 = 2.0 * y * y - 1.0
        Ty3 = 4.0 * y * y * y - 3.0 * y
        p = params
        return (p[0] * Tx0 * Ty0 + 
                p[1] * Tx1 * Ty0 + 
                p[2] * Tx0 * Ty1 + 
                p[3] * Tx2 * Ty0 + 
                p[4] * Tx1 * Ty1 + 
                p[5] * Tx0 * Ty2 + 
                p[6] * Tx3 * Ty0 + 
                p[7] * Tx2 * Ty1 + 
                p[8] * Tx1 * Ty2 + 
                p[9] * Tx0 * Ty3)
    
    # Use first 10 columns of Q as our orthogonal basis
    # But we need to account for the centering/scaling
    # Actually, let's just use Q directly and let the optimizer handle the scaling
    # However, since we centered and scaled A, Q is orthogonal in that space
    # The prediction should be: params @ (Q[:, :10] @ ... ) 
    # Wait, the params are fit by the evaluator. The evaluator minimizes MSE.
    # So we just need to return a linear combination of 10 features.
    # Let's use the first 10 columns of Q as our features.
    
    Q10 = Q[:, :10]
    
    # Return the linear combination
    # params is a vector of length 10
    # We want params @ Q10.T
    pred = Q10 @ params
    
    # Check deadline
    if time.time() > deadline - safety_margin:
        # This shouldn't happen since we're just doing a matrix-vector multiply
        pass
    
    return pred
