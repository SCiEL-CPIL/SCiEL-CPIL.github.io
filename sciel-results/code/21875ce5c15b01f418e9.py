import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model:
    - Linear terms
    - Pairwise interactions
    - Gaussian RBFs
    - Periodic term
    
    This structural change moves from a pure polynomial model to a 
    mixed functional family, allowing the capture of non-polynomial 
    features like exponentials and sine waves which are common in 
    symbolic regression tasks.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-4. Linear
    b1 = x
    b2 = y
    b3 = z
    
    # 5-7. Pairwise Interactions
    b4 = x * y
    b5 = x * z
    b6 = y * z
    
    # 8. Gaussian RBF centered at (1, 1, 1)
    # Width chosen to cover the domain [0.5, 2]
    dist1 = (x - 1.0)**2 + (y - 1.0)**2 + (z - 1.0)**2
    b7 = np.exp(-dist1)
    
    # 9. Sine of sum
    b8 = np.sin(x + y + z)
    
    # 10. Gaussian RBF centered at (1.5, 1.5, 1.5)
    dist2 = (x - 1.5)**2 + (y - 1.5)**2 + (z - 1.5)**2
    b9 = np.exp(-dist2)
    
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + 
            p4 * b4 + p5 * b5 + p6 * b6 + p7 * b7 + 
            p8 * b8 + p9 * b9)
            
    return pred
