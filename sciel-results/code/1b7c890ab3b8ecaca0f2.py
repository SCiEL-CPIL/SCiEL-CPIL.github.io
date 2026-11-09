import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Cosine Series basis.
    This basis is orthogonal on [-1, 1] and well-suited for even and periodic functions,
    which are likely candidates if a linear model scores 1.0 (predicts mean).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basis functions
    # We use frequencies pi, 2pi, 3pi
    c_x1 = np.cos(np.pi * x)
    c_y1 = np.cos(np.pi * y)
    c_x2 = np.cos(2 * np.pi * x)
    c_y2 = np.cos(2 * np.pi * y)
    c_x3 = np.cos(3 * np.pi * x)
    
    # Construct prediction
    # 1. Constant
    # 2. cos(pi x)
    # 3. cos(pi y)
    # 4. cos(2 pi x)
    # 5. cos(2 pi y)
    # 6. cos(pi x) cos(pi y)
    # 7. cos(2 pi x) cos(2 pi y)
    # 8. cos(pi x) cos(2 pi y)
    # 9. cos(2 pi x) cos(pi y)
    # 10. cos(3 pi x)
    
    pred = (p0 * np.ones_like(x) + 
            p1 * c_x1 + 
            p2 * c_y1 + 
            p3 * c_x2 + 
            p4 * c_y2 + 
            p5 * c_x1 * c_y1 + 
            p6 * c_x2 * c_y2 + 
            p7 * c_x1 * c_y2 + 
            p8 * c_x2 * c_y1 + 
            p9 * c_x3)
            
    return pred
