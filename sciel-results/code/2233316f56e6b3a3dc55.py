import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Quadratic B-spline basis functions
    # 10 basis functions, degree 2
    # Knots: 3 repeated at start, 7 internal, 3 repeated at end
    # Total 13 knots
    
    internal_knots = np.linspace(-3.0, 3.0, 9)[1:-1]
    knots = np.concatenate((
        np.full(3, -3.0), 
        internal_knots, 
        np.full(3, 3.0)
    ))
    
    n_basis = 10
    degree = 2
    n_knots = len(knots)
    n_points = len(x)
    
    # Recursive computation of B-spline basis functions
    # Level 0: N_{i, 0}
    N_prev = np.zeros((n_knots - 1, n_points))
    for i in range(n_knots - 1):
        # Handle the last interval to include the right endpoint
        if i == n_knots - 2:
            mask = (x >= knots[i]) & (x <= knots[i+1])
        else:
            mask = (x >= knots[i]) & (x < knots[i+1])
        N_prev[i] = mask.astype(float)
        
    # Iterate for degree 1 to 2
    for d in range(1, degree + 1):
        N_curr = np.zeros((n_knots - 1 - d, n_points))
        for i in range(n_knots - 1 - d):
            # Term 1
            denom1 = knots[i + d] - knots[i]
            if denom1 > 0:
                term1 = (x - knots[i]) / denom1 * N_prev[i]
            else:
                term1 = np.zeros(n_points)
            
            # Term 2
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 > 0:
                term2 = (knots[i + d + 1] - x) / denom2 * N_prev[i + 1]
            else:
                term2 = np.zeros(n_points)
                
            N_curr[i] = term1 + term2
        
        N_prev = N_curr
        
    # N_prev now contains N_{i, 2} for i=0 to 9
    basis = N_prev[:n_basis]
    
    return params @ basis
