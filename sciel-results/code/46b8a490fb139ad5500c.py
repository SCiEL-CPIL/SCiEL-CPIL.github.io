import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a local piecewise-linear basis (linear splines).
    
    Unlike global polynomials, local hat functions have disjoint or nearly disjoint
    supports. This decouples the parameters, making the design matrix sparse and 
    well-conditioned. BFGS can therefore update individual coefficients independently 
    without interference, reliably fitting local features of the hidden function.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    p = params[:10]
    
    # Ensure finite output
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Define a fine grid of 10 nodes over [-3, 3] to maximize local resolution
    # with the available 10 parameters.
    nodes = np.linspace(-3.0, 3.0, 10)
    
    # Prediction is the sum of hat functions scaled by parameters
    y_pred = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        xi = nodes[i]
        
        # Determine the support of the i-th hat function
        # For interior nodes, support is between the previous and next node
        # For boundary nodes, support is one-sided
        if i == 0:
            left = nodes[0]
            right = nodes[1]
        elif i == 9:
            left = nodes[8]
            right = nodes[9]
        else:
            left = nodes[i-1]
            right = nodes[i+1]
        
        # Calculate hat values
        # Left slope: (x - left) / (xi - left)
        # Right slope: (right - x) / (right - xi)
        
        # We can compute this vectorized for efficiency
        # h_i(x) = max(0, min( (x-left)/(xi-left), (right-x)/(right-xi) ))
        
        # Handle division by zero if grid points coincide (not the case here, but safe)
        if xi > left:
            slope_l = 1.0 / (xi - left)
        else:
            slope_l = 0.0
            
        if right > xi:
            slope_r = 1.0 / (right - xi)
        else:
            slope_r = 0.0
            
        # Linear ramps
        ramp_l = (x - left) * slope_l
        ramp_r = (right - x) * slope_r
        
        # The hat is the minimum of the two ramps, clamped to [0, 1]
        # Actually, for a standard hat function centered at xi with support [left, right]:
        # h(x) = (x-left)/(xi-left) for left <= x <= xi
        # h(x) = (right-x)/(right-xi) for xi <= x <= right
        # h(x) = 0 otherwise
        
        # Using min(ramp_l, ramp_r) works if the ramps cross at xi.
        # At x=xi, ramp_l = (xi-left)/(xi-left) = 1, ramp_r = (right-xi)/(right-xi) = 1.
        # For x < xi, ramp_l < 1 and ramp_r > 1 (since x < xi => right-x > right-xi). 
        # Wait, if x < left, ramp_l < 0. If x > right, ramp_r < 0.
        # So min(ramp_l, ramp_r) is exactly the hat shape, but we must clamp negatives to 0.
        
        hat = np.minimum(ramp_l, ramp_r)
        hat = np.maximum(hat, 0.0)
        
        y_pred += p[i] * hat
        
    return y_pred
