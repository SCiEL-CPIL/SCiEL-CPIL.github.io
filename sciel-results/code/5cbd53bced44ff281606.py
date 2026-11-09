"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a piecewise-linear basis (hat functions).
    This local basis is well-conditioned for BFGS optimization and can represent
    arbitrary continuous functions defined on the domain [-3, 3].
    
    We use 10 parameters corresponding to nodes at x = -3, -2.33, ..., 3.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Number of basis functions (parameters)
    n_basis = 10
    
    # Node positions uniformly spaced in [-3, 3]
    # x_k = -3 + k * (6 / (n_basis - 1))
    nodes = np.linspace(-3.0, 3.0, n_basis)
    
    # Compute prediction: y = sum(params[i] * hat_i(x))
    # hat_i(x) is a tent function centered at nodes[i] with support [nodes[i-1], nodes[i+1]]
    y = np.zeros_like(x, dtype=np.float64)
    
    for k in range(n_basis):
        c = nodes[k]
        # Left neighbor (or c if at boundary)
        left = nodes[k-1] if k > 0 else c
        # Right neighbor (or c if at boundary)
        right = nodes[k+1] if k < n_basis - 1 else c
        
        # Calculate tent shape
        # For left boundary (k=0), left=c, so (x-c)/(c-c) is 0/0. 
        # We handle boundaries by using one-sided linear ramps.
        if k == 0:
            # Ramp from -3 to -2.33 (approx)
            # Actually, standard hat functions on a grid usually imply:
            # h_0(x) = (x - x_1) / (x_0 - x_1) for x_0 <= x <= x_1
            # h_n-1(x) = (x_n - x) / (x_n - x_{n-1}) for x_{n-1} <= x <= x_n
            # h_k(x) = (x - x_{k-1}) / (x_k - x_{k-1}) for x_{k-1} <= x <= x_k
            #          (x_{k+1} - x) / (x_{k+1} - x_k) for x_k <= x <= x_{k+1}
            
            # Left edge: linear from 1 at x_0 to 0 at x_1
            # Slope is -1/(x_1-x_0)
            denom = nodes[1] - nodes[0]
            # x is in [-3, nodes[1]]
            # hat = 1 - (x - nodes[0]) / denom
            mask = x <= nodes[1]
            y += params[k] * (1.0 - (x - nodes[0]) / denom) * mask
            
        elif k == n_basis - 1:
            # Right edge: linear from 0 at x_{n-2} to 1 at x_{n-1}
            denom = nodes[-1] - nodes[-2]
            # x is in [nodes[-2], 3]
            # hat = (x - nodes[-2]) / denom
            mask = x >= nodes[-2]
            y += params[k] * ((x - nodes[-2]) / denom) * mask
            
        else:
            # Interior: two-sided tent
            left_denom = nodes[k] - nodes[k-1]
            right_denom = nodes[k+1] - nodes[k]
            
            # Left side: x in [nodes[k-1], nodes[k]]
            # hat = (x - nodes[k-1]) / left_denom
            mask_left = (x >= nodes[k-1]) & (x <= nodes[k])
            y += params[k] * ((x - nodes[k-1]) / left_denom) * mask_left
            
            # Right side: x in [nodes[k], nodes[k+1]]
            # hat = (nodes[k+1] - x) / right_denom
            mask_right = (x >= nodes[k]) & (x <= nodes[k+1])
            y += params[k] * ((nodes[k+1] - x) / right_denom) * mask_right
            
    return y
# EVOLVE-BLOCK-END
