import numpy as np

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    
    T_rem = T - t
    if T_rem <= 0:
        # No time left, no quote or 0 depth
        return np.zeros(len(inventory)), np.zeros(len(inventory))
    
    # Coefficient C
    C = (2 * gamma * sigma**2) / np.sqrt(A * T_rem)
    
    q = inventory.astype(float)
    
    # Calculate depths
    # For q > 0, d_b > 0, d_a = 0
    # For q < 0, d_b = 0, d_a > 0
    # For q = 0, d_b = 0, d_a = 0
    
    # d_b = (1/k) * ln(1 + C*q) if q > 0, else 0
    # d_a = (1/k) * ln(1 - C*q) if q < 0, else 0
    # Note: 1 - C*q for q < 0 is 1 + C*|q|
    
    pos_mask = q > 0
    neg_mask = q < 0
    
    d_b = np.zeros(len(q))
    d_a = np.zeros(len(q))
    
    if np.any(pos_mask):
        args_b = 1 + C * q[pos_mask]
        # Ensure args_b > 0
        args_b = np.maximum(args_b, 1e-12)
        d_b[pos_mask] = (1.0 / k) * np.log(args_b)
        
    if np.any(neg_mask):
        args_a = 1 - C * q[neg_mask] # 1 + C*|q|
        args_a = np.maximum(args_a, 1e-12)
        d_a[neg_mask] = (1.0 / k) * np.log(args_a)
        
    # Clamp to 0 if negative (should not happen with max)
    d_b = np.maximum(d_b, 0)
    d_a = np.maximum(d_a, 0)
    
    return d_b, d_a
