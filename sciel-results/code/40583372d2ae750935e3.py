# Precompute angles and support function values for the outer hexagon
# This avoids recomputing them in every objective evaluation
ALPHAS = np.linspace(0, 2*np.pi, 720, endpoint=False)
K_VERTS = (np.arange(6) * np.pi / 3).reshape(-1, 1) # (6, 1)

def h_unit(alpha):
    # Support function of unit hexagon (side 1, vertex at distance 1)
    # alpha: array of angles
    # Returns max distance from center to edge in direction alpha
    # Vectorized: alpha (M, 1) - K_VERTS (1, 6) -> (M, 6)
    # Max over axis 1 -> (M,)
    alpha_arr = np.atleast_1d(alpha).reshape(-1, 1)
    return np.max(np.cos(alpha_arr - K_VERTS), axis=1)

# Precompute h_unit for the outer hexagon directions (which are fixed at 0 rotation in the objective formulation)
# Note: The outer hexagon is assumed to be at angle 0 in the constraint derivation 
# c_i . u(alpha) + h_unit(alpha - phi_i) <= S * h_unit(alpha)
# The term h_unit(alpha) on the RHS is independent of the inner hexagons.
H_CONT = h_unit(ALPHAS) # Shape (720,)

def calculate_side_length(centers, angles):
    # centers: (N, 2)
    # angles: (N,)
    # Returns minimum S
    max_s = 0.0
    
    # Precompute u components for all alphas
    cos_alphas = np.cos(ALPHAS)
    sin_alphas = np.sin(ALPHAS)
    
    for i in range(N):
        cx, cy = centers[i]
        phi = angles[i]
        
        # proj = cx * cos(alpha) + cy * sin(alpha)
        proj = cx * cos_alphas + cy * sin_alphas
        
        # h_in = h_unit(alpha - phi)
        h_in = h_unit(ALPHAS - phi)
        
        # s_req = (proj + h_in) / H_CONT
        # H_CONT is always > 0 for hexagon support function
        s_req = (proj + h_in) / H_CONT
        
        current_max = np.max(s_req)
        if current_max > max_s:
            max_s = current_max
            
    return max_s
