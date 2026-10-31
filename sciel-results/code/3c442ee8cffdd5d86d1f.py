def fast_min_outer_side(params):
    # params: 33 (inner) + 3 (outer)
    inner_params = params[:33].reshape(11, 3)
    outer_center = params[33:35]
    outer_angle = params[35]
    
    # Vertices
    # angles_rad: (11,)
    # v_angles: (6,)
    # total: (11, 6)
    # cos/sin: (11, 6)
    # verts: (11, 6, 2)
    
    # Flatten verts: (66, 2)
    # Translate
    # Rotate
    # Project
    
    # Return scalar
