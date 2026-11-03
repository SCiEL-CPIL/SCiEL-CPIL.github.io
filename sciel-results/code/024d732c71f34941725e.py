<<<<<<< SEARCH
    # Construct Return Values
    inner = []
    for i in range(12):
        inner.append([best_result[2*i], best_result[2*i+1], 30.0])
        
    outer_center = (best_result[24], best_result[25])
    outer_side = best_score
    outer_angle = best_result[26]
    
    return inner, outer_center, outer_side, outer_angle
=======
    # Construct Return Values
    inner = []
    for i in range(12):
        inner.append([best_result[2*i], best_result[2*i+1], 30.0])
        
    outer_center = (best_result[24], best_result[25])
    # Re-calculate side length from best_result to ensure consistency and add safety margin
    # The objective function already adds 1e-7, but let's be extra safe
    c_centers = []
    for i in range(12):
        c_centers.append(np.array([best_result[2*i], best_result[2*i+1]]))
    ox = best_result[24]
    oy = best_result[25]
    o_phi = best_result[26]
    
    max_proj = 0.0
    for k in range(6):
        n_angle = o_phi + math.pi/6 + k * math.pi/3
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        for c in c_centers:
            for m in range(6):
                theta = math.radians(30.0) + m * math.pi / 3
                vx = c[0] + math.cos(theta)
                vy = c[1] + math.sin(theta)
                proj = (vx - ox) * nx + (vy - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    outer_side = 2.0 * (max_proj + 1e-6) / math.sqrt(3)
    outer_angle = best_result[26]
    
    return inner, outer_center, outer_side, outer_angle
>>>>>>> REPLACE
