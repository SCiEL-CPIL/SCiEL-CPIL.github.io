<<<<<<< SEARCH
    def refine_outer(c_coords, c_angles_rad):
        best_side = float('inf')
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                for da_deg in np.linspace(-15, 15, 31):
                    test_ocx = ocx + dx
                    test_ocy = ocy + dy
                    test_otheta = otheta + math.radians(da_deg)
                    side = compute_outer_side_vectorized(c_coords, c_angles_rad, test_ocx, test_ocy, test_otheta)
                    if side < best_side:
                        best_side = side
                        best_ocx = test_ocx
                        best_ocy = test_ocy
                        best_otheta = test_otheta
                        
        return best_ocx, best_ocy, best_otheta, best_side

    ocx, ocy, otheta, final_side = refine_outer(c_coords, c_angles_rad)
    
    inner = [[c_coords[i, 0], c_coords[i, 1], c_angles_deg[i]] for i in range(12)]
    outer_center = (ocx, ocy)
    outer_side = final_side
    outer_angle = math.degrees(otheta)
    
    return inner, outer_center, outer_side, outer_angle
=======
    def refine_outer(c_coords, c_angles_rad):
        best_side = float('inf')
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        # Coarse grid search for outer hexagon center and angle
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                for da_deg in np.linspace(-15, 15, 31):
                    test_ocx = ocx + dx
                    test_ocy = ocy + dy
                    test_otheta = otheta + math.radians(da_deg)
                    side = compute_outer_side_vectorized(c_coords, c_angles_rad, test_ocx, test_ocy, test_otheta)
                    if side < best_side:
                        best_side = side
                        best_ocx = test_ocx
                        best_ocy = test_ocy
                        best_otheta = test_otheta
                        
        return best_ocx, best_ocy, best_otheta, best_side

    ocx, ocy, otheta, final_side = refine_outer(c_coords, c_angles_rad)
    
    inner = [[c_coords[i, 0], c_coords[i, 1], c_angles_deg[i]] for i in range(12)]
    outer_center = (ocx, ocy)
    outer_side = final_side
    outer_angle = math.degrees(otheta)
    
    return inner, outer_center, outer_side, outer_angle
>>>>>>> REPLACE
