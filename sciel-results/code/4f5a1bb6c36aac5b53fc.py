<<<<<<< SEARCH
    def find_best_container(vertices_list):
        """Find the minimal enclosing hexagon for a set of vertices."""
        # Initial guess: centroid, 0 angle
        cx, cy = 0.0, 0.0
        angle = 0.0
        
        def cont_obj(params_c):
            ocx, ocy, oangle = params_c
            return compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Optimize container parameters
        res = minimize(cont_obj, [cx, cy, angle], method='Powell', options={'maxiter': 500, 'maxfev': 2000, 'xtol': 1e-6, 'ftol': 1e-6})
        return res.x, res.fun
=======
    def find_best_container(vertices_list):
        """Find the minimal enclosing hexagon for a set of vertices."""
        # Compute centroid for better initial guess
        all_verts = np.vstack(vertices_list)
        cx, cy = np.mean(all_verts, axis=0)
        
        best_c = None
        best_s = float('inf')
        
        # Try multiple initial angles
        for init_angle in [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            for init_cx in [cx, cx + 0.1, cx - 0.1]:
                for init_cy in [cy, cy + 0.1, cy - 0.1]:
                    def cont_obj(params_c):
                        ocx, ocy, oangle = params_c
                        return compute_S_for_container(vertices_list, ocx, ocy, oangle)
                    
                    try:
                        res = minimize(cont_obj, [init_cx, init_cy, init_angle], method='Powell', 
                                       options={'maxiter': 200, 'maxfev': 1000, 'xtol': 1e-6, 'ftol': 1e-6})
                        if res.fun < best_s:
                            best_s = res.fun
                            best_c = res.x
                    except Exception:
                        pass
        
        return best_c, best_s
>>>>>>> REPLACE
