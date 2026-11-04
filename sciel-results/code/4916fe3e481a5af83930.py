<<<<<<< SEARCH
    # --- FINAL POLISH: Optimize outer center and angle ---
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        orient_idx = best_params[2*N:].astype(int)
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            side = calc_outer_side(best_params, outer_center=oc, outer_angle=np.degrees(oa))
            return side

        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_outer.fun < best_side:
                # Verify no overlap with new center? Overlap is independent of outer center.
                # But we need to make sure the inner hexagons are still inside.
                # calc_outer_side already checks this.
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            pass
=======
    # --- FINAL POLISH: Optimize outer center and angle with fine grid + local refine ---
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        orient_idx = best_params[2*N:].astype(int)
        
        # First, do a fine grid search over outer angle (0 to 60 deg, step 0.1 deg = 600 points)
        # and outer center (small neighborhood around centroid)
        centroid = np.mean(pos, axis=0)
        
        best_side_refined = best_side
        best_oc_refined = np.array(best_outer_center)
        best_ao_refined = best_outer_angle
        
        # Coarse angle search: 0 to 60 deg, step 1 deg
        for ao_deg in range(0, 61):
            # Center search: small grid around centroid
            for dx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                for dy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                    oc = [centroid[0] + dx, centroid[1] + dy]
                    side = calc_outer_side(best_params, outer_center=oc, outer_angle=float(ao_deg))
                    if side < best_side_refined:
                        best_side_refined = side
                        best_oc_refined = np.array(oc)
                        best_ao_refined = float(ao_deg)
        
        # Fine angle search around best coarse angle
        if time_left() > 15:
            ao_center = best_ao_refined
            for da in np.arange(-1.0, 1.01, 0.05):
                ao_deg = ao_center + da
                # Center search: finer grid
                for dx in np.arange(-0.3, 0.31, 0.05):
                    for dy in np.arange(-0.3, 0.31, 0.05):
                        oc = [centroid[0] + dx, centroid[1] + dy]
                        side = calc_outer_side(best_params, outer_center=oc, outer_angle=float(ao_deg))
                        if side < best_side_refined:
                            best_side_refined = side
                            best_oc_refined = np.array(oc)
                            best_ao_refined = float(ao_deg)
        
        # Local Nelder-Mead polish on (cx, cy, angle)
        if time_left() > 10:
            def outer_obj(outer_params):
                oc = outer_params[:2]
                oa = outer_params[2]
                side = calc_outer_side(best_params, outer_center=oc, outer_angle=float(oa))
                return side

            x0_outer = np.array([best_oc_refined[0], best_oc_refined[1], best_ao_refined])
            
            try:
                res_outer = minimize(
                    outer_obj, 
                    x0_outer, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                if res_outer.fun < best_side_refined:
                    best_side_refined = res_outer.fun
                    best_oc_refined = res_outer.x[:2]
                    best_ao_refined = float(res_outer.x[2])
            except:
                pass
        
        best_side = best_side_refined
        best_outer_center = list(best_oc_refined)
        best_outer_angle = best_ao_refined
>>>>>>> REPLACE

