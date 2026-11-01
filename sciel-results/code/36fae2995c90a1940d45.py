<<<<<<< SEARCH
    # Phase 4: Fine-grid polishing with ultra-fine grid
    if best_points is not None and time.time() < deadline - 15.0:
        lat_uf = np.linspace(-np.pi/2, np.pi/2, 100)
        lon_uf = np.linspace(0, 2*np.pi, 200, endpoint=False)
        LA_uf, LO_uf = np.meshgrid(lat_uf, lon_uf, indexing='ij')
        z_guf = np.sin(LA_uf)
        x_guf = np.cos(LA_uf) * np.cos(LO_uf)
        y_guf = np.sin(LA_uf) * np.sin(LO_uf)
        grid_ultra_fine = np.column_stack([x_guf.ravel(), y_guf.ravel(), z_guf.ravel()])
        
        beta_fine = [40.0, 60.0, 80.0]
        alpha_fine = [0.008, 0.005, 0.003]
        decay_fine = [0.996, 0.994, 0.992]
        iter_fine = [80, 60, 50]
        
        for idx, (b, a0, dcy, n_it) in enumerate(zip(beta_fine, alpha_fine, decay_fine, iter_fine)):
            if time.time() > deadline - 8.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, a0, dcy, n_it, beta=b, grid_pts=grid_ultra_fine)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
=======
    # Phase 4: Fine-grid polishing with ultra-fine grid
    if best_points is not None and time.time() < deadline - 12.0:
        lat_uf = np.linspace(-np.pi/2, np.pi/2, 100)
        lon_uf = np.linspace(0, 2*np.pi, 200, endpoint=False)
        LA_uf, LO_uf = np.meshgrid(lat_uf, lon_uf, indexing='ij')
        z_guf = np.sin(LA_uf)
        x_guf = np.cos(LA_uf) * np.cos(LO_uf)
        y_guf = np.sin(LA_uf) * np.sin(LO_uf)
        grid_ultra_fine = np.column_stack([x_guf.ravel(), y_guf.ravel(), z_guf.ravel()])
        
        beta_fine = [50.0, 80.0, 120.0]
        alpha_fine = [0.006, 0.004, 0.002]
        decay_fine = [0.996, 0.995, 0.994]
        iter_fine = [60, 50, 40]
        
        for idx, (b, a0, dcy, n_it) in enumerate(zip(beta_fine, alpha_fine, decay_fine, iter_fine)):
            if time.time() > deadline - 6.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, a0, dcy, n_it, beta=b, grid_pts=grid_ultra_fine)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
>>>>>>> REPLACE

