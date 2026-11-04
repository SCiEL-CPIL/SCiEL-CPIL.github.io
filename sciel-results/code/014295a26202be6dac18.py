<<<<<<< SEARCH
        # Multi-scale grid search with finer scales
        scales = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Perturbations
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            new_otheta = best_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True
=======
        # Multi-scale grid search with finer scales
        scales = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Perturbations - include diagonal moves for better coverage
                for dx in [-scale, -scale/2, 0, scale/2, scale]:
                    for dy in [-scale, -scale/2, 0, scale/2, scale]:
                        for da_deg in [-scale*10, -scale*5, 0, scale*5, scale*10]:
                            new_otheta = best_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-14:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True
>>>>>>> REPLACE

