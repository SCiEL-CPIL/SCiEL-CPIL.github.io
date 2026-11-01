<<<<<<< SEARCH
        # First pass: L-BFGS-B optimization
        time_left = deadline - time.time() - 10
        if time_left < 3:
            break
        
        opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 12.0))
        
        # Second pass: Riemannian conjugate gradient on the sphere
        if time.time() < deadline - 8:
            rcg_time = min(deadline - time.time() - 3, 8.0)
            if rcg_time > 0.5:
                rcg_pts, rcg_energy = _riemannian_cg(opt_pts, time_limit=rcg_time)
                if rcg_energy < opt_energy:
                    opt_pts = rcg_pts
                    opt_energy = rcg_energy
        
        # Third pass: Gradient descent for fine-tuning
        if time.time() < deadline - 5:
            gd_time = min(deadline - time.time() - 2, 4.0)
            if gd_time > 0.5:
                gd_pts, gd_energy = _grad_descent(opt_pts, time_limit=gd_time)
                if gd_energy < opt_energy:
                    opt_pts = gd_pts
                    opt_energy = gd_energy
        
        # Fourth pass: Polish
        if time.time() < deadline - 3:
            polish_time = min(deadline - time.time() - 1, 3.0)
            if polish_time > 0.3:
                polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                if polished_energy < opt_energy:
                    opt_pts = polished_pts
                    opt_energy = polished_energy
        
        # Update best
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
=======
        # First pass: L-BFGS-B optimization
        time_left = deadline - time.time() - 10
        if time_left < 3:
            break
        
        opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 12.0))
        
        # Second pass: Riemannian conjugate gradient on the sphere
        if time.time() < deadline - 8:
            rcg_time = min(deadline - time.time() - 3, 8.0)
            if rcg_time > 0.5:
                rcg_pts, rcg_energy = _riemannian_cg(opt_pts, time_limit=rcg_time)
                if rcg_energy < opt_energy:
                    opt_pts = rcg_pts
                    opt_energy = rcg_energy
        
        # Third pass: Gradient descent for fine-tuning
        if time.time() < deadline - 5:
            gd_time = min(deadline - time.time() - 2, 4.0)
            if gd_time > 0.5:
                gd_pts, gd_energy = _grad_descent(opt_pts, time_limit=gd_time)
                if gd_energy < opt_energy:
                    opt_pts = gd_pts
                    opt_energy = gd_energy
        
        # Fourth pass: Polish
        if time.time() < deadline - 3:
            polish_time = min(deadline - time.time() - 1, 3.0)
            if polish_time > 0.3:
                polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                if polished_energy < opt_energy:
                    opt_pts = polished_pts
                    opt_energy = polished_energy
        
        # Fifth pass: Symmetry projection and re-optimization for icosahedral inits
        if name.startswith("icosahedral") and time.time() < deadline - 4:
            try:
                rotations = _icosahedral_symmetries()
                sym_pts = _symmetry_project(opt_pts, rotations)
                sym_energy = _log_energy_fast(sym_pts)
                
                if sym_energy < opt_energy:
                    # Re-optimize the symmetric configuration
                    sym_opt_pts, sym_opt_energy = _lbfgs_optimize(sym_pts, time_limit=min(time_left * 0.3, 6.0))
                    if sym_opt_energy < opt_energy:
                        opt_pts = sym_opt_pts
                        opt_energy = sym_opt_energy
                    
                    # Polish
                    if time.time() < deadline - 2:
                        pol_pts, pol_energy = _polish(opt_pts, time_limit=min(deadline - time.time() - 1, 3.0))
                        if pol_energy < opt_energy:
                            opt_pts = pol_pts
                            opt_energy = pol_energy
            except:
                pass
        
        # Update best
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
>>>>>>> REPLACE
