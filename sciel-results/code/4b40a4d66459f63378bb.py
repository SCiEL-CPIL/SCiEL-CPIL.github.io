<<<<<<< SEARCH
            if valid:
                # Try to snap to half-integers
                U_snapped = snap_to_halfint(U_list, tol=1e-4)
                V_snapped = snap_to_halfint(V_list, tol=1e-4)
                W_snapped = snap_to_halfint(W_list, tol=1e-4)
                
                valid_snapped, max_err_snapped = verify(U_snapped, V_snapped, W_snapped, tol=1e-10)
                if valid_snapped:
                    best = {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Use the non-snapped version if it's exact
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
=======
            if valid:
                # Try to snap to half-integers
                U_snapped = snap_to_halfint(U_list, tol=1e-4)
                V_snapped = snap_to_halfint(V_list, tol=1e-4)
                W_snapped = snap_to_halfint(W_list, tol=1e-4)
                
                valid_snapped, max_err_snapped = verify(U_snapped, V_snapped, W_snapped, tol=1e-10)
                if valid_snapped:
                    best = {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                    best_rank = target_rank
                    found = True
                    break
                # If snapping fails, do NOT update best with raw floats.
                # The evaluator requires integer/half-integer coefficients.
                # We keep the previous best (likely schoolbook or a previous valid snapped solution).
>>>>>>> REPLACE
