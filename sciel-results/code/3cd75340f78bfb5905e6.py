<<<<<<< SEARCH
        t_curr = np.clip(state[0], 0.15, 0.85)
        w_curr = 2.0 * t_curr
        h_curr = 2.0 * (1.0 - t_curr)
        c_curr = state[1:].reshape(-1, 3)
        c_curr[:, 0] = np.clip(c_curr[:, 0], 0, w_curr)
        c_curr[:, 1] = np.clip(c_curr[:, 1], 0, h_curr)
        c_curr[:, 2] = np.clip(c_curr[:, 2], 1e-9, min(w_curr, h_curr)/2)
        
        if _strictly_valid(c_curr, w_curr, h_curr):
            val = float(np.sum(c_curr[:, 2]))
            if val > best_val:
                best_val = val
                best_state = state.copy()
=======
        t_curr = np.clip(state[0], 0.15, 0.85)
        w_curr = 2.0 * t_curr
        h_curr = 2.0 * (1.0 - t_curr)
        c_curr = state[1:].reshape(-1, 3)
        c_curr[:, 0] = np.clip(c_curr[:, 0], 0, w_curr)
        c_curr[:, 1] = np.clip(c_curr[:, 1], 0, h_curr)
        c_curr[:, 2] = np.clip(c_curr[:, 2], 1e-9, min(w_curr, h_curr)/2)
        
        # Enforce constraints
        c_curr = _enforce_constraints(c_curr, w_curr, h_curr)
        
        if _strictly_valid(c_curr, w_curr, h_curr):
            val = float(np.sum(c_curr[:, 2]))
            if val > best_val:
                best_val = val
                # Update state with enforced circles to keep consistency
                state_new = np.concatenate([[t_curr], c_curr.flatten()])
                best_state = state_new.copy()
>>>>>>> REPLACE
