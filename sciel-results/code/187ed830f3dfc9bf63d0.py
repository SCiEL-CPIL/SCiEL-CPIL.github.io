    def local_search(c, time_limit):
        c = c.copy()
        best_c = c.copy()
        base_vals = V @ c
        best_score = np.abs(base_vals).max() / sqrt_N
        
        while time.time() < time_limit:
            improved = False
            best_i = -1
            best_new_score = best_score
            best_new_vals = None
            
            for i in range(N):
                # Flip bit i
                delta = -2.0 * c[i] * V[:, i]
                new_vals = base_vals + delta
                new_score = np.abs(new_vals).max() / sqrt_N
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_i = i
                    best_new_vals = new_vals
            
            if best_i != -1:
                # Apply the best flip
                c[best_i] = -c[best_i]
                base_vals = best_new_vals
                best_score = best_new_score
                best_c = c.copy()
                improved = True
            else:
                # Local minimum reached. Perturb to escape.
                k_pert = np.random.randint(2, 5)
                idxs = np.random.choice(N, k_pert, replace=False)
                c[idxs] = -c[idxs]
                # Recalculate base_vals
                base_vals = V @ c
                # Update best_score? No, we keep the global best_c/best_score separate.
                # The current state 'c' is now perturbed.
                # We don't update best_c here because the perturbation likely increased the score.
                # We just continue searching from this new state.
        
        return best_c, best_score
