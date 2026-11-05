        while time.time() - start_time < max_time:
            G = (P.T @ S).A.flatten()
            deltas = -4 * signs * G + 4 * counts
            best_idx = np.argmin(deltas)
            
            if deltas[best_idx] < 0:
                old_sign = signs[best_idx]
                signs[best_idx] = -old_sign
                S[pos_to_aps[best_idx]] -= 2 * old_sign
                
                max_disc = np.max(np.abs(S))
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
            else:
                # Local minimum reached for this restart
                break
