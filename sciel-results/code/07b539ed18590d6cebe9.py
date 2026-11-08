            dots = current_set @ v
            dists_sq = vn + current_norms - 2.0 * dots
            conflicts = np.where(dists_sq < target_N)[0]
            
            if len(conflicts) == 0:
                # Add
                current_set = np.vstack([current_set, v])
                current_norms = np.append(current_norms, vn)
            elif len(conflicts) == 1:
                # Swap
                u_idx = conflicts[0]
                current_set[u_idx] = v
                current_norms[u_idx] = vn
            else:
                # Skip or try multi-swap (too expensive)
                pass
