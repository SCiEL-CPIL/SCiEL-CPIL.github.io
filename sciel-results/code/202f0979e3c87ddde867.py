        # Greedy farthest point sampling, vectorized
        for _ in range(N - 1):
            if not remaining_indices:
                break
            # Vectorized distance calculation
            # remaining_candidates: (M, 2)
            # selected_candidates: (K, 2)
            # We want min distance from each remaining to any selected.
            
            rem_pts = candidates[remaining_indices]
            sel_pts = candidates[selected_indices]
            
            # Compute pairwise distances
            # dists[i, j] = || rem_pts[i] - sel_pts[j] ||
            # Using broadcasting
            # rem_pts[:, np.newaxis, :] - sel_pts[np.newaxis, :, :]
            # This creates (M, K, 2)
            
            diffs = rem_pts[:, np.newaxis, :] - sel_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # Find index of max min_dist
            # max_val = np.max(min_dists)
            # max_idx_local = np.argmax(min_dists)
            # max_idx_global = remaining_indices[max_idx_local]
            
            max_idx_local = np.argmax(min_dists)
            max_idx_global = remaining_indices[max_idx_local]
            
            selected_indices.append(max_idx_global)
            remaining_indices.remove(max_idx_global)
