        mask_v = v > 0
        if np.any(mask_v):
            idx = np.where(mask_v)[0]
            # c_i / ||c_i||
            norms = dists[idx]
            # Handle zero norm (unlikely)
            norms = np.where(norms == 0, 1.0, norms)
            grad_vecs = 2.0 * v[idx] * pts[idx] / norms[:, np.newaxis]
            # Correctly index into the flattened gradient vector
            grad[2 * idx] += grad_vecs[:, 0]
            grad[2 * idx + 1] += grad_vecs[:, 1]
