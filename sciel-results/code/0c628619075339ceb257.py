    def min_triangle_area(pts):
        # pts: (N, 2)
        # Compute all pairwise differences? No, we need triples.
        # There are 165 triples.
        # We can create an array of indices.
        idx = np.indices((N, N, N))
        # Filter i < j < k
        mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
        i_idx, j_idx, k_idx = idx[:, :, :][mask].T # This creates large arrays.
        # Better:
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        p_i = pts[i_idx]
        p_j = pts[j_idx]
        p_k = pts[k_idx]
        
        # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        # det is v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(det)
        return np.min(areas)
