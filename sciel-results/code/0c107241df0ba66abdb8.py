    def _jc_distance_matrix(seqs):
        n = len(seqs)
        L = len(seqs[0])
        # Stack into a 2D numpy array
        # seqs is a list of arrays.
        # We can do:
        # mat = np.vstack(seqs)
        # Then compute pairwise differences.
        # But for n=32, L=2000, mat is 32x2000.
        # diff = mat[:, None, :] != mat[None, :, :]
        # This creates a 32x32x2000 boolean array. 32*32*2000 = 2M elements. This is fast in numpy.
        # Then sum over the last axis.
        
        mat = np.vstack(seqs)
        # mat shape (n, L)
        # We want diff[i,j] = sum(mat[i] != mat[j])
        # Using broadcasting:
        # diffs = (mat[:, None, :] != mat[None, :, :]).sum(axis=2)
        # This is O(n^2 * L) operations but vectorized.
        
        diffs = (mat[:, None, :] != mat[None, :, :]).sum(axis=2)
        p = diffs / L
        # Handle p >= 0.75
        D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        D[p >= 0.7499] = 3.0
        np.fill_diagonal(D, 0)
        return D
    