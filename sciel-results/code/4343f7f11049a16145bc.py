<<<<<<< SEARCH
    def compute_radii(positions):
        """For given positions, compute max valid radii such that circles are disjoint and inside [0,1]^2."""
        n = len(positions)
        radii = np.zeros(n)
        # Wall constraints
        for i in range(n):
            radii[i] = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        # Pairwise constraints: r_i + r_j <= dist(i,j)
        # Use iterative relaxation
        for _ in range(50):
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(positions[i] - positions[j])
                    # r_i + r_j <= d  =>  each can be at most (d - other_r)
                    # Symmetric: r_i <= d - r_j, r_j <= d - r_i
                    # To maximize sum, set r_i = r_j = d/2 if unconstrained by walls
                    # But we need consistent solution. Use: r_i = min(r_i, d - r_j), r_j = min(r_j, d - r_i)
                    # This is a standard iterative approach
                    new_ri = min(radii[i], max(0, d - radii[j]))
                    new_rj = min(radii[j], max(0, d - new_ri))
                    radii[i] = new_ri
                    radii[j] = new_rj
            # Re-apply wall constraints
            for i in range(n):
                radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        return radii
=======
    def compute_radii(positions):
        """For given positions, compute max valid radii such that circles are disjoint and inside [0,1]^2."""
        n = len(positions)
        # Wall constraints
        wall_r = np.minimum.reduce([positions[:, 0], 1 - positions[:, 0], positions[:, 1], 1 - positions[:, 1]])
        radii = wall_r.copy()
        
        # Precompute distance matrix
        # D[i, j] = ||p_i - p_j||
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        D = np.linalg.norm(diff, axis=2)
        
        # Iterative relaxation: r_i = min(wall_r_i, min_{j!=i} (D_ij - r_j))
        # We do this for a few iterations. It converges fast.
        for _ in range(10):
            # For each i, we want min over j != i of (D[i, j] - radii[j])
            # Create a matrix where row i is D[i, :] - radii
            # But we need to exclude j=i. D[i, i] = 0, so D[i, i] - radii[i] = -radii[i].
            # We want min of (D[i, j] - radii[j]) for j != i.
            # If we take min over all j, we might pick j=i which is negative, which is wrong.
            # So we set D[i, i] to a large number or handle it.
            # Actually, the constraint is r_i + r_j <= D_ij.
            # So r_i <= D_ij - r_j.
            # Let's compute M[i, j] = D[i, j] - radii[j].
            # We want min_j (M[i, j]) for j != i.
            
            # Efficient way:
            # Temporarily set diagonal to inf?
            # M = D - radii[np.newaxis, :]
            # np.fill_diagonal(M, np.inf)
            # min_M = np.min(M, axis=1)
            # radii_new = np.minimum(wall_r, min_M)
            
            # To avoid creating M every time (memory), we can do:
            # But for N=32, M is 32x32, which is tiny.
            
            M = D - radii[np.newaxis, :]
            np.fill_diagonal(M, np.inf)
            min_constraints = np.min(M, axis=1)
            radii_new = np.minimum(wall_r, min_constraints)
            
            # Check convergence
            if np.allclose(radii_new, radii, atol=1e-8):
                radii = radii_new
                break
            radii = radii_new
            
        # Ensure non-negative
        radii = np.maximum(radii, 0.0)
        return radii
>>>>>>> REPLACE

