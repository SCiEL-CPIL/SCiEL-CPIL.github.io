    def can_add_fast(A, C, i, j):
        # Check if adding (i,j) to A is valid, given C is A@A
        # A[i,j] must be 0
        if A[i,j]: return False
        
        # Check pairs (i, x) and (j, x)
        # For x != i, j:
        # C[i,x] + A[j,x] <= 2
        # C[j,x] + A[i,x] <= 2
        
        # Vectorized
        # mask = ~np.eye(n, dtype=bool)
        # mask[i] = False
        # mask[j] = False
        
        # We can just check the whole row and ignore i,j
        # But C[i,i] is deg(i). A[j,i] is 0.
        # deg(i) can be > 2. So we MUST mask out i and j.
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # Check i
        if np.any((C[i, mask] + A[j, mask]) > 2):
            return False
        # Check j
        if np.any((C[j, mask] + A[i, mask]) > 2):
            return False
            
        return True
