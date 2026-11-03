    def build_decomposition(tiling_strategy):
        """
        tiling_strategy: list of (i_range, j_range, k_range, method)
        method: 'strassen' for 2x2x2, 'schoolbook' for any size
        """
        U = np.zeros((n * m, 200))
        V = np.zeros((m * p, 200))
        W = np.zeros((p * n, 200))
        
        col = 0
        
        def add_strassen_2x2x2(i0, j0, k0):
            nonlocal col
            # Standard Strassen 2x2x2
            # M1 = (A11+A22)(B11+B22)
            U[i0 * m + j0, col] += 1
            U[(i0+1) * m + j0+1, col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0+1) * p + k0+1, col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0+1) * n + i0+1, col] += 1
            col += 1
            
            # M2 = (A21+A22)B11
            U[(i0+1) * m + j0, col] += 1
            U[(i0+1) * m + j0+1, col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + i0+1, col] += 1
            W[(k0+1) * n + i0+1, col] -= 1
            col += 1
            
            # M3 = A11(B21-B11)
            U[i0 * m + j0, col] += 1
            V[j0 * p + k0+1, col] += 1
            V[(j0+1) * p + k0+1, col] -= 1
            W[(k0+1) * n + i0, col] += 1
            W[(k0+1) * n + i0+1, col] += 1
            col += 1
            
            # M4 = A22(B22-B12)
            U[(i0+1) * m + j0+1, col] += 1
            V[(j0+1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + i0+1, col] += 1
            col += 1
            
            # M5 = (A11+A12)B22
            U[i0 * m + j0, col] += 1
            U[i0 * m + j0+1, col] += 1
            V[(j0+1) * p + k0+1, col] += 1
            W[(k0+1) * n + i0, col] += 1
            W[k0 * n + i0, col] -= 1
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[(i0+1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + k0+1, col] += 1
            W[(k0+1) * n + i0+1, col] += 1
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[i0 * m + j0+1, col] += 1
            U[(i0+1) * m + j0+1, col] -= 1
            V[(j0+1) * p + k0, col] += 1
            V[(j0+1) * p + k0+1, col] += 1
            W[k0 * n + i0, col] += 1
            col += 1
        
        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
                        
        def add_rank1_slice(i_list, j_list, k_list):
            """
            Adds a rank-1 decomposition for a slice where the tensor is 1 
            for all i in i_list, j in j_list, k in k_list.
            This is valid if the slice corresponds to a single contraction index j 
            or a single output index i? 
            Actually, my derivation above showed it works for:
            1. Fixed j, varying i, k. (Outer product of A_col and B_row)
            2. Fixed i, varying j, k. (Outer product of A_row and B_col)
            
            Case 1: Fixed j. 
            T[i, j, k] = 1.
            U[i*m+j, r] = 1
            V[j*p+k, r] = 1
            W[k*n+i, r] = 1
            
            Case 2: Fixed i.
            T[i, j, k] = 1.
            U[i*m+j, r] = 1
            V[j*p+k, r] = 1
            W[k*n+i, r] = 1
            
            Both have the same form!
            U_{a,r} = 1 for a in slice
            V_{b,r} = 1 for b in slice
            W_{c,r} = 1 for c in slice
            """
            nonlocal col
            for i in i_list:
                for j in j_list:
                    U[i * m + j, col] = 1
            for j in j_list:
                for k in k_list:
                    V[j * p + k, col] = 1
            for k in k_list:
                for i in i_list:
                    W[k * n + i, col] = 1
            col += 1

        # Tiling: i={0,1}, j={0,1}, k={0,1},2,3,4,5,6
        for k0 in [0, 2, 4]:
            add_strassen_2x2x2(0, 0, k0)
        add_schoolbook([0, 1], [0, 1], [6])
        
        # i={0,1}, j={2,3}, k=0..6
        for k0 in [0, 2, 4]:
            add_strassen_2x2x2(0, 2, k0)
        add_schoolbook([0, 1], [2, 3], [6])
        
        # i={0,1}, j={4}, k=0..6
        # This is a fixed j slice. Rank 1.
        add_rank1_slice([0, 1], [4], range(7))
        
        # i={2}, j=0..4, k=0..6
        # This is a fixed i slice. Rank 1.
        add_rank1_slice([2], range(5), range(7))
        
        return U[:, :col], V[:, :col], W[:, :col], col
