<<<<<<< SEARCH
    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        # Schoolbook for k = k0+6
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        For 1x2 * 2x7: each output C[i,k] = A[i,j0]*B[j0,k] + A[i,j1]*B[j1,k]
        This is essentially a dot product of length 2 for each of 7 outputs.
        Schoolbook rank = 2*7 = 14.
        """
        j1 = j0 + 1
        for k in range(7):
            kk = k0 + k
            # C[k] = A[j0]*B[j0,k] + A[j1]*B[j1,k]
            # Term 1: A[j0] * B[j0,k]
            add_rank1_term([i0*m+j0], [1.0],
                           [j0*p+kk], [1.0],
                           [kk*n+i0], [1.0])
            # Term 2: A[j1] * B[j1,k]
            add_rank1_term([i0*m+j1], [1.0],
                           [j1*p+kk], [1.0],
                           [kk*n+i0], [1.0])

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 25 (Strassen x3 + schoolbook)
          M_{1,2,7}: 14 (schoolbook)
        Total: 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)
=======
    def m221_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x1 block.
        Rank 1.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        
        # U: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # V: (j0, k0), (j1, k0)
        # W: (k0, i0), (k0, i1)
        add_rank1_term([i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1], [1, 1, 1, 1],
                       [j0*p+k0, j1*p+k0], [1, 1],
                       [k0*n+i0, k0*n+i1], [1, 1])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Rank 1.
        """
        j1 = j0 + 1
        
        # U: (i0, j0), (i0, j1)
        # V: (j0, k0..k0+6), (j1, k0..k0+6)
        # W: (k0..k0+6, i0)
        
        u_indices = [i0*m+j0, i0*m+j1]
        u_vals = [1, 1]
        
        v_indices = []
        v_vals = []
        w_indices = []
        w_vals = []
        
        for k in range(7):
            kk = k0 + k
            v_indices.append(j0*p+kk)
            v_vals.append(1)
            v_indices.append(j1*p+kk)
            v_vals.append(1)
            w_indices.append(kk*n+i0)
            w_vals.append(1)
            
        add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + m221 = 7+7+7+1 = 22
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        m221_decomp(i0, j0, k0 + 6)

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 22
          M_{1,2,7}: 1
        Total: 23
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)
>>>>>>> REPLACE
