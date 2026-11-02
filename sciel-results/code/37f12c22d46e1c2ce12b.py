import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    # We will use a block decomposition strategy.
    # Partition n=3 into [2, 1]
    # Partition m=5 into [2, 2, 1]
    # Partition p=6 into [2, 2, 2]
    
    # This creates blocks of sizes (ni, mj, pk)
    # For 2x2x2 blocks, we use Strassen (rank 7)
    # For others, we use Schoolbook (rank ni*mj*pk)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Partitions
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Start indices
    n_starts = [0, 2]
    m_starts = [0, 2, 4]
    p_starts = [0, 2, 4]
    
    # Helper to add a schoolbook block
    def add_schoolblock(ni, mj, pk, i0, j0, k0):
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    col_idx = len(U_list)
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    
                    U[(i0 + ii) * m + (j0 + jj)] = 1.0
                    V[(j0 + jj) * p + (k0 + kk)] = 1.0
                    W[(k0 + kk) * n + (i0 + ii)] = 1.0
                    
                    U_list.append(U)
                    V_list.append(V)
                    W_list.append(W)
                    
    # Helper to add a Strassen block (2x2x2)
    def add_strassen_block(i0, j0, k0):
        # Local indices 0,1 for i,j,k
        # Global indices:
        # i: i0, i0+1
        # j: j0, j0+1
        # k: k0, k0+1
        
        # Strassen formulas
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to construct U, V, W vectors for each of the 7 multiplications
        # U[i, r] corresponds to A[i, j] coeff
        # V[j, r] corresponds to B[j, k] coeff
        # W[k, r] corresponds to C[k, i] coeff? 
        # Wait, the tensor is T[i,j,k] = sum U[i,j,r] V[j,k,r] W[k,i,r]?
        # The standard tensor decomposition is T[i,j,k] = sum U_i^r V_j^r W_k^r.
        # Here indices are i (row of A), j (col of A/row of B), k (col of B).
        # So U is associated with A[i,j], V with B[j,k], W with C[i,k]?
        # The task says: W[k*n+i, r]. This suggests W is indexed by (k, i).
        # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # Summing over r gives C[i,k] = sum_j A[i,j] B[j,k].
        # Yes, this matches.
        
        # Strassen M1: (A11+A22)(B11+B22)
        # A part: A11 (i=0,j=0) coeff 1, A22 (i=1,j=1) coeff 1
        # B part: B11 (j=0,k=0) coeff 1, B22 (j=1,k=1) coeff 1
        # C part: C11 (i=0,k=0) coeff 1, C12 (i=0,k=1) coeff 1, C21 (i=1,k=0) coeff 1, C22 (i=1,k=1) coeff 1?
        # No, M1 contributes to C11 and C22?
        # C11 = M1 + ...
        # C22 = M1 + ...
        # So W for M1 has 1 at (k=0, i=0) and (k=1, i=1).
        
        def make_col(u_vals, v_vals, w_vals):
            # u_vals: dict {(i,j): coeff}
            # v_vals: dict {(j,k): coeff}
            # w_vals: dict {(k,i): coeff}
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            
            for (ii, jj), c in u_vals.items():
                U[(i0 + ii) * m + (j0 + jj)] = c
            for (jj, kk), c in v_vals.items():
                V[(j0 + jj) * p + (k0 + kk)] = c
            for (kk, ii), c in w_vals.items():
                W[(k0 + kk) * n + (i0 + ii)] = c
                
            U_list.append(U)
            V_list.append(V)
            W_list.append(W)
            
        # M1
        make_col({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        
        # M2: (A21+A22)B11
        # A: A21 (i=1,j=0) 1, A22 (i=1,j=1) 1
        # B: B11 (j=0,k=0) 1
        # C: C21 (i=1,k=0) 1
        make_col({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1})
        
        # M3: A11(B12-B22)
        # A: A11 (i=0,j=0) 1
        # B: B12 (j=0,k=1) 1, B22 (j=1,k=1) -1
        # C: C12 (i=0,k=1) 1
        make_col({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1})
        
        # M4: A22(B21-B11)
        # A: A22 (i=1,j=1) 1
        # B: B21 (j=1,k=0) 1, B11 (j=0,k=0) -1
        # C: C21 (i=1,k=0) 1, C11 (i=0,k=0) 1
        make_col({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (0,0): 1})
        
        # M5: (A11+A12)B22
        # A: A11 (i=0,j=0) 1, A12 (i=0,j=1) 1
        # B: B22 (j=1,k=1) 1
        # C: C11 (i=0,k=0) -1, C12 (i=0,k=1) 1
        make_col({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1})
        
        # M6: (A21-A11)(B11+B12)
        # A: A21 (i=1,j=0) 1, A11 (i=0,j=0) -1
        # B: B11 (j=0,k=0) 1, B12 (j=0,k=1) 1
        # C: C22 (i=1,k=1) 1
        make_col({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
        
        # M7: (A12-A22)(B21+B22)
        # A: A12 (i=0,j=1) 1, A22 (i=1,j=1) -1
        # B: B21 (j=1,k=0) 1, B22 (j=1,k=1) 1
        # C: C11 (i=0,k=0) 1, C22 (i=1,k=1) 1
        make_col({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        
    # Iterate over blocks
    for ni_idx, ni in enumerate(n_parts):
        for mj_idx, mj in enumerate(m_parts):
            for pk_idx, pk in enumerate(p_parts):
                i0 = n_starts[ni_idx]
                j0 = m_starts[mj_idx]
                k0 = p_starts[pk_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    add_strassen_block(i0, j0, k0)
                else:
                    add_schoolblock(ni, mj, pk, i0, j0, k0)
                    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
