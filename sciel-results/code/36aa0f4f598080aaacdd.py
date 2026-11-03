"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Block structure:
    # A (3x4): rows [2,1], cols [2,2]
    # B (4x6): rows [2,2], cols [2,2,2]
    # C (3x6): rows [2,1], cols [2,2,2]
    
    # A blocks: A[ri, ci] where ri in {0,1}, ci in {0,1}
    #   A[0,0] = A[0:2, 0:2] (2x2)
    #   A[0,1] = A[0:2, 2:4] (2x2)
    #   A[1,0] = A[2:3, 0:2] (1x2)
    #   A[1,1] = A[2:3, 2:4] (1x2)
    
    # B blocks: B[ri, ci] where ri in {0,1}, ci in {0,1,2}
    #   B[0,0] = B[0:2, 0:2] (2x2)
    #   B[0,1] = B[0:2, 2:4] (2x2)
    #   B[0,2] = B[0:2, 4:6] (2x2)
    #   B[1,0] = B[2:4, 0:2] (2x2)
    #   B[1,1] = B[2:4, 2:4] (2x2)
    #   B[1,2] = B[2:4, 4:6] (2x2)
    
    # C blocks: C[ri, ci] where ri in {0,1}, ci in {0,1,2}
    #   C[0,ci] = A[0,0]*B[0,ci] + A[0,1]*B[1,ci]  (2x2 = 2x2*2x2 + 2x2*2x2)
    #   C[1,ci] = A[1,0]*B[0,ci] + A[1,1]*B[1,ci]  (1x2 = 1x2*2x2 + 1x2*2x2)
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A01)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each 2x2*2x2 multiplication, we need 7 rank-1 terms.
    # Each term is: (a combination of A entries) * (b combination of B entries) * (contributes to specific C entries)
    
    # Let's build the decomposition.
    # Each "rank-1 term" in the full tensor decomposition has:
    #   U entry: coefficient for each (i,j) in A
    #   V entry: coefficient for each (j',k) in B  
    #   W entry: coefficient for each (k',i') in C (using trace convention W[k*n+i])
    
    # Actually, the tensor T[i,j,k] = 1 for all valid i,j,k means:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (the multiplication tensor)
    
    # A rank-r decomposition writes T as sum_{c=1}^r U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector over (i,j), V_c over (j',k), W_c over (k',i')
    # and the reconstruction is: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # Wait, the indices need to match. Let me re-read the convention.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column index of B (and C) and i is the row index of A (and C).
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the multiplication to work: C[i,k] = sum_j A[i,j]*B[j,k] = T[i,j,k] summed over j
    # So T[i,j,k] should equal 1 for all i,j,k (since the multiplication tensor has all 1s)
    # And the reconstruction gives: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is the standard matrix multiplication tensor.
    
    # Let me build the Strassen-based decomposition.
    # For a 2x2*2x2 block at position (row_block ri, col_block ci) in C:
    # C_block[ri,ci] = A_block[ri,0]*B_block[0,ci] + A_block[ri,1]*B_block[1,ci]
    
    # For ri=0 (top 2 rows), both A and B blocks are 2x2, so use Strassen.
    # For ri=1 (bottom 1 row), A blocks are 1x2, B blocks are 2x2, direct multiplication.
    
    # Let me enumerate all the rank-1 terms.
    
    terms = []  # Each term: (U_dict, V_dict, W_dict) where dicts map (i,j)->coeff, (j,k)->coeff, (k,i)->coeff
    
    def add_term(U_entries, V_entries, W_entries):
        """Add a rank-1 term. U_entries: list of (i,j,coeff), etc."""
        U_d = {}
        for (i,j,c) in U_entries:
            U_d[(i,j)] = c
        V_d = {}
        for (j,k,c) in V_entries:
            V_d[(j,k)] = c
        W_d = {}
        for (k,i,c) in W_entries:
            W_d[(k,i)] = c
        terms.append((U_d, V_d, W_d))
    
    # Strassen terms for 2x2*2x2:
    # The 2x2 A block has entries at (i,j) for i in {r0,r1}, j in {c0,c1}
    # The 2x2 B block has entries at (j',k) for j' in {r0',r1'}, k in {c0',c1'}
    # The 2x2 C block has entries at (i',k') for i' in {r0,r1}, k' in {c0',c1'}
    
    def strassen_terms(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Generate 7 Strassen terms for multiplying A[A_rows, A_cols] * B[B_rows, B_cols] -> C[C_rows, C_cols]
        where all are 2x2 blocks. A_rows = (r0,r1), A_cols=(c0,c1), B_rows=(s0,s1), B_cols=(t0,t1),
        C_rows = (r0,r1), C_cols=(t0,t1). Note: A_cols must equal B_rows for the multiplication to make sense,
        but in our block structure, the column indices of A block = row indices of B block.
        """
        r0, r1 = A_rows
        c0, c1 = A_cols  # = B_rows
        s0, s1 = B_rows  # = A_cols
        t0, t1 = B_cols
        
        # A block entries: A[r0,c0], A[r0,c1], A[r1,c0], A[r1,c1]
        # B block entries: B[s0,t0], B[s0,t1], B[s1,t0], B[s1,t1]
        # C block entries: C[r0,t0], C[r0,t1], C[r1,t0], C[r1,t1]
        
        # M1 = (A[r0,c0]+A[r1,c1]) * (B[s0,t0]+B[s1,t1])
        # C[r0,t0] += M1, C[r1,t1] += M1
        add_term(
            [(r0,c0,1), (r1,c1,1)],
            [(s0,t0,1), (s1,t1,1)],
            [(t0,r0,1), (t1,r1,1)]
        )
        
        # M2 = (A[r1,c0]+A[r1,c1]) * B[s0,t0]
        # C[r1,t0] += M2, C[r1,t1] -= M2
        add_term(
            [(r1,c0,1), (r1,c1,1)],
            [(s0,t0,1)],
            [(t0,r1,1), (t1,r1,-1)]
        )
        
        # M3 = A[r0,c0] * (B[s0,t1]-B[s1,t1])
        # C[r0,t1] += M3, C[r1,t1] += M3
        add_term(
            [(r0,c0,1)],
            [(s0,t1,1), (s1,t1,-1)],
            [(t1,r0,1), (t1,r1,1)]
        )
        
        # M4 = A[r1,c1] * (B[s1,t0]-B[s0,t0])
        # C[r1,t0] += M4, C[r0,t0] += M4
        add_term(
            [(r1,c1,1)],
            [(s1,t0,1), (s0,t0,-1)],
            [(t0,r1,1), (t0,r0,1)]
        )
        
        # M5 = (A[r0,c0]+A[r0,c1]) * B[s1,t1]
        # C[r1,t1] += M5, C[r0,t0] -= M5
        add_term(
            [(r0,c0,1), (r0,c1,1)],
            [(s1,t1,1)],
            [(t1,r1,1), (t0,r0,-1)]
        )
        
        # M6 = (A[r1,c1]-A[r0,c1]) * (B[s0,t0]+B[s0,t1])
        # C[r1,t1] += M6
        add_term(
            [(r1,c1,1), (r0,c1,-1)],
            [(s0,t0,1), (s0,t1,1)],
            [(t1,r1,1)]
        )
        
        # M7 = (A[r0,c1]-A[r1,c1]) * (B[s1,t1]+B[s0,t1])
        # C[r0,t0] += M7
        add_term(
            [(r0,c1,1), (r1,c1,-1)],
            [(s1,t1,1), (s0,t1,1)],
            [(t0,r0,1)]
        )
    
    # Direct multiplication terms for 1x2 * 2x2 -> 1x2:
    # A block: 1x2 at row r, cols (c0,c1)
    # B block: 2x2 at rows (s0,s1), cols (t0,t1)
    # C block: 1x2 at row r, cols (t0,t1)
    # C[r,t0] = A[r,c0]*B[s0,t0] + A[r,c1]*B[s1,t0]
    # C[r,t1] = A[r,c0]*B[s0,t1] + A[r,c1]*B[s1,t1]
    
    def direct_1x2_2x2_terms(A_row, A_cols, B_rows, B_cols, C_row, C_cols):
        r = A_row
        c0, c1 = A_cols
        s0, s1 = B_rows
        t0, t1 = B_cols
        
        # Term 1: A[r,c0]*B[s0,t0] -> C[r,t0]
        add_term(
            [(r,c0,1)],
            [(s0,t0,1)],
            [(t0,r,1)]
        )
        # Term 2: A[r,c1]*B[s1,t0] -> C[r,t0]
        add_term(
            [(r,c1,1)],
            [(s1,t0,1)],
            [(t0,r,1)]
        )
        # Term 3: A[r,c0]*B[s0,t1] -> C[r,t1]
        add_term(
            [(r,c0,1)],
            [(s0,t1,1)],
            [(t1,r,1)]
        )
        # Term 4: A[r,c1]*B[s1,t1] -> C[r,t1]
        add_term(
            [(r,c1,1)],
            [(s1,t1,1)],
            [(t1,r,1)]
        )
    
    # Now, the block structure:
    # A rows: [0,1] (top 2), [2] (bottom 1)
    # A cols: [0,1] (left 2), [2,3] (right 2)
    # B rows: [0,1] (top 2), [2,3] (bottom 2)
    # B cols: [0,1], [2,3], [4,5] (three 2-col groups)
    # C rows: [0,1] (top 2), [2] (bottom 1)
    # C cols: [0,1], [2,3], [4,5]
    
    # For each column group ci in {0,1,2}:
    #   C[0,ci] = A[0,0]*B[0,ci] + A[0,1]*B[1,ci]  (2x2 = 2x2*2x2 + 2x2*2x2)
    #   C[1,ci] = A[1,0]*B[0,ci] + A[1,1]*B[1,ci]  (1x2 = 1x2*2x2 + 1x2*2x2)
    
    # Column group 0: cols [0,1]
    # Column group 1: cols [2,3]
    # Column group 2: cols [4,5]
    
    col_groups = [(0,1), (2,3), (4,5)]
    
    for ci, (t0, t1) in enumerate(col_groups):
        # Top 2 rows: C[0:2, t0:t1+1] = A[0:2, 0:2]*B[0:2, t0:t1+1] + A[0:2, 2:4]*B[2:4, t0:t1+1]
        # First term: A[0:2, 0:2] * B[0:2, t0:t1+1]
        strassen_terms((0,1), (0,1), (0,1), (t0,t1), (0,1), (t0,t1))
        # Second term: A[0:2, 2:4] * B[2:4, t0:t1+1]
        strassen_terms((0,1), (2,3), (2,3), (t0,t1), (0,1), (t0,t1))
        
        # Bottom 1 row: C[2, t0:t1+1] = A[2, 0:2]*B[0:2, t0:t1+1] + A[2, 2:4]*B[2:4, t0:t1+1]
        # First term: A[2, 0:2] * B[0:2, t0:t1+1]
        direct_1x2_2x2_terms(2, (0,1), (0,1), (t0,t1), 2, (t0,t1))
        # Second term: A[2, 2:4] * B[2:4, t0:t1+1]
        direct_1x2_2x2_terms(2, (2,3), (2,3), (t0,t1), 2, (t0,t1))
    
    # Now build the U, V, W matrices
    # U: shape (n*m, rank) = (12, rank)
    # V: shape (m*p, rank) = (24, rank)
    # W: shape (p*n, rank) = (18, rank)
    
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (U_d, V_d, W_d) in enumerate(terms):
        for (i, j), coeff in U_d.items():
            U[i * m + j, c] = coeff
        for (j, k), coeff in V_d.items():
            V[j * p + k, c] = coeff
        for (k, i), coeff in W_d.items():
            W[k * n + i, c] = coeff
    
    # Verify
    def verify(U_mat, V_mat, W_mat, r):
        T_reconstructed = np.zeros((n, m, p))
        for c in range(r):
            u = U_mat[:, c].reshape(n, m)
            v = V_mat[:, c].reshape(m, p)
            w = W_mat[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)
    
    if not verify(U, V, W, rank):
        # Fall back to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = rank_school
    
    # Convert to lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
