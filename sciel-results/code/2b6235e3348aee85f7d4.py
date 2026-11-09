"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
_start = time.time()
_DEADLINE = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U index: i*m+j for i in [0,n), j in [0,m)
    # V index: j*p+k for j in [0,m), k in [0,p)
    # W index: k*n+i for k in [0,p), i in [0,n)
    
    # We'll build the decomposition using block structure.
    # Split into 2x2 blocks.
    
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x5: B[j,k] for j in 0..3, k in 0..4
    # C is 4x5: C[i,k] for i in 0..3, k in 0..4
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] where B00,B10 are 2x2, B01,B11 are 2x3
    # C = [[C00, C01], [C10, C11]] where C00,C10 are 2x2, C01,C11 are 2x3
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each 2x2*2x2 product, use Strassen (rank 7)
    # For each 2x2*2x3 product, use schoolbook (rank 6) for now
    
    # Let's define helper functions to add a rank-1 term
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """Add a rank-1 term. u, v, w are dicts mapping index -> coefficient."""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen for 2x2 * 2x2
    # Inputs: A[a,b] for a,b in 0,1 ; B[c,d] for c,d in 0,1
    # Output: C[e,f] = sum_c A[e,c]*B[c,f] for e,f in 0,1
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Compute 2x2 * 2x2 product.
        a_row_offset: starting row in A (0 or 2)
        b_col_offset: starting col in B (0 or 2) -- but B has 5 cols, so this is tricky
        Actually, let me think in terms of the full indices.
        """
        pass  # I'll implement inline below
    
    # Let me just directly construct the terms.
    # I'll use a dictionary-based approach.
    
    # For a 2x2*2x2 block at A rows [ar, ar+1], cols [ac, ac+1]
    # and B rows [br, br+1], cols [bc, bc+1]
    # producing C rows [cr, cr+1], cols [cc, cc+1]
    
    def add_strassen(ar, ac, br, bc, cr, cc):
        """Add Strassen decomposition for A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[cr:cr+2, cc:cc+2]"""
        # A indices: (ar,ac), (ar,ac+1), (ar+1,ac), (ar+1,ac+1)
        # B indices: (br,bc), (br,bc+1), (br+1,bc), (br+1,bc+1)
        # C indices: (cr,cc), (cr,cc+1), (cr+1,cc), (cr+1,cc+1)
        
        # M1 = (A[ar,ac]+A[ar+1,ac+1]) * (B[br,bc]+B[br+1,bc+1])
        u = {(ar,ac): 1, (ar+1,ac+1): 1}
        v = {(br,bc): 1, (br+1,bc+1): 1}
        w = {(cr,cc): 1, (cr+1,cc+1): 1}  # C00 += M1, C11 += M1
        add_term(u, v, w)
        
        # M2 = (A[ar+1,ac]+A[ar+1,ac+1]) * B[br,bc]
        u = {(ar+1,ac): 1, (ar+1,ac+1): 1}
        v = {(br,bc): 1}
        w = {(cr+1,cc): 1}  # C10 += M2
        add_term(u, v, w)
        
        # M3 = A[ar,ac] * (B[br+1,bc]-B[br+1,bc+1])
        u = {(ar,ac): 1}
        v = {(br+1,bc): 1, (br+1,bc+1): -1}
        w = {(cr,cc+1): 1}  # C01 += M3
        add_term(u, v, w)
        
        # M4 = A[ar+1,ac+1] * (B[br,bc+1]-B[br,bc])
        u = {(ar+1,ac+1): 1}
        v = {(br,bc+1): 1, (br,bc): -1}
        w = {(cr,cc): 1, (cr+1,cc): 1}  # C00 += M4, C10 += M4
        add_term(u, v, w)
        
        # M5 = (A[ar,ac]+A[ar,ac+1]) * B[br+1,bc+1]
        u = {(ar,ac): 1, (ar,ac+1): 1}
        v = {(br+1,bc+1): 1}
        w = {(cr,cc): -1, (cr,cc+1): 1}  # C00 -= M5, C01 += M5
        add_term(u, v, w)
        
        # M6 = (A[ar+1,ac]-A[ar,ac]) * (B[br,bc]+B[br+1,bc])
        u = {(ar+1,ac): 1, (ar,ac): -1}
        v = {(br,bc): 1, (br+1,bc): 1}
        w = {(cr+1,cc+1): 1}  # C11 += M6
        add_term(u, v, w)
        
        # M7 = (A[ar,ac+1]-A[ar+1,ac+1]) * (B[br,bc+1]+B[br+1,bc+1])
        u = {(ar,ac+1): 1, (ar+1,ac+1): -1}
        v = {(br,bc+1): 1, (br+1,bc+1): 1}
        w = {(cr,cc): 1, (cr+1,cc+1): 1}  # C00 += M7, C11 += M7
        add_term(u, v, w)
    
    # For a 2x2*2x3 block: A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+3] -> C[cr:cr+2, cc:cc+3]
    # Schoolbook: 2*2*3 = 6 terms
    def add_schoolbook_2x2_2x3(ar, ac, br, bc, cr, cc):
        for di in range(2):
            for dj in range(2):
                for dk in range(3):
                    u = {(ar+di, ac+dj): 1}
                    v = {(br+dj, bc+dk): 1}
                    w = {(cr+di, cc+dk): 1}
                    add_term(u, v, w)
    
    # Now build the full decomposition:
    # C00 = A00*B00 + A01*B10
    # A00: rows 0-1, cols 0-1; B00: rows 0-1, cols 0-1; C00: rows 0-1, cols 0-1
    add_strassen(0, 0, 0, 0, 0, 0)
    # A01: rows 0-1, cols 2-3; B10: rows 2-3, cols 0-1; C00: rows 0-1, cols 0-1
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C01 = A00*B01 + A01*B11
    # A00: rows 0-1, cols 0-1; B01: rows 0-1, cols 2-4; C01: rows 0-1, cols 2-4
    add_schoolbook_2x2_2x3(0, 0, 0, 2, 0, 2)
    # A01: rows 0-1, cols 2-3; B11: rows 2-3, cols 2-4; C01: rows 0-1, cols 2-4
    add_schoolbook_2x2_2x3(0, 2, 2, 2, 0, 2)
    
    # C10 = A10*B00 + A11*B10
    # A10: rows 2-3, cols 0-1; B00: rows 0-1, cols 0-1; C10: rows 2-3, cols 0-1
    add_strassen(2, 0, 0, 0, 2, 0)
    # A11: rows 2-3, cols 2-3; B10: rows 2-3, cols 0-1; C10: rows 2-3, cols 0-1
    add_strassen(2, 2, 2, 0, 2, 0)
    
    # C11 = A10*B01 + A11*B11
    # A10: rows 2-3, cols 0-1; B01: rows 0-1, cols 2-4; C11: rows 2-3, cols 2-4
    add_schoolbook_2x2_2x3(2, 0, 0, 2, 2, 2)
    # A11: rows 2-3, cols 2-3; B11: rows 2-3, cols 2-4; C11: rows 2-3, cols 2-4
    add_schoolbook_2x2_2x3(2, 2, 2, 2, 2, 2)
    
    # Now convert the list of terms into U, V, W matrices
    rank = len(U_cols)
    
    # U shape: (n*m, rank) = (16, rank)
    # U index: i*m + j
    U = [[0.0] * rank for _ in range(n * m)]
    # V shape: (m*p, rank) = (20, rank)
    # V index: j*p + k
    V = [[0.0] * rank for _ in range(m * p)]
    # W shape: (p*n, rank) = (20, rank)
    # W index: k*n + i
    W = [[0.0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        u_dict = U_cols[col]
        v_dict = V_cols[col]
        w_dict = W_cols[col]
        
        for (i, j), val in u_dict.items():
            U[i * m + j][col] = val
        
        for (j, k), val in v_dict.items():
            V[j * p + k][col] = val
        
        for (i, k), val in w_dict.items():
            W[k * n + i][col] = val
    
    # Convert to integers if possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
