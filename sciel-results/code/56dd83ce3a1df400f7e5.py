import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists or numpy arrays.
    # Rank is not known a priori, so we will append columns.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a term
    def add_term(u_val, v_val, w_val):
        # u_val is dict (i,j) -> val
        # v_val is dict (j,k) -> val
        # w_val is dict (i,k) -> val
        pass

    # This approach is hard to manage dynamically.
    # Better: Pre-calculate the rank.
    # 4 Strassen blocks * 7 = 28.
    # Remaining 43 * 1 = 43.
    # Total Rank = 71.
    
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 blocks
    # Block (i_start, j_start, k_start)
    blocks = [
        (0, 0, 0), # i=0,1; j=0,1; k=0,1
        (0, 2, 0), # i=0,1; j=2,3; k=0,1
        (0, 0, 2), # i=0,1; j=0,1; k=2,3
        (0, 2, 2)  # i=0,1; j=2,3; k=2,3
    ]
    
    covered = set()
    
    for i0, j0, k0 in blocks:
        # Strassen for this block
        # Local indices 0,1 map to global i0, i0+1 etc.
        
        # M1 = A_00 (B_11 - B_01)
        # A_00 is (i0, j0)
        # B_11 is (j0+1, k0+1), B_01 is (j0, k0+1)
        # C_00 is (i0, k0)
        
        # We need to define the 7 terms.
        # Term 1: A_00 * (B_11 - B_01) -> C_00, C_11? No.
        # C_00 = M1 + M4 - M5 + M7
        # C_01 = M2 + M5
        # C_10 = M3 + M4
        # C_11 = M1 - M2 + M3 + M6
        
        # M1: A(i0, j0) * (B(j0+1, k0+1) - B(j0, k0+1))
        # U: (i0, j0) = 1
        # V: (j0+1, k0+1) = 1, (j0, k0+1) = -1
        # W: (i0, k0) = 1, (i0+1, k0+1) = 1
        
        # Let's manually define the 7 vectors for U, V, W for this block.
        
        # M1
        U[i0 * m + j0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        V[j0 * p + (k0+1)][col] = -1
        W[k0 * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M2: (A_00 + A_11) * B_01
        # A_00 (i0, j0), A_11 (i0+1, j0+1)
        # B_01 (j0, k0+1)
        # C_01 (i0, k0+1), C_11 (i0+1, k0+1)
        
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M3: (A_10 + A_11) * B_00
        # A_10 (i0+1, j0), A_11 (i0+1, j0+1)
        # B_00 (j0, k0)
        # C_10 (i0+1, k0), C_11 (i0+1, k0+1)
        
        U[(i0+1) * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M4: A_11 * (B_00 - B_10)
        # A_11 (i0+1, j0+1)
        # B_00 (j0, k0), B_10 (j0+1, k0)
        # C_00 (i0, k0), C_10 (i0+1, k0)
        
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0+1) * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        col += 1
        
        # M5: (A_00 + A_01) * B_11
        # A_00 (i0, j0), A_01 (i0, j0+1)
        # B_11 (j0+1, k0+1)
        # C_00 (i0, k0), C_01 (i0, k0+1)
        
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k0+1) * n + i0][col] = 1
        col += 1
        
        # M6: (A_10 - A_00) * (B_00 + B_01)
        # A_10 (i0+1, j0), A_00 (i0, j0)
        # B_00 (j0, k0), B_01 (j0, k0+1)
        # C_11 (i0+1, k0+1)
        
        U[(i0+1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M7: (A_01 - A_11) * (B_10 + B_11)
        # A_01 (i0, j0+1), A_11 (i0+1, j0+1)
        # B_10 (j0+1, k0), B_11 (j0+1, k0+1)
        # C_00 (i0, k0), C_11 (i0+1, k0+1)
        
        U[i0 * m + (j0+1)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # Mark covered indices
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
                    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    # Verify col == rank
    assert col == rank, f"Col {col} != Rank {rank}"
    
    return {"U": U, "V": V, "W": W}
