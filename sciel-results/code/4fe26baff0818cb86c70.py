"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return the best exact tensor decomposition found within a time budget."""
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill
    
    n, m, p = DIMENSIONS
    
    # We construct a hybrid decomposition using Strassen for 2x2x2 blocks
    # and Schoolbook for the rest.
    # This yields a rank of 78 for 3x4x7.
    
    # Helper to append a schoolbook term
    def add_schoolbook(i, j, k, U, V, W, col_idx):
        """
        Adds a rank-1 term for A[i,j] * B[j,k] contributing to C[i,k].
        U is indexed by i*m+j
        V is indexed by j*p+k
        W is indexed by k*n+i
        """
        U.append([0] * (len(U[0]) if U else 0)) # Placeholder, we build lists of lists
        # Actually, it's easier to build U, V, W as lists of columns (vectors)
        # Or lists of rows? The interface expects U as (n*m, rank).
        # So U is a list of rows, each row is a list of rank elements.
        # Or we can build columns and transpose at the end.
        # Let's build columns: U_cols[r], V_cols[r], W_cols[r].
        pass

    # Let's build U, V, W as lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, c_i_start, c_i_end, c_k_start, c_k_end):
        """
        Adds schoolbook terms for A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]
        contributing to C[c_i_start:c_i_end, c_k_start:c_k_end].
        Note: i_start==c_i_start, k_end==c_k_end usually.
        """
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    
                    add_term(u, v, w)

    def add_strassen_block(i_start, j_start, k_start):
        """
        Adds Strassen terms for A[i_start:i_start+2, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]
        contributing to C[i_start:i_start+2, k_start:k_start+2].
        """
        # Local indices 0,1 map to global:
        # i0 = i_start, i1 = i_start+1
        # j0 = j_start, j1 = j_start+1
        # k0 = k_start, k1 = k_start+1
        
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        
        # 1. M1 = (P+Q)W -> C00, C11
        u1 = [0] * (n * m)
        v1 = [0] * (m * p)
        w1 = [0] * (p * n)
        u1[i0 * m + j0] = 1; u1[i0 * m + j1] = 1
        u1[i1 * m + j0] = 1; u1[i1 * m + j1] = 1
        v1[j0 * p + k0] = 1; v1[j1 * p + k0] = 1
        w1[k0 * n + i0] = 1
        w1[k1 * n + i1] = 1
        add_term(u1, v1, w1)
        
        # 2. M2 = P(Y-Z) -> C01, C10
        u2 = [0] * (n * m)
        v2 = [0] * (m * p)
        w2 = [0] * (p * n)
        u2[i0 * m + j0] = 1; u2[i0 * m + j1] = 1
        v2[j0 * p + k1] = 1; v2[j1 * p + k1] = -1
        w2[k1 * n + i0] = 1
        w2[k0 * n + i1] = 1
        add_term(u2, v2, w2)
        
        # 3. M3 = (R+Q)Z -> C01, C11
        u3 = [0] * (n * m)
        v3 = [0] * (m * p)
        w3 = [0] * (p * n)
        u3[i1 * m + j0] = 1; u3[i1 * m + j1] = 1
        u3[i0 * m + j1] = 1
        v3[j0 * p + k1] = 1; v3[j1 * p + k1] = 1
        w3[k1 * n + i0] = 1
        w3[k1 * n + i1] = 1
        add_term(u3, v3, w3)
        
        # 4. M4 = S(X-Y) -> C00, C11
        u4 = [0] * (n * m)
        v4 = [0] * (m * p)
        w4 = [0] * (p * n)
        u4[i1 * m + j0] = 1; u4[i1 * m + j1] = 1
        v4[j0 * p + k1] = 1; v4[j1 * p + k1] = -1
        w4[k0 * n + i0] = 1
        w4[k1 * n + i1] = 1
        add_term(u4, v4, w4)
        
        # 5. M5 = (P+R)Y -> C01, C10
        u5 = [0] * (n * m)
        v5 = [0] * (m * p)
        w5 = [0] * (p * n)
        u5[i0 * m + j0] = 1; u5[i0 * m + j1] = 1
        u5[i1 * m + j0] = 1; u5[i1 * m + j1] = 1
        v5[j0 * p + k0] = 1; v5[j1 * p + k0] = 1
        w5[k1 * n + i0] = 1
        w5[k0 * n + i1] = 1
        add_term(u5, v5, w5)
        
        # 6. M6 = (Q-S)(X+Z) -> C11
        u6 = [0] * (n * m)
        v6 = [0] * (m * p)
        w6 = [0] * (p * n)
        u6[i0 * m + j1] = 1
        u6[i1 * m + j1] = -1
        v6[j0 * p + k1] = 1; v6[j1 * p + k1] = 1
        w6[k1 * n + i1] = 1
        add_term(u6, v6, w6)
        
        # 7. M7 = (P-R)(W+X) -> C00, C11
        u7 = [0] * (n * m)
        v7 = [0] * (m * p)
        w7 = [0] * (p * n)
        u7[i0 * m + j0] = 1
        u7[i1 * m + j0] = -1
        v7[j0 * p + k0] = 1; v7[j1 * p + k0] = 1
        w7[k0 * n + i0] = 1
        w7[k1 * n + i1] = 1
        add_term(u7, v7, w7)

    # Define Blocks
    # A splits: i: [0,1], [2]; j: [0,1], [2,3]
    # B splits: j: [0,1], [2,3]; k: [0,1], [2,3], [4,5], [6]
    
    # C11 (i 0-1, k 0-1)
    # A11 (0-1, 0-1) * B11 (0-1, 0-1) -> Strassen
    add_strassen_block(0, 0, 0)
    # A12 (0-1, 2-3) * B21 (2-3, 0-1) -> Strassen
    add_strassen_block(0, 2, 0)
    
    # C12 (i 0-1, k 2-3)
    # A11 (0-1, 0-1) * B12 (0-1, 2-3) -> Strassen
    add_strassen_block(0, 0, 2)
    # A12 (0-1, 2-3) * B22 (2-3, 2-3) -> Strassen
    add_strassen_block(0, 2, 2)
    
    # C13 (i 0-1, k 4-5)
    # A11 (0-1, 0-1) * B13 (0-1, 4-5) -> Strassen
    add_strassen_block(0, 0, 4)
    # A12 (0-1, 2-3) * B23 (2-3, 4-5) -> Strassen
    add_strassen_block(0, 2, 4)
    
    # C14 (i 0-1, k 6)
    # A11 (0-1, 0-1) * B14 (0-1, 6) -> Schoolbook
    add_schoolbook_block(0, 2, 0, 2, 6, 7, 0, 2, 6, 7)
    # A12 (0-1, 2-3) * B24 (2-3, 6) -> Schoolbook
    add_schoolbook_block(0, 2, 2, 4, 6, 7, 0, 2, 6, 7)
    
    # C21 (i 2, k 0-1)
    # A21 (2, 0-1) * B11 (0-1, 0-1) -> Schoolbook
    add_schoolbook_block(2, 3, 0, 2, 0, 2, 2, 3, 0, 2)
    # A22 (2, 2-3) * B21 (2-3, 0-1) -> Schoolbook
    add_schoolbook_block(2, 3, 2, 4, 0, 2, 2, 3, 0, 2)
    
    # C22 (i 2, k 2-3)
    # A21 (2, 0-1) * B12 (0-1, 2-3) -> Schoolbook
    add_schoolbook_block(2, 3, 0, 2, 2, 4, 2, 3, 2, 4)
    # A22 (2, 2-3) * B22 (2-3, 2-3) -> Schoolbook
    add_schoolbook_block(2, 3, 2, 4, 2, 4, 2, 3, 2, 4)
    
    # C23 (i 2, k 4-5)
    # A21 (2, 0-1) * B13 (0-1, 4-5) -> Schoolbook
    add_schoolbook_block(2, 3, 0, 2, 4, 6, 2, 3, 4, 6)
    # A22 (2, 2-3) * B23 (2-3, 4-5) -> Schoolbook
    add_schoolbook_block(2, 3, 2, 4, 4, 6, 2, 3, 4, 6)
    
    # C24 (i 2, k 6)
    # A21 (2, 0-1) * B14 (0-1, 6) -> Schoolbook
    add_schoolbook_block(2, 3, 0, 2, 6, 7, 2, 3, 6, 7)
    # A22 (2, 2-3) * B24 (2-3, 6) -> Schoolbook
    add_schoolbook_block(2, 3, 2, 4, 6, 7, 2, 3, 6, 7)

    # Convert columns to rows
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
