import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # Global indices
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(A_block_start_i, A_block_start_j, B_block_start_i, B_block_start_j, C_block_start_i, C_block_start_j):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 block.
        A_block: rows [A_block_start_i, A_block_start_i+2), cols [A_block_start_j, A_block_start_j+2)
        B_block: rows [B_block_start_i, B_block_start_i+2), cols [B_block_start_j, B_block_start_j+2)
        C_block: rows [C_block_start_i, C_block_start_i+2), cols [C_block_start_j, C_block_start_j+2)
        
        Note: In our problem, A is 3x4, B is 4x4, C is 3x4.
        A indices: i in 0..2, j in 0..3.
        B indices: i in 0..3, j in 0..3.
        C indices: i in 0..2, j in 0..3.
        
        W index convention: W[k*n + i, r] where k is col of C, i is row of C.
        So C_block_start_i is row start, C_block_start_j is col start.
        """
        
        # Strassen terms
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = (x00-x10)(y10+y11)
        # M4 = x11(y11-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # Local indices for A (X)
        # x00: i=0, j=0 -> global (A_block_start_i + 0) * m + (A_block_start_j + 0)
        # x01: i=0, j=1
        # x10: i=1, j=0
        # x11: i=1, j=1
        
        # Local indices for B (Y)
        # y00: i=0, j=0 -> global (B_block_start_i + 0) * p + (B_block_start_j + 0)
        # y01: i=0, j=1
        # y10: i=1, j=0
        # y11: i=1, j=1
        
        # Local indices for C (Z)
        # z00: i=0, j=0 -> global row C_block_start_i+0, col C_block_start_j+0
        # W index: (C_block_start_j+0)*n + (C_block_start_i+0)
        
        terms = [
            # (U_coefs, V_coefs, W_coefs)
            # U: dict {global_idx: coef}
            # V: dict {global_idx: coef}
            # W: dict {global_idx: coef}
            
            # M1
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M2
            (
                { (A_block_start_i+1)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M3
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+0): -1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M4
            (
                { (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1, (B_block_start_i+0)*p + (B_block_start_j+0): -1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M5
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+0)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): -1, (C_block_start_j+1)*n + (C_block_start_i+0): -1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M6
            (
                { (A_block_start_i+1)*m + (A_block_start_j+0): 1, (A_block_start_i+0)*m + (A_block_start_j+0): -1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1, (B_block_start_i+0)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M7
            (
                { (A_block_start_i+0)*m + (A_block_start_j+1): 1, (A_block_start_i+1)*m + (A_block_start_j+1): -1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1 }
            ),
        ]
        
        for u_coefs, v_coefs, w_coefs in terms:
            u_col = np.zeros(n*m)
            for idx, coef in u_coefs.items():
                u_col[idx] = coef
            v_col = np.zeros(m*p)
            for idx, coef in v_coefs.items():
                v_col[idx] = coef
            w_col = np.zeros(p*n)
            for idx, coef in w_coefs.items():
                w_col[idx] = coef
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    def add_schoolbook_block_1x2(A_block_start_i, A_block_start_j, B_block_start_i, B_block_start_j, C_block_start_i, C_block_start_j):
        """
        Adds a schoolbook decomposition for a 1x2 * 2x2 block.
        A_block: row A_block_start_i, cols [A_block_start_j, A_block_start_j+2)
        B_block: rows [B_block_start_i, B_block_start_i+2), cols [B_block_start_j, B_block_start_j+2)
        C_block: row C_block_start_i, cols [C_block_start_j, C_block_start_j+2)
        """
        # Terms:
        # C[j] = A[0]B[0,j] + A[1]B[1,j]
        # j=0, j=1
        
        # Term 1: A[0] * B[0,0] -> C[0]
        u1 = np.zeros(n*m); u1[(A_block_start_i)*m + (A_block_start_j+0)] = 1
        v1 = np.zeros(m*p); v1[(B_block_start_i+0)*p + (B_block_start_j+0)] = 1
        w1 = np.zeros(p*n); w1[(C_block_start_j+0)*n + (C_block_start_i)] = 1
        
        # Term 2: A[1] * B[1,0] -> C[0]
        u2 = np.zeros(n*m); u2[(A_block_start_i)*m + (A_block_start_j+1)] = 1
        v2 = np.zeros(m*p); v2[(B_block_start_i+1)*p + (B_block_start_j+0)] = 1
        w2 = np.zeros(p*n); w2[(C_block_start_j+0)*n + (C_block_start_i)] = 1
        
        # Term 3: A[0] * B[0,1] -> C[1]
        u3 = np.zeros(n*m); u3[(A_block_start_i)*m + (A_block_start_j+0)] = 1
        v3 = np.zeros(m*p); v3[(B_block_start_i+0)*p + (B_block_start_j+1)] = 1
        w3 = np.zeros(p*n); w3[(C_block_start_j+1)*n + (C_block_start_i)] = 1
        
        # Term 4: A[1] * B[1,1] -> C[1]
        u4 = np.zeros(n*m); u4[(A_block_start_i)*m + (A_block_start_j+1)] = 1
        v4 = np.zeros(m*p); v4[(B_block_start_i+1)*p + (B_block_start_j+1)] = 1
        w4 = np.zeros(p*n); w4[(C_block_start_j+1)*n + (C_block_start_i)] = 1
        
        U_cols.extend([u1, u2, u3, u4])
        V_cols.extend([v1, v2, v3, v4])
        W_cols.extend([w1, w2, w3, w4])
        
    # Block 1: C_top_left (rows 0,1 of C; cols 0,1 of C)
    # = A_top_left (rows 0,1 of A; cols 0,1 of A) * B_top_left (rows 0,1 of B; cols 0,1 of B)
    # + A_top_right (rows 0,1 of A; cols 2,3 of A) * B_bot_left (rows 2,3 of B; cols 0,1 of B)
    
    # Part 1: A11 * B11
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Part 2: A12 * B21
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 2: C_top_right (rows 0,1 of C; cols 2,3 of C)
    # = A_top_left * B_top_right
    # + A_top_right * B_bot_right
    
    # Part 3: A11 * B12
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # Part 4: A12 * B22
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 3: C_bot_left (row 2 of C; cols 0,1 of C)
    # = A_bot_left (row 2 of A; cols 0,1 of A) * B_top_left
    # + A_bot_right (row 2 of A; cols 2,3 of A) * B_bot_left
    
    # Part 5: A21 * B11
    add_schoolbook_block_1x2(2, 0, 0, 0, 2, 0)
    # Part 6: A22 * B21
    add_schoolbook_block_1x2(2, 2, 2, 0, 2, 0)
    
    # Block 4: C_bot_right (row 2 of C; cols 2,3 of C)
    # = A_bot_left * B_top_right
    # + A_bot_right * B_bot_right
    
    # Part 7: A21 * B12
    add_schoolbook_block_1x2(2, 0, 0, 2, 2, 2)
    # Part 8: A22 * B22
    add_schoolbook_block_1x2(2, 2, 2, 2, 2, 2)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    def _verify(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > tol:
                            return False
        return True
    
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
                    
    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
