import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Target Rank: 72
    # Strategy:
    # Split A (4x4) into A1 (2x4) and A2 (2x4)
    # Split B (4x5) into B1 (4x2), B2 (4x2), B3 (4x1)
    # C1 = A1 * B1 (2x4 * 4x2) -> Rank 14 (Two 2x2x2 Strassen)
    # C2 = A1 * B2 (2x4 * 4x2) -> Rank 14
    # C3 = A1 * B3 (2x4 * 4x1) -> Rank 8 (Schoolbook)
    # C4 = A2 * B1 (2x4 * 4x2) -> Rank 14
    # C5 = A2 * B2 (2x4 * 4x2) -> Rank 14
    # C6 = A2 * B3 (2x4 * 4x1) -> Rank 8
    # Total Rank = 72
    
    rank = 72
    
    # U: (n*m, rank) = (16, 72)
    # V: (m*p, rank) = (20, 72)
    # W: (p*n, rank) = (20, 72)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a 2x2x2 Strassen decomposition
    # A_block is 2x2, B_block is 2x2, C_block is 2x2
    # Indices for A_block: rows r1, r2 in A, cols c1, c2 in A
    # Indices for B_block: rows r1, r2 in B, cols c1, c2 in B
    # Indices for C_block: rows r1, r2 in C, cols c1, c2 in C
    # Note: A is n x m (4x4). B is m x p (4x5). C is n x p (4x5).
    # A indices: row i (0-3), col j (0-3). Flat index i*m + j.
    # B indices: row j (0-3), col k (0-4). Flat index j*p + k.
    # C indices: row i (0-3), col k (0-4). Flat index k*n + i.
    
    def add_strassen_222(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_idx):
        """
        Adds a rank-7 decomposition for 2x2x2 multiplication.
        a_rows, a_cols: indices of the 2x2 block in A
        b_rows, b_cols: indices of the 2x2 block in B
        c_rows, c_cols: indices of the 2x2 block in C
        start_idx: current column index in U, V, W
        """
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices:
        a11_idx = a_rows[0] * m + a_cols[0]
        a12_idx = a_rows[0] * m + a_cols[1]
        a21_idx = a_rows[1] * m + a_cols[0]
        a22_idx = a_rows[1] * m + a_cols[1]
        
        # B indices:
        b11_idx = b_rows[0] * p + b_cols[0]
        b12_idx = b_rows[0] * p + b_cols[1]
        b21_idx = b_rows[1] * p + b_cols[0]
        b22_idx = b_rows[1] * p + b_cols[1]
        
        # C indices:
        c11_idx = c_cols[0] * n + c_rows[0]
        c12_idx = c_cols[1] * n + c_rows[0]
        c21_idx = c_cols[0] * n + c_rows[1]
        c22_idx = c_cols[1] * n + c_rows[1]
        
        # M1
        U[a11_idx, start_idx] += 1
        U[a22_idx, start_idx] += 1
        V[b11_idx, start_idx] += 1
        V[b22_idx, start_idx] += 1
        W[c11_idx, start_idx] += 1
        W[c22_idx, start_idx] += 1
        
        # M2
        U[a21_idx, start_idx+1] += 1
        U[a22_idx, start_idx+1] += 1
        V[b11_idx, start_idx+1] += 1
        W[c21_idx, start_idx+1] += 1
        W[c22_idx, start_idx+1] -= 1
        
        # M3
        U[a11_idx, start_idx+2] += 1
        V[b21_idx, start_idx+2] += 1
        V[b22_idx, start_idx+2] -= 1
        W[c12_idx, start_idx+2] += 1
        W[c22_idx, start_idx+2] += 1
        
        # M4
        U[a22_idx, start_idx+3] += 1
        V[b12_idx, start_idx+3] += 1
        V[b11_idx, start_idx+3] -= 1
        W[c11_idx, start_idx+3] += 1
        W[c21_idx, start_idx+3] += 1
        
        # M5
        U[a11_idx, start_idx+4] += 1
        U[a12_idx, start_idx+4] += 1
        V[b22_idx, start_idx+4] += 1
        W[c11_idx, start_idx+4] -= 1
        W[c12_idx, start_idx+4] += 1
        
        # M6
        U[a21_idx, start_idx+5] += 1
        U[a11_idx, start_idx+5] -= 1
        V[b11_idx, start_idx+5] += 1
        V[b12_idx, start_idx+5] += 1
        W[c22_idx, start_idx+5] += 1
        
        # M7
        U[a12_idx, start_idx+6] += 1
        U[a22_idx, start_idx+6] -= 1
        V[b21_idx, start_idx+6] += 1
        V[b22_idx, start_idx+6] += 1
        W[c11_idx, start_idx+6] += 1
        
        return start_idx + 7

    def add_schoolbook_241(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_idx):
        """
        Adds a rank-8 decomposition for 2x4 * 4x1 multiplication.
        a_rows: 2 rows in A
        a_cols: 4 cols in A
        b_rows: 4 rows in B (matches a_cols)
        b_cols: 1 col in B
        c_rows: 2 rows in C
        c_cols: 1 col in C
        """
        for i in range(2):
            for j in range(4):
                a_idx = a_rows[i] * m + a_cols[j]
                b_idx = b_rows[j] * p + b_cols[0]
                c_idx = c_cols[0] * n + c_rows[i]
                
                U[a_idx, start_idx] = 1
                V[b_idx, start_idx] = 1
                W[c_idx, start_idx] = 1
                start_idx += 1
        return start_idx

    # Define blocks
    # A1: rows 0,1; cols 0,1,2,3
    # A2: rows 2,3; cols 0,1,2,3
    a1_rows = [0, 1]
    a1_cols = [0, 1, 2, 3]
    a2_rows = [2, 3]
    a2_cols = [0, 1, 2, 3]
    
    # B1: rows 0,1,2,3; cols 0,1
    # B2: rows 0,1,2,3; cols 2,3
    # B3: rows 0,1,2,3; cols 4
    b1_rows = [0, 1, 2, 3]
    b1_cols = [0, 1]
    b2_rows = [0, 1, 2, 3]
    b2_cols = [2, 3]
    b3_rows = [0, 1, 2, 3]
    b3_cols = [4]
    
    # C1: rows 0,1; cols 0,1
    # C2: rows 0,1; cols 2,3
    # C3: rows 0,1; cols 4
    # C4: rows 2,3; cols 0,1
    # C5: rows 2,3; cols 2,3
    # C6: rows 2,3; cols 4
    c1_rows = [0, 1]
    c1_cols = [0, 1]
    c2_rows = [0, 1]
    c2_cols = [2, 3]
    c3_rows = [0, 1]
    c3_cols = [4]
    c4_rows = [2, 3]
    c4_cols = [0, 1]
    c5_rows = [2, 3]
    c5_cols = [2, 3]
    c6_rows = [2, 3]
    c6_cols = [4]
    
    idx = 0
    
    # C1 = A1 * B1 (2x4 * 4x2)
    # Split into two 2x2x2 blocks?
    # A1 = [A1_left, A1_right] where A1_left is 2x2 (cols 0,1), A1_right is 2x2 (cols 2,3)
    # B1 = [B1_top; B1_bottom] where B1_top is 2x2 (rows 0,1), B1_bottom is 2x2 (rows 2,3)
    # C1 = A1_left * B1_top + A1_right * B1_bottom
    
    # Block 1: A1_left (rows 0,1; cols 0,1) * B1_top (rows 0,1; cols 0,1) -> C1_top_left (rows 0,1; cols 0,1)
    # Wait, C1 is 2x2.
    # C1[0,0] = sum A1[i,0]B1[0,k] + A1[i,1]B1[1,k] ...
    # Actually, C1 = A1 * B1.
    # A1 is 2x4. B1 is 4x2.
    # Let's split the inner dimension 4 into two 2s.
    # A1 = [A1a, A1b] (2x2 each)
    # B1 = [B1a; B1b] (2x2 each)
    # C1 = A1a * B1a + A1b * B1b
    
    # A1a: rows 0,1; cols 0,1
    # A1b: rows 0,1; cols 2,3
    # B1a: rows 0,1; cols 0,1
    # B1b: rows 2,3; cols 0,1
    # C1: rows 0,1; cols 0,1
    
    idx = add_strassen_222(
        a_rows=[0,1], a_cols=[0,1],
        b_rows=[0,1], b_cols=[0,1],
        c_rows=[0,1], c_cols=[0,1],
        start_idx=idx
    )
    idx = add_strassen_222(
        a_rows=[0,1], a_cols=[2,3],
        b_rows=[2,3], b_cols=[0,1],
        c_rows=[0,1], c_cols=[0,1],
        start_idx=idx
    )
    
    # C2 = A1 * B2 (2x4 * 4x2)
    # A1 = [A1a, A1b]
    # B2 = [B2a; B2b] where B2a is rows 0,1 cols 2,3; B2b is rows 2,3 cols 2,3
    # C2 = A1a * B2a + A1b * B2b
    # C2 is 2x2 (rows 0,1; cols 2,3)
    
    idx = add_strassen_222(
        a_rows=[0,1], a_cols=[0,1],
        b_rows=[0,1], b_cols=[2,3],
        c_rows=[0,1], c_cols=[2,3],
        start_idx=idx
    )
    idx = add_strassen_222(
        a_rows=[0,1], a_cols=[2,3],
        b_rows=[2,3], b_cols=[2,3],
        c_rows=[0,1], c_cols=[2,3],
        start_idx=idx
    )
    
    # C3 = A1 * B3 (2x4 * 4x1)
    # A1 is 2x4. B3 is 4x1.
    # Schoolbook rank 8.
    idx = add_schoolbook_241(
        a_rows=[0,1], a_cols=[0,1,2,3],
        b_rows=[0,1,2,3], b_cols=[4],
        c_rows=[0,1], c_cols=[4],
        start_idx=idx
    )
    
    # C4 = A2 * B1 (2x4 * 4x2)
    # A2 = [A2a, A2b] where A2a is rows 2,3 cols 0,1; A2b is rows 2,3 cols 2,3
    # B1 = [B1a; B1b]
    # C4 = A2a * B1a + A2b * B1b
    # C4 is 2x2 (rows 2,3; cols 0,1)
    
    idx = add_strassen_222(
        a_rows=[2,3], a_cols=[0,1],
        b_rows=[0,1], b_cols=[0,1],
        c_rows=[2,3], c_cols=[0,1],
        start_idx=idx
    )
    idx = add_strassen_222(
        a_rows=[2,3], a_cols=[2,3],
        b_rows=[2,3], b_cols=[0,1],
        c_rows=[2,3], c_cols=[0,1],
        start_idx=idx
    )
    
    # C5 = A2 * B2 (2x4 * 4x2)
    # A2 = [A2a, A2b]
    # B2 = [B2a; B2b]
    # C5 = A2a * B2a + A2b * B2b
    # C5 is 2x2 (rows 2,3; cols 2,3)
    
    idx = add_strassen_222(
        a_rows=[2,3], a_cols=[0,1],
        b_rows=[0,1], b_cols=[2,3],
        c_rows=[2,3], c_cols=[2,3],
        start_idx=idx
    )
    idx = add_strassen_222(
        a_rows=[2,3], a_cols=[2,3],
        b_rows=[2,3], b_cols=[2,3],
        c_rows=[2,3], c_cols=[2,3],
        start_idx=idx
    )
    
    # C6 = A2 * B3 (2x4 * 4x1)
    # A2 is 2x4. B3 is 4x1.
    # Schoolbook rank 8.
    idx = add_schoolbook_241(
        a_rows=[2,3], a_cols=[0,1,2,3],
        b_rows=[0,1,2,3], b_cols=[4],
        c_rows=[2,3], c_cols=[4],
        start_idx=idx
    )
    
    # Verify idx is 72
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")
        
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_dot_a = np.dot(U[:, r], A.flatten())
                    v_dot_b = np.dot(V[:, r], B.flatten())
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True
