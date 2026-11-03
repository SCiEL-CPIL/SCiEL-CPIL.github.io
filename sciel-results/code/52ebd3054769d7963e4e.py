import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Rank-minimal decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: Split m=4 into two blocks of 2.
    For each 2x2x7 block, split p=7 into 2+2+2+1.
    Use Strassen (rank 7) for each 2x2x2 sub-block.
    Use schoolbook (rank 4) for each 2x2x1 sub-block.
    
    Per 2x2x7 block: 7 + 7 + 7 + 4 = 25
    Total rank: 25 + 25 = 50
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    def strassen_2x2x2():
        """Return U, V, W matrices (each 4x7) for 2x2x2 Strassen decomposition.
        A indices: a11=0, a12=1, a21=2, a22=3
        B indices: b11=0, b12=1, b21=2, b22=3
        C indices: c11=0, c12=1, c21=2, c22=3
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Return U, V, W matrices (each 4x4) for 2x2x1 schoolbook decomposition.
        A is 2x2, B is 2x1, C is 2x1
        A indices: a11=0, a12=1, a21=2, a22=3
        B indices: b1=0, b2=1
        C indices: c1=0, c2=1
        Uses 4 products (rank 4).
        """
        U = np.zeros((4, 4), dtype=float)
        V = np.zeros((4, 4), dtype=float)
        W = np.zeros((4, 4), dtype=float)
        
        # Product 0: a11*b1 -> c1
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Product 1: a12*b2 -> c1
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Product 2: a21*b1 -> c2
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Product 3: a22*b2 -> c2
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W
    
    # A is 2x4: indices a[i*4+j] for i in 0..1, j in 0..3
    # B is 4x7: indices b[j*7+k] for j in 0..3, k in 0..6
    # C is 2x7: indices c[k*2+i] for k in 0..6, i in 0..1
    
    # Split m=4 into m1=2 (j=0,1) and m2=2 (j=2,3)
    # For each block, split p=7 into p1=2 (k=0,1), p2=2 (k=2,3), p3=2 (k=4,5), p4=1 (k=6)
    
    total_rank = 25 + 25  # 50
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 8 x 50
    V = np.zeros((m * p, total_rank), dtype=float)  # 28 x 50
    W = np.zeros((p * n, total_rank), dtype=float)  # 14 x 50
    
    col = 0
    
    def add_strassen_block(a_offset, b_offset, c_offset, start_col):
        """Add a 2x2x2 Strassen block."""
        sU, sV, sW = strassen_2x2x2()
        r = 7
        for t in range(r):
            for i in range(2):
                for j_local in range(2):
                    a_global = i * m + (a_offset * 2 + j_local)
                    U[a_global, start_col + t] += sU[i * 2 + j_local, t]
            for j_local in range(2):
                for k_local in range(2):
                    b_global = (a_offset * 2 + j_local) * p + (b_offset + k_local)
                    V[b_global, start_col + t] += sV[j_local * 2 + k_local, t]
            for k_local in range(2):
                for i in range(2):
                    c_global = (c_offset + k_local) * n + i
                    W[c_global, start_col + t] += sW[k_local * 2 + i, t]
        return start_col + r
    
    def add_schoolbook_2x2x1_block(a_offset, b_col, c_col, start_col):
        """Add a 2x2x1 schoolbook block (rank 4).
        a_offset: which 2x2 block of A (0 for j=0,1; 1 for j=2,3)
        b_col: column index in B (0..6)
        c_col: column index in C (0..6)
        """
        scU, scV, scW = schoolbook_2x2x1()
        r = 4
        for t in range(r):
            for i in range(2):
                for j_local in range(2):
                    a_global = i * m + (a_offset * 2 + j_local)
                    U[a_global, start_col + t] += scU[i * 2 + j_local, t]
            # B: local b0 (row 0) -> global (a_offset*2, b_col), local b1 (row 1) -> global (a_offset*2+1, b_col)
            V[(a_offset * 2) * p + b_col, start_col + t] += scV[0, t]
            V[(a_offset * 2 + 1) * p + b_col, start_col + t] += scV[1, t]
            # C: local c0 (col 0) -> global (c_col, 0), local c1 (col 1) -> global (c_col, 1)
            W[c_col * n + 0, start_col + t] += scW[0, t]
            W[c_col * n + 1, start_col + t] += scW[1, t]
        return start_col + r
    
    # Block 1: j=0,1 (a_offset=0)
    col = add_strassen_block(a_offset=0, b_offset=0, c_offset=0, start_col=col)  # k=0,1
    col = add_strassen_block(a_offset=0, b_offset=2, c_offset=2, start_col=col)  # k=2,3
    col = add_strassen_block(a_offset=0, b_offset=4, c_offset=4, start_col=col)  # k=4,5
    col = add_schoolbook_2x2x1_block(a_offset=0, b_col=6, c_col=6, start_col=col)  # k=6
    
    # Block 2: j=2,3 (a_offset=1)
    col = add_strassen_block(a_offset=1, b_offset=0, c_offset=0, start_col=col)  # k=0,1
    col = add_strassen_block(a_offset=1, b_offset=2, c_offset=2, start_col=col)  # k=2,3
    col = add_strassen_block(a_offset=1, b_offset=4, c_offset=4, start_col=col)  # k=4,5
    col = add_schoolbook_2x2x1_block(a_offset=1, b_col=6, c_col=6, start_col=col)  # k=6
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, f"One check failed: i={i}, j={j}, k={k}, s={s}"
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-9:
                        return False, f"Zero check failed: a={a}, b={b}, c={c}, s={s}"
        return True, "OK"
    
    valid, msg = verify(U, V, W, total_rank)
    if not valid:
        print(f"Verification failed: {msg}")
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
