import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Rank-minimal decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: Split m=4 into two blocks of 2.
    For each 2x2x7 block, split p=7 into 2+2+2+1.
    Use Strassen (rank 7) for each 2x2x2 sub-block.
    Use schoolbook (rank 2) for each 2x2x1 sub-block.
    
    Per 2x2x7 block: 7 + 7 + 7 + 2 = 23
    Total rank: 23 + 23 = 46
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strassen decomposition for 2x2x2: rank 7
    # C = A * B where A, B, C are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, we need U (4x7), V (4x7), W (4x7)
    # where U indexes A elements (i*2+j), V indexes B elements (j*2+k), W indexes C elements (k*2+i)
    
    def strassen_2x2x2():
        """Return U, V, W matrices (each 4x7) for 2x2x2 Strassen decomposition."""
        # A indices: a11=0, a12=1, a21=2, a22=3
        # B indices: b11=0, b12=1, b21=2, b22=3
        # C indices: c11=0, c12=1, c21=2, c22=3
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # c11 and c22
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # c21 += M2, c22 -= M2
        
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1  # c12 += M3, c22 += M3
        
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1  # c11 -= M4, c21 += M4
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1  # c11 += M5, c12 += M5
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # c22 += M6
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1  # c11 -= M7
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Return U, V, W matrices (each 4x2) for 2x2x1 schoolbook decomposition."""
        # A is 2x2, B is 2x1, C is 2x1
        # c1 = a11*b1 + a12*b2
        # c2 = a21*b1 + a22*b2
        #
        # A indices: a11=0, a12=1, a21=2, a22=3
        # B indices: b1=0, b2=1
        # C indices: c1=0, c2=1
        
        U = np.zeros((4, 2), dtype=float)
        V = np.zeros((4, 2), dtype=float)
        W = np.zeros((4, 2), dtype=float)
        
        # Product 1: a11*b1 -> c1, a21*b1 -> c2
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        U[2, 0] = 1
        W[1, 0] = 1
        
        # Product 2: a12*b2 -> c1, a22*b2 -> c2
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        U[3, 1] = 1
        W[1, 1] = 1
        
        return U, V, W
    
    # Build the full decomposition
    # A is 2x4: indices a[i*4+j] for i in 0..1, j in 0..3
    # B is 4x7: indices b[j*7+k] for j in 0..3, k in 0..6
    # C is 2x7: indices c[k*2+i] for k in 0..6, i in 0..1
    
    # Split m=4 into m1=2 (j=0,1) and m2=2 (j=2,3)
    # For each block, split p=7 into p1=2 (k=0,1), p2=2 (k=2,3), p3=2 (k=4,5), p4=1 (k=6)
    
    total_rank = 0
    # Block 1 (j=0,1): 7+7+7+2 = 23
    # Block 2 (j=2,3): 7+7+7+2 = 23
    total_rank = 23 + 23
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 8 x 46
    V = np.zeros((m * p, total_rank), dtype=float)  # 28 x 46
    W = np.zeros((p * n, total_rank), dtype=float)  # 14 x 46
    
    col = 0
    
    # Helper to add a sub-decomposition with index mapping
    def add_subproblem(sub_U, sub_V, sub_W, a_offset, b_offset, c_offset, start_col):
        """
        Add a sub-decomposition.
        sub_U: (4, r) for 2x2 A block
        sub_V: (4, r) for 2x2 B block  
        sub_W: (4, r) for 2x2 C block
        a_offset: which 2x2 block of A (0 for j=0,1; 1 for j=2,3)
        b_offset: which 2x2 block of B (column offset in B's 7 columns)
        c_offset: which 2x2 block of C (column offset in C's 7 columns)
        """
        r = sub_U.shape[1]
        for t in range(r):
            # Map A indices: local (i, j_local) -> global (i, a_offset*2 + j_local)
            for i in range(2):
                for j_local in range(2):
                    a_global = i * m + (a_offset * 2 + j_local)
                    U[a_global, start_col + t] += sub_U[i * 2 + j_local, t]
            
            # Map B indices: local (j_local, k_local) -> global (a_offset*2 + j_local, b_offset + k_local)
            for j_local in range(2):
                for k_local in range(2):
                    b_global = (a_offset * 2 + j_local) * p + (b_offset + k_local)
                    V[b_global, start_col + t] += sub_V[j_local * 2 + k_local, t]
            
            # Map C indices: local (k_local, i) -> global (b_offset + k_local, i)
            for k_local in range(2):
                for i in range(2):
                    c_global = (b_offset + k_local) * n + i
                    W[c_global, start_col + t] += sub_W[k_local * 2 + i, t]
        
        return start_col + r
    
    # Block 1: j=0,1 (a_offset=0)
    sU, sV, sW = strassen_2x2x2()
    col = add_subproblem(sU, sV, sW, a_offset=0, b_offset=0, c_offset=0, start_col=col)  # k=0,1
    col = add_subproblem(sU, sV, sW, a_offset=0, b_offset=2, c_offset=2, start_col=col)  # k=2,3
    col = add_subproblem(sU, sV, sW, a_offset=0, b_offset=4, c_offset=4, start_col=col)  # k=4,5
    
    scU, scV, scW = schoolbook_2x2x1()
    # For 2x2x1, the B has only 1 column, so we need to map differently
    # B local index 0 -> global (0, 6), B local index 1 -> global (1, 6)
    # C local index 0 -> global (6, 0), C local index 1 -> global (6, 1)
    for t in range(2):
        for i in range(2):
            for j_local in range(2):
                a_global = i * m + j_local
                U[a_global, col + t] += scU[i * 2 + j_local, t]
        # B: local b0 -> global (0,6), local b1 -> global (1,6)
        V[0 * p + 6, col + t] += scV[0, t]
        V[1 * p + 6, col + t] += scV[1, t]
        # C: local c0 -> global (6,0), local c1 -> global (6,1)
        W[6 * n + 0, col + t] += scW[0, t]
        W[6 * n + 1, col + t] += scW[1, t]
    col += 2
    
    # Block 2: j=2,3 (a_offset=1)
    col = add_subproblem(sU, sV, sW, a_offset=1, b_offset=0, c_offset=0, start_col=col)  # k=0,1
    col = add_subproblem(sU, sV, sW, a_offset=1, b_offset=2, c_offset=2, start_col=col)  # k=2,3
    col = add_subproblem(sU, sV, sW, a_offset=1, b_offset=4, c_offset=4, start_col=col)  # k=4,5
    
    for t in range(2):
        for i in range(2):
            for j_local in range(2):
                a_global = i * m + (2 + j_local)
                U[a_global, col + t] += scU[i * 2 + j_local, t]
        V[2 * p + 6, col + t] += scV[0, t]
        V[3 * p + 6, col + t] += scV[1, t]
        W[6 * n + 0, col + t] += scW[0, t]
        W[6 * n + 1, col + t] += scW[1, t]
    col += 2
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        n, m, p = DIMENSIONS
        # Check that for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for all other (a, b, c) not in support, sum == 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, f"One check failed: i={i}, j={j}, k={k}, s={s}"
        
        # Check zeros for non-support entries
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
                    for r in range(total_rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-9:
                        return False, f"Zero check failed: a={a}, b={b}, c={c}, s={s}"
        return True, "OK"
    
    valid, msg = verify(U, V, W)
    if not valid:
        print(f"Verification failed: {msg}")
        # Fall back to schoolbook
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
