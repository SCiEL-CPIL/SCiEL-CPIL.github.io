import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Strategy: Recursive Block Decomposition
    # 1. Split m=4 into m1=2, m2=2
    # 2. For each m_block, split n=3 into n1=1, n2=2
    # 3. For n1=1, m=2, p=6: Schoolbook (Rank 12)
    # 4. For n2=2, m=2, p=6: Split p=6 into p1=2, p2=2, p3=2
    #    Each 2x2x2 block uses Strassen (Rank 7)
    # Total Rank = 2 * (12 + 3 * 7) = 2 * 33 = 66
    
    total_rank = 66
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (12, 66)
    # V: (m*p, rank) -> (24, 66)
    # W: (p*n, rank) -> (18, 66)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add a schoolbook decomposition for a sub-block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Helper to add a Strassen decomposition for a 2x2x2 sub-block
    # i_start, j_start, k_start are the top-left indices
    def add_strassen(i_start, j_start, k_start):
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # A is 2x2, B is 2x2, C is 2x2
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M1 - M2 + M6
        # C21 = M2 + M4 - M3 + M5
        # C22 = M1 - M4 + M2 - M6
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is at (i_start, j_start), A22 is at (i_start+1, j_start+1)
        # B11 is at (j_start, k_start), B22 is at (j_start+1, k_start+1)
        
        # We need to construct U, V, W for each M_k such that
        # M_k = (sum U_A * A) * (sum V_B * B)
        
        # M1: A_part = A11 + A22, B_part = B11 + B22
        U[i_start * m + j_start, col] = 1
        U[(i_start + 1) * m + (j_start + 1), col] = 1
        V[j_start * p + k_start, col] = 1
        V[(j_start + 1) * p + (k_start + 1), col] = 1
        # W contributions:
        W[k_start * n + i_start, col] = 1       # C11
        W[(k_start + 1) * n + i_start, col] = 1   # C12
        W[k_start * n + (i_start + 1), col] = 1   # C21
        W[(k_start + 1) * n + (i_start + 1), col] = 1 # C22
        col += 1
        
        # M2: A_part = A21 + A22, B_part = B11
        U[(i_start + 1) * m + j_start, col] = 1
        U[(i_start + 1) * m + (j_start + 1), col] = 1
        V[j_start * p + k_start, col] = 1
        # W contributions:
        # C12 += -M2, C21 += M2, C22 += M2
        W[(k_start + 1) * n + i_start, col] = -1
        W[k_start * n + (i_start + 1), col] = 1
        W[(k_start + 1) * n + (i_start + 1), col] = 1
        col += 1
        
        # M3: A_part = A11, B_part = B12 - B22
        U[i_start * m + j_start, col] = 1
        V[j_start * p + (k_start + 1), col] = 1
        V[(j_start + 1) * p + (k_start + 1), col] = -1
        # W contributions:
        # C12 += M3, C21 += -M3
        W[(k_start + 1) * n + i_start, col] = 1
        W[k_start * n + (i_start + 1), col] = -1
        col += 1
        
        # M4: A_part = A22, B_part = B21 - B11
        U[(i_start + 1) * m + (j_start + 1), col] = 1
        V[(j_start + 1) * p + k_start, col] = 1
        V[j_start * p + k_start, col] = -1
        # W contributions:
        # C11 += M4, C21 += M4, C22 += -M4
        W[k_start * n + i_start, col] = 1
        W[k_start * n + (i_start + 1), col] = 1
        W[(k_start + 1) * n + (i_start + 1), col] = -1
        col += 1
        
        # M5: A_part = A11 + A12, B_part = B22
        U[i_start * m + j_start, col] = 1
        U[i_start * m + (j_start + 1), col] = 1
        V[(j_start + 1) * p + (k_start + 1), col] = 1
        # W contributions:
        # C11 += -M5, C21 += M5
        W[k_start * n + i_start, col] = -1
        W[k_start * n + (i_start + 1), col] = 1
        col += 1
        
        # M6: A_part = A21 - A11, B_part = B11 + B12
        U[(i_start + 1) * m + j_start, col] = 1
        U[i_start * m + j_start, col] = -1
        V[j_start * p + k_start, col] = 1
        V[j_start * p + (k_start + 1), col] = 1
        # W contributions:
        # C12 += M6, C22 += -M6
        W[(k_start + 1) * n + i_start, col] = 1
        W[(k_start + 1) * n + (i_start + 1), col] = -1
        col += 1
        
        # M7: A_part = A12 - A22, B_part = B21 + B22
        U[i_start * m + (j_start + 1), col] = 1
        U[(i_start + 1) * m + (j_start + 1), col] = -1
        V[(j_start + 1) * p + k_start, col] = 1
        V[(j_start + 1) * p + (k_start + 1), col] = 1
        # W contributions:
        # C11 += M7, C22 += M7
        W[k_start * n + i_start, col] = 1
        W[(k_start + 1) * n + (i_start + 1), col] = 1
        col += 1

    # Iterate over the two m-blocks (j_start = 0 and 2)
    for j_block in range(2):
        j_start = j_block * 2
        
        # n-block 1: i_start=0, i_end=1 (size 1)
        # p is full 0..6
        # This is a 1x2x6 block
        # We can just use schoolbook for this 1x2x6 block
        add_schoolbook(0, 1, j_start, j_start + 2, 0, 6)
        
        # n-block 2: i_start=1, i_end=3 (size 2)
        # p is split into three 2s: 0..2, 2..4, 4..6
        for k_block in range(3):
            k_start = k_block * 2
            # This is a 2x2x2 block
            add_strassen(1, j_start, k_start)
            
    # Verify
    def verify(U, V, W):
        rank = U.shape[1]
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = sum(U[i * m + j, r] * A[i, j] for j in range(m))
                    v_val = sum(V[j * p + k, r] * B[j, k] for j in range(m))
                    C_computed[i, k] += W[k * n + i, r] * u_val * v_val
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        total_rank = schoolbook_rank

    return {"U": U, "V": V, "W": W}
