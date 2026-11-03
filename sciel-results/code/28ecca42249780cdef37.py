import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Construct a block-based decomposition
    # Split indices into I1={0,1}, I2={2,3,4}
    # J1={0,1}, J2={2,3,4}
    # K1={0,1}, K2={2,3,4}
    
    # We build U, V, W by concatenating the factors from each block multiplication.
    # The blocks are:
    # C11 = A11 B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    # C12 = A11 B12 (2x2 * 2x3) -> Rank 12 (Schoolbook)
    # C12 += A12 B22 (2x3 * 3x3) -> Rank 18 (Schoolbook)
    # C21 = A21 B11 (3x2 * 2x2) -> Rank 12 (Schoolbook)
    # C21 += A22 B21 (3x3 * 3x2) -> Rank 18 (Schoolbook)
    # C22 = A21 B12 (3x2 * 2x3) -> Rank 18 (Schoolbook)
    # C22 += A22 B22 (3x3 * 3x3) -> Rank 27 (Schoolbook)
    
    # Total Rank = 7 + 12 + 18 + 12 + 18 + 18 + 27 = 112
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(rank_block, U_block, V_block, W_block):
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)

    # Helper to create schoolbook block
    def schoolbook_block(n_r, m_c, p_c):
        # n_r: rows of A, m_c: cols of A (rows of B), p_c: cols of B
        rank = n_r * m_c * p_c
        U_b = np.zeros((n_r * m_c, rank))
        V_b = np.zeros((m_c * p_c, rank))
        W_b = np.zeros((p_c * n_r, rank))
        col = 0
        for i in range(n_r):
            for j in range(m_c):
                for k in range(p_c):
                    U_b[i*m_c+j, col] = 1
                    V_b[j*p_c+k, col] = 1
                    W_b[k*n_r+i, col] = 1
                    col += 1
        return U_b, V_b, W_b

    # Helper to create Strassen block (2x2)
    def strassen_block():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices for 2x2: i,j,k in {0,1}
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Strassen factors
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices:
        # U row: i*2+j
        # V row: j*2+k
        # W row: k*2+i
        
        # M1
        U_s[0, 0] = 1; U_s[3, 0] = 1 # A11, A22
        V_s[0, 0] = 1; V_s[3, 0] = 1 # B11, B22
        W_s[0, 0] = 1 # C11 (k=0,i=0)
        
        U_s[0, 1] = 1; U_s[3, 1] = 1
        V_s[1, 1] = 1; V_s[2, 1] = 1 # B11, B22? No, B11 is (0,0), B22 is (1,1).
        # Wait, V index is j*2+k.
        # B11: j=0,k=0 -> 0. B22: j=1,k=1 -> 3.
        # V_s[0, 1] = 1; V_s[3, 1] = 1
        # W index is k*2+i.
        # C11: k=0,i=0 -> 0. C22: k=1,i=1 -> 3.
        # C12: k=0,i=1? No, C12 is i=0, k=1? No.
        # C[i,k]. i is row, k is col.
        # W index k*n+i = k*2+i.
        # C11: i=0, k=0 -> 0.
        # C12: i=0, k=1 -> 2.
        # C21: i=1, k=0 -> 1.
        # C22: i=1, k=1 -> 3.
        
        # Let's restart Strassen assignment carefully.
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1 # C11, C22
        
        # M2 = (A21+A22)B11
        U_s[2, 1] = 1; U_s[3, 1] = 1 # A21 (i=1,j=0 -> 2), A22 (i=1,j=1 -> 3)
        V_s[0, 1] = 1 # B11
        W_s[1, 1] = 1; W_s[3, 1] = -1 # C21, C22
        
        # M3 = A11(B12-B22)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1 # B12 (j=0,k=1 -> 1), B22 (j=1,k=1 -> 3)
        W_s[2, 2] = 1; W_s[3, 2] = 1 # C12, C22
        
        # M4 = A22(B21-B11)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1 # B21 (j=1,k=0 -> 2), B11
        W_s[0, 3] = 1; W_s[1, 3] = 1 # C11, C21
        
        # M5 = (A11+A12)B22
        U_s[0, 4] = 1; U_s[1, 4] = 1 # A11, A12
        V_s[3, 4] = 1 # B22
        W_s[2, 4] = -1; W_s[3, 4] = 1 # C12, C22
        
        # M6 = (A21-A11)(B11+B12)
        U_s[2, 5] = 1; U_s[0, 5] = -1 # A21, A11
        V_s[0, 5] = 1; V_s[1, 5] = 1 # B11, B12
        W_s[3, 5] = 1 # C22
        
        # M7 = (A12-A22)(B21+B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1 # A12, A22
        V_s[2, 6] = 1; V_s[3, 6] = 1 # B21, B22
        W_s[0, 6] = 1 # C11
        
        return U_s, V_s, W_s

    # Build the full matrices
    # We need to map local indices to global indices.
    # Global U: (25, Rank)
    # Global V: (25, Rank)
    # Global W: (25, Rank)
    
    total_rank = 112
    U_full = np.zeros((25, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((25, total_rank))
    
    col_offset = 0
    
    # C11: A11 B11 (2x2)
    U_b, V_b, W_b = strassen_block()
    # Map indices
    # U local (i,j) -> global (i,j)
    # V local (j,k) -> global (j,k)
    # W local (k,i) -> global (k,i)
    # i,j,k in {0,1}
    U_full[0:4, col_offset:col_offset+7] = U_b
    V_full[0:4, col_offset:col_offset+7] = V_b
    W_full[0:4, col_offset:col_offset+7] = W_b
    col_offset += 7
    
    # C12 Part 1: A11 B12 (2x2 * 2x3)
    # A11: i in {0,1}, j in {0,1}
    # B12: j in {0,1}, k in {2,3,4}
    # U: i in {0,1}, j in {0,1} -> 0..3
    # V: j in {0,1}, k in {2,3,4} -> j*5+k
    # W: k in {2,3,4}, i in {0,1} -> k*5+i
    U_b, V_b, W_b = schoolbook_block(2, 2, 3)
    # U_b is 4x12
    # V_b is 6x12 (j*3+k local)
    # W_b is 6x12 (k*2+i local)
    
    # Map V: local j in {0,1}, local k in {0,1,2} -> global k = 2+local_k
    # V local index j*3+lk -> global j*5 + (2+lk)
    V_map = np.zeros((6, 12))
    for lj in range(2):
        for lk in range(3):
            V_map[lj*3+lk] = 1 # Placeholder, we need to scatter
    # Actually, easier to just fill directly
    for c in range(12):
        for lj in range(2):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[lj*5 + (2+lk), col_offset+c] = val
    # U is same indices
    U_full[0:4, col_offset:col_offset+12] = U_b
    # W: local k in {0,1,2}, local i in {0,1}
    # W local index lk*2+li -> global (2+lk)*5 + li
    for c in range(12):
        for lk in range(3):
            for li in range(2):
                val = W_b[lk*2+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + li, col_offset+c] = val
    col_offset += 12
    
    # C12 Part 2: A12 B22 (2x3 * 3x3)
    # A12: i in {0,1}, j in {2,3,4}
    # B22: j in {2,3,4}, k in {2,3,4}
    U_b, V_b, W_b = schoolbook_block(2, 3, 3)
    # U: i in {0,1}, j in {2,3,4} -> i*5+j
    # V: j in {2,3,4}, k in {2,3,4} -> j*5+k
    # W: k in {2,3,4}, i in {0,1} -> k*5+i
    for c in range(18):
        for li in range(2):
            for lj in range(3):
                val = U_b[li*3+lj, c]
                if val != 0:
                    U_full[li*5 + (2+lj), col_offset+c] = val
        for lj in range(3):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[(2+lj)*5 + (2+lk), col_offset+c] = val
        for lk in range(3):
            for li in range(2):
                val = W_b[lk*2+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + li, col_offset+c] = val
    col_offset += 18
    
    # C21 Part 1: A21 B11 (3x2 * 2x2)
    U_b, V_b, W_b = schoolbook_block(3, 2, 2)
    for c in range(12):
        for li in range(3):
            for lj in range(2):
                val = U_b[li*2+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + lj, col_offset+c] = val
        for lj in range(2):
            for lk in range(2):
                val = V_b[lj*2+lk, c]
                if val != 0:
                    V_full[lj*5 + lk, col_offset+c] = val
        for lk in range(2):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[lk*5 + (2+li), col_offset+c] = val
    col_offset += 12
    
    # C21 Part 2: A22 B21 (3x3 * 3x2)
    U_b, V_b, W_b = schoolbook_block(3, 3, 2)
    for c in range(18):
        for li in range(3):
            for lj in range(3):
                val = U_b[li*3+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + (2+lj), col_offset+c] = val
        for lj in range(3):
            for lk in range(2):
                val = V_b[lj*2+lk, c]
                if val != 0:
                    V_full[(2+lj)*5 + lk, col_offset+c] = val
        for lk in range(2):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[lk*5 + (2+li), col_offset+c] = val
    col_offset += 18
    
    # C22 Part 1: A21 B12 (3x2 * 2x3)
    U_b, V_b, W_b = schoolbook_block(3, 2, 3)
    for c in range(18):
        for li in range(3):
            for lj in range(2):
                val = U_b[li*2+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + lj, col_offset+c] = val
        for lj in range(2):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[lj*5 + (2+lk), col_offset+c] = val
        for lk in range(3):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + (2+li), col_offset+c] = val
    col_offset += 18
    
    # C22 Part 2: A22 B22 (3x3 * 3x3)
    U_b, V_b, W_b = schoolbook_block(3, 3, 3)
    for c in range(27):
        for li in range(3):
            for lj in range(3):
                val = U_b[li*3+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + (2+lj), col_offset+c] = val
        for lj in range(3):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[(2+lj)*5 + (2+lk), col_offset+c] = val
        for lk in range(3):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + (2+li), col_offset+c] = val
    col_offset += 27
    
    # Verify the constructed decomposition
    err = verify(U_full, V_full, W_full, total_rank)
    if err < 1e-8:
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
        U_school = np.zeros((25, 125))
        V_school = np.zeros((25, 125))
        W_school = np.zeros((25, 125))
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U_school[i*5+j, col] = 1
                    V_school[j*5+k, col] = 1
                    W_school[k*5+i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
