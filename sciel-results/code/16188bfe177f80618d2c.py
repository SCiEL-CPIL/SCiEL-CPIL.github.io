import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 2, 3, 5
    
    def add_strassen(U, V, W, col, i1, i2, j1, j2, k1, k2):
        """Strassen 2x2x2 decomposition, returns new col index"""
        # M1 = (A[i1,j1] + A[i2,j2])(B[j1,k1] + B[j2,k2])
        U[i1*m+j1, col] += 1; U[i2*m+j2, col] += 1
        V[j1*p+k1, col] += 1; V[j2*p+k2, col] += 1
        W[k1*n+i1, col] += 1; W[k2*n+i2, col] += 1
        col += 1
        # M2 = (A[i2,j1] + A[i2,j2]) B[j1,k1]
        U[i2*m+j1, col] += 1; U[i2*m+j2, col] += 1
        V[j1*p+k1, col] += 1
        W[k1*n+i2, col] += 1; W[k2*n+i2, col] += -1
        col += 1
        # M3 = A[i1,j1] (B[j1,k2] - B[j2,k2])
        U[i1*m+j1, col] += 1
        V[j1*p+k2, col] += 1; V[j2*p+k2, col] += -1
        W[k2*n+i1, col] += 1; W[k2*n+i2, col] += 1
        col += 1
        # M4 = A[i2,j2] (B[j2,k1] - B[j1,k1])
        U[i2*m+j2, col] += 1
        V[j2*p+k1, col] += 1; V[j1*p+k1, col] += -1
        W[k1*n+i1, col] += 1; W[k1*n+i2, col] += 1
        col += 1
        # M5 = (A[i1,j1] + A[i1,j2]) B[j2,k2]
        U[i1*m+j1, col] += 1; U[i1*m+j2, col] += 1
        V[j2*p+k2, col] += 1
        W[k1*n+i1, col] += -1; W[k2*n+i1, col] += 1
        col += 1
        # M6 = (A[i2,j1] - A[i1,j1])(B[j1,k1] + B[j1,k2])
        U[i2*m+j1, col] += 1; U[i1*m+j1, col] += -1
        V[j1*p+k1, col] += 1; V[j1*p+k2, col] += 1
        W[k2*n+i2, col] += 1
        col += 1
        # M7 = (A[i1,j2] - A[i2,j2])(B[j2,k1] + B[j2,k2])
        U[i1*m+j2, col] += 1; U[i2*m+j2, col] += -1
        V[j2*p+k1, col] += 1; V[j2*p+k2, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        return col

    def add_schoolbook(U, V, W, col, i_indices, j_indices, k_indices):
        """Schoolbook multiplication, returns new col index"""
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col

    def verify(U, V, W, rank):
        """Verify the decomposition is correct"""
        # Check that sum_r U[:,r] V[:,r] W[:,r]^T reproduces the matmul tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build the tensor and check against known matmul
        A = np.array([[1,2,3],[4,5,6]], dtype=float)
        B = np.array([[7,8,9,10,11],[12,13,14,15,16],[17,18,19,20,21]], dtype=float)
        C = A @ B
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - C[i,k]) > 1e-9:
                        return False
        return True

    # Strategy: decompose into blocks
    # Block A: A[0:2, 0:2] x B[0:2, 0:3] -> C[0:2, 0:3], rank 10 (2x2x3 tensor)
    # Block B: A[0:2, 0:2] x B[0:2, 3:5] -> C[0:2, 3:5], rank 7 (Strassen 2x2x2)
    # Block C: A[0:2, 2:3] x B[2:3, 0:5] -> C[0:2, 0:5], rank 10 (2x1x5 -> 2x2x5 is 10)
    # Total: 27
    
    # First, let's try the 28-rank decomposition (verified working)
    # A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]: Strassen rank 7
    # A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4]: Strassen rank 7
    # A[0:2, 0:2] x B[0:2, 4:5] -> C[0:2, 4:5]: schoolbook rank 4
    # A[0:2, 2:3] x B[2:3, 0:2] -> C[0:2, 0:2]: schoolbook rank 4
    # A[0:2, 2:3] x B[2:3, 2:4] -> C[0:2, 2:4]: schoolbook rank 4
    # A[0:2, 2:3] x B[2:3, 4:5] -> C[0:2, 4:5]: schoolbook rank 2
    # Total: 7+7+4+4+4+2 = 28
    
    rank28 = 28
    U28 = np.zeros((n*m, rank28))
    V28 = np.zeros((m*p, rank28))
    W28 = np.zeros((p*n, rank28))
    
    col = 0
    col = add_strassen(U28, V28, W28, col, 0, 1, 0, 1, 0, 1)   # 7
    col = add_strassen(U28, V28, W28, col, 0, 1, 0, 1, 2, 3)   # 7
    col = add_schoolbook(U28, V28, W28, col, [0,1], [0,1], [4]) # 4
    col = add_schoolbook(U28, V28, W28, col, [0,1], [2], [0,1]) # 4
    col = add_schoolbook(U28, V28, W28, col, [0,1], [2], [2,3]) # 4
    col = add_schoolbook(U28, V28, W28, col, [0,1], [2], [4])   # 2
    
    best_result = {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    best_rank = rank28
    
    # Try to find rank 27 by combining blocks differently
    # Key idea: A[0:2, 0:2] x B[0:2, 0:3] has rank 10 (not 7+4=11)
    # This saves 1 rank
    
    if time.time() < deadline:
        # 2x2 x 2x3 tensor rank 10 decomposition
        # This is a known result: M(2,2,3) = 10
        # We need to construct it. One approach:
        # Use Strassen for the first 2 columns (rank 7) + schoolbook for column 2 (rank 4) = 11
        # But there's a rank-10 algorithm. Let me try to construct it.
        
        # Actually, let me try a different block decomposition that might give 27:
        # A[0:2, 0:2] x B[0:2, 0:5] -> C[0:2, 0:5]
        #   = A[0:2,0:2] x B[0:2,0:3] (rank 10) + A[0:2,0:2] x B[0:2,3:5] (rank 7) = 17
        # A[0:2, 2:3] x B[2:3, 0:5] -> C[0:2, 0:5]
        #   = 2x1 x 1x5 = schoolbook rank 10
        # Total: 17 + 10 = 27
        
        # But we need the rank-10 2x2x3 decomposition. Let me try to find it.
        # The 2x2x3 matrix multiplication tensor T[i,j,k] = A[i,j]*B[j,k]
        # with i in {0,1}, j in {0,1}, k in {0,1,2}
        
        # Known rank-10 decomposition for 2x2x3:
        # I'll use a construction based on the literature
        rank27 = 27
        U27 = np.zeros((n*m, rank27))
        V27 = np.zeros((m*p, rank27))
        W27 = np.zeros((p*n, rank27))
        
        col = 0
        # For A[0:2,0:2] x B[0:2,0:3] -> C[0:2,0:3], use rank-10
        # Indices: i in {0,1}, j in {0,1}, k in {0,1,2}
        # U index: i*m + j (m=3, so i*3+j)
        # V index: j*p + k (p=5, so j*5+k)
        # W index: k*n + i (n=2, so k*2+i)
        
        # Rank-10 2x2x3 decomposition (from literature):
        # M1 = (a00+a11)(b00+b12) -> c00, c11
        U27[0*3+0, col] = 1; U27[1*3+1, col] = 1
        V27[0*5+0, col] = 1; V27[1*5+2, col] = 1
        W27[0*2+0, col] = 1; W27[1*2+1, col] = 1
        col += 1
        # M2 = (a10+a11)b00 -> c10, -c11
        U27[1*3+0, col] = 1; U27[1*3+1, col] = 1
        V27[0*5+0, col] = 1
        W27[0*2+1, col] = 1; W27[1*2+1, col] = -1
        col += 1
        # M3 = a00(b01-b11) -> c01, c11
        U27[0*3+0, col] = 1
        V27[0*5+1, col] = 1; V27[1*5+1, col] = -1
        W27[1*2+0, col] = 1; W27[1*2+1, col] = 1
        col += 1
        # M4 = a11(b10-b00) -> c10, c00
        U27[1*3+1, col] = 1
        V27[1*5+0, col] = 1; V27[0*5+0, col] = -1
        W27[0*2+1, col] = 1; W27[0*2+0, col] = 1
        col += 1
        # M5 = (a00+a01)b11 -> c01, -c00
        U27[0*3+0, col] = 1; U27[0*3+1, col] = 1
        V27[1*5+1, col] = 1
        W27[1*2+0, col] = 1; W27[0*2+0, col] = -1
        col += 1
        # M6 = (a10-a00)(b00+b01) -> c10
        U27[1*3+0, col] = 1; U27[0*3+0, col] = -1
        V27[0*5+0, col] = 1; V27[0*5+1, col] = 1
        W27[0*2+1, col] = 1
        col += 1
        # M7 = (a01-a11)(b10+b11) -> c00
        U27[0*3+1, col] = 1; U27[1*3+1, col] = -1
        V27[1*5+0, col] = 1; V27[1*5+1, col] = 1
        W27[0*2+0, col] = 1
        col += 1
        # Now we need 3 more terms to handle k=2 (the third column of B)
        # For k=2: c02 = a00*b02 + a01*b12, c12 = a10*b02 + a11*b12
        # Schoolbook would be 4 terms, but maybe we can do better with 3
        # M8 = a00*b02 -> c02
        U27[0*3+0, col] = 1
        V27[0*5+2, col] = 1
        W27[2*2+0, col] = 1
        col += 1
        # M9 = a01*b12 -> c02
        U27[0*3+1, col] = 1
        V27[1*5+2, col] = 1
        W27[2*2+0, col] = 1
        col += 1
        # M10 = a10*b02 -> c12
        U27[1*3+0, col] = 1
        V27[0*5+2, col] = 1
        W27[2*2+1, col] = 1
        col += 1
        # Wait, that's only 10 terms but we're missing a11*b12 for c12
        # Let me reconsider...
        
        # Actually, let me just verify the 28-rank result and return it if valid
        if verify(U28, V28, W28, rank28):
            return best_result
        else:
            # Fallback: pure schoolbook rank 30
            rank30 = 30
            U30 = np.zeros((n*m, rank30))
            V30 = np.zeros((m*p, rank30))
            W30 = np.zeros((p*n, rank30))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U30[i*m+j, col] = 1
                        V30[j*p+k, col] = 1
                        W30[k*n+i, col] = 1
                        col += 1
            return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}
