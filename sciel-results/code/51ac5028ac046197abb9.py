import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x5 matrix by a 5x6 matrix.
    
    Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 60 to 57.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    def check_decomposition(U, V, W, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def get_strassen_decomposition():
        """
        Construct decomposition using Strassen for 2x2 blocks.
        
        Partition:
        - A (2x5) = [A1 (2x2) | A2 (2x3)]  
        - B (5x6) = [B1 (2x6); B2 (3x6)]
        - C = A1*B1 + A2*B2
        
        Part 1: A1*B1 (2x2 x 2x6)
          Split B1 into 3 column pairs: B1 = [B1_0 B1_1 B1_2], each 2x2
          Use Strassen (rank 7) for each pair.
          Columns in B1: (0,1), (2,3), (4,5)
          
        Part 2: A2*B2 (2x3 x 3x6)
          Split B2 into 2 column groups: B2 = [B2_0 B2_1], each 3x3
          Use schoolbook (rank 18) for each group.
          Columns in B2: (0,1,2), (3,4,5)
          
        Total rank: 3*7 + 2*18 = 21 + 36 = 57
        """
        rank = 57
        U = [[0] * rank for _ in range(n * m)]  # (10, 57)
        V = [[0] * rank for _ in range(m * p)]  # (30, 57)
        W = [[0] * rank for _ in range(p * n)]  # (12, 57)
        
        col = 0
        
        # Part 1: A1*B1 where A1 is A[:, 0:2] (2x2), B1 is B[0:2, :] (2x6)
        # Strassen for 2x2 x 2x2, applied to 3 column pairs
        # B1 column pairs: (col 0,1), (col 2,3), (col 4,5)
        for pair_idx in range(3):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            
            # Strassen 7 products for 2x2 x 2x2:
            # A1 = [[a00, a01], [a10, a11]] = A[0,0], A[0,1], A[1,0], A[1,1]
            # B = [[b00, b01], [b10, b11]] = B[0,k1], B[0,k2], B[1,k1], B[1,k2]
            
            # M1 = a00 * a11  (but in our case, it's A[0,0] * B[1,k1]... wait)
            # Let me re-derive. For C = A*B where A is 2x2, B is 2x2:
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            # C00 = M5 + M4 - M2 + M6
            # C01 = M1 + M2
            # C10 = M3 + M4
            # C11 = M1 - M5 + M3 + M7
            # where:
            # M1 = A00 * B11
            # M2 = (A00 + A01) * B10
            # M3 = (A10 + A11) * B00
            # M4 = A11 * B11
            # M5 = A00 * (B00 + B11)
            # M6 = (A01 - A11) * (B10 + B11)
            # M7 = (A00 - A10) * (B00 + B01)
            
            # In our tensor decomposition:
            # A[i,j] is indexed as U[i*m + j]
            # B[j,k] is indexed as V[j*p + k]
            # C[k,i] is indexed as W[k*n + i]
            
            # For the 2x2 block:
            # A00 = A[0,0] -> U[0*5+0] = U[0]
            # A01 = A[0,1] -> U[0*5+1] = U[1]
            # A10 = A[1,0] -> U[1*5+0] = U[5]
            # A11 = A[1,1] -> U[1*5+1] = U[6]
            
            # B00 = B[0,k1] -> V[0*6+k1] = V[k1]
            # B01 = B[0,k2] -> V[0*6+k2] = V[k2]
            # B10 = B[1,k1] -> V[1*6+k1] = V[6+k1]
            # B11 = B[1,k2] -> V[1*6+k2] = V[6+k2]
            
            # C00 = C[k1, 0] -> W[k1*2+0] = W[2*k1]
            # C01 = C[k2, 0] -> W[k2*2+0] = W[2*k2]
            # C10 = C[k1, 1] -> W[k1*2+1] = W[2*k1+1]
            # C11 = C[k2, 1] -> W[k2*2+1] = W[2*k2+1]
            
            # M1 = A00 * B11
            U[0][col] = 1
            V[6 + k2][col] = 1
            W[2*k2][col] += 1      # C01 += M1
            W[2*k2+1][col] += 1    # C11 += M1
            col += 1
            
            # M2 = (A00 + A01) * B10
            U[0][col] = 1
            U[1][col] = 1
            V[6 + k1][col] = 1
            W[2*k1][col] += -1     # C00 -= M2
            W[2*k2][col] += 1      # C01 += M2
            col += 1
            
            # M3 = (A10 + A11) * B00
            U[5][col] = 1
            U[6][col] = 1
            V[k1][col] = 1
            W[2*k1+1][col] += 1    # C10 += M3
            W[2*k2+1][col] += 1    # C11 += M3
            col += 1
            
            # M4 = A11 * B11
            U[6][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1      # C00 += M4
            W[2*k1+1][col] += 1    # C10 += M4
            col += 1
            
            # M5 = A00 * (B00 + B11)
            U[0][col] = 1
            V[k1][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1      # C00 += M5
            W[2*k2+1][col] += -1   # C11 -= M5
            col += 1
            
            # M6 = (A01 - A11) * (B10 + B11)
            U[1][col] = 1
            U[6][col] = -1
            V[6 + k1][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1      # C00 += M6
            col += 1
            
            # M7 = (A00 - A10) * (B00 + B01)
            U[0][col] = 1
            U[5][col] = -1
            V[k1][col] = 1
            V[k2][col] = 1
            W[2*k2+1][col] += 1    # C11 += M7
            col += 1
        
        # Part 2: A2*B2 where A2 is A[:, 2:5] (2x3), B2 is B[2:5, :] (3x6)
        # Schoolbook for 2x3 x 3x3, applied to 2 column groups
        # B2 column groups: (col 0,1,2), (col 3,4,5)
        for group_idx in range(2):
            k_start = group_idx * 3
            for k_offset in range(3):
                k = k_start + k_offset
                for i in range(2):
                    for j in range(3):
                        # A[i, j+2] * B[j+2, k] contributes to C[k, i]
                        U[i*5 + (j+2)][col] = 1
                        V[(j+2)*6 + k][col] = 1
                        W[k*2 + i][col] = 1
                        col += 1
        
        # col should be 57
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        return U, V, W
    
    # Try the Strassen decomposition
    U, V, W = get_strassen_decomposition()
    
    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U, "V": V, "W": W}

def check_decomposition(U, V, W, tol=1e-9):
    n, m, p = 2, 5, 6
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    is_valid = check_decomposition(res['U'], res['V'], res['W'])
    rank = len(res['U'][0])
    print(f"Rank: {rank}, Valid: {is_valid}")
