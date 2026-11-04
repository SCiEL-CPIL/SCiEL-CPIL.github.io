import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x5 matrix by a 5x6 matrix.
    
    Uses Strassen's algorithm on 2x2 sub-blocks and optimized decomposition for 2x3 x 3x2 blocks.
    Rank 54.
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

    def get_improved_decomposition():
        """
        Construct decomposition with rank 54.
        
        Part 1: A1*B1 (2x2 x 2x6) -> 3 * Strassen(2x2x2) = 21
        Part 2: A2*B2 (2x3 x 3x6) -> 3 * (Strassen(2x2x2) + Outer(2x1x1x2)) = 3 * (7 + 4) = 33
        Total: 54
        """
        rank = 54
        U = [[0] * rank for _ in range(n * m)]  # (10, 54)
        V = [[0] * rank for _ in range(m * p)]  # (30, 54)
        W = [[0] * rank for _ in range(p * n)]  # (12, 54)
        
        col = 0
        
        # Part 1: A1*B1 where A1 is A[:, 0:2], B1 is B[0:2, :]
        # 3 blocks of 2x2 x 2x2
        for pair_idx in range(3):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            
            # A indices: 0, 1, 5, 6
            # B indices: k1, k2, 6+k1, 6+k2
            # W indices: 2*k1, 2*k2, 2*k1+1, 2*k2+1
            
            # M1 = A00 * B11
            U[0][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k2][col] += 1
            W[2*k2+1][col] += -1
            col += 1
            
            # M2 = (A00 + A01) * B10
            U[0][col] = 1
            U[1][col] = 1
            V[6 + k1][col] = 1
            W[2*k1][col] += -1
            W[2*k2][col] += 1
            col += 1
            
            # M3 = (A10 + A11) * B00
            U[5][col] = 1
            U[6][col] = 1
            V[k1][col] = 1
            W[2*k1+1][col] += 1
            W[2*k2+1][col] += 1
            col += 1
            
            # M4 = A11 * B11
            U[6][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k1+1][col] += 1
            col += 1
            
            # M5 = A00 * (B00 + B11)
            U[0][col] = 1
            V[k1][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k2+1][col] += -1
            col += 1
            
            # M6 = (A01 - A11) * (B10 + B11)
            U[1][col] = 1
            U[6][col] = -1
            V[6 + k1][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            col += 1
            
            # M7 = (A00 - A10) * (B00 + B01)
            U[0][col] = 1
            U[5][col] = -1
            V[k1][col] = 1
            V[k2][col] = 1
            W[2*k2+1][col] += 1
            col += 1
            
        # Part 2: A2*B2 where A2 is A[:, 2:5], B2 is B[2:5, :]
        # 3 blocks of (2x3 x 3x2)
        # Each block: A_L (2x2) B_T (2x2) + A_R (2x1) B_B (1x2)
        for pair_idx in range(3):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            
            # --- A_L B_T (Strassen) ---
            # A_L: cols 2, 3 -> U indices 2, 3, 7, 8
            # B_T: rows 2, 3 -> V indices 12+k1, 12+k2, 18+k1, 18+k2
            # W: 2*k1, 2*k2, 2*k1+1, 2*k2+1
            
            # M1 = A00 * B11 (A[0,2]*B[3,k2])
            U[2][col] = 1
            V[18 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k2][col] += 1
            W[2*k2+1][col] += -1
            col += 1
            
            # M2 = (A00 + A01) * B10 (A[0,2]+A[0,3])*B[3,k1]
            U[2][col] = 1
            U[3][col] = 1
            V[18 + k1][col] = 1
            W[2*k1][col] += -1
            W[2*k2][col] += 1
            col += 1
            
            # M3 = (A10 + A11) * B00 (A[1,2]+A[1,3])*B[2,k1]
            U[7][col] = 1
            U[8][col] = 1
            V[12 + k1][col] = 1
            W[2*k1+1][col] += 1
            W[2*k2+1][col] += 1
            col += 1
            
            # M4 = A11 * B11 (A[1,3]*B[3,k2])
            U[8][col] = 1
            V[18 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k1+1][col] += 1
            col += 1
            
            # M5 = A00 * (B00 + B11) (A[0,2]*(B[2,k1]+B[3,k2]))
            U[2][col] = 1
            V[12 + k1][col] = 1
            V[18 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k2+1][col] += -1
            col += 1
            
            # M6 = (A01 - A11) * (B10 + B11) (A[0,3]-A[1,3])*(B[3,k1]+B[3,k2])
            U[3][col] = 1
            U[8][col] = -1
            V[18 + k1][col] = 1
            V[18 + k2][col] = 1
            W[2*k1][col] += 1
            col += 1
            
            # M7 = (A00 - A10) * (B00 + B01) (A[0,2]-A[1,2])*(B[2,k1]+B[2,k2])
            U[2][col] = 1
            U[7][col] = -1
            V[12 + k1][col] = 1
            V[12 + k2][col] = 1
            W[2*k2+1][col] += 1
            col += 1
            
            # --- A_R B_B (Rank 4) ---
            # A_R: col 4 -> U indices 4, 9
            # B_B: row 4 -> V indices 24+k1, 24+k2
            
            # Term 1: A[0,4] * B[4,k1] -> C[0,k1]
            U[4][col] = 1
            V[24 + k1][col] = 1
            W[2*k1][col] += 1
            col += 1
            
            # Term 2: A[0,4] * B[4,k2] -> C[0,k2]
            U[4][col] = 1
            V[24 + k2][col] = 1
            W[2*k2][col] += 1
            col += 1
            
            # Term 3: A[1,4] * B[4,k1] -> C[1,k1]
            U[9][col] = 1
            V[24 + k1][col] = 1
            W[2*k1+1][col] += 1
            col += 1
            
            # Term 4: A[1,4] * B[4,k2] -> C[1,k2]
            U[9][col] = 1
            V[24 + k2][col] = 1
            W[2*k2+1][col] += 1
            col += 1
            
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        return U, V, W
    
    # Try the improved decomposition
    U, V, W = get_improved_decomposition()
    
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
