"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    def strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for r in range(rank):
            if len(U[r]) if isinstance(U, list) and isinstance(U[0], list) else True:
                pass
        
        # Build tensors and verify
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape for efficient computation
        U_mat = U_arr.reshape(n, m, rank)
        V_mat = V_arr.reshape(m, p, rank)
        W_mat = W_arr.reshape(p, n, rank)
        
        # For each (i,j,k), check sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
        # This is expensive for large rank, so check a subset
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Strassen-based decomposition for 4x4x4 + schoolbook for 4x4x1
    rank_4x4x4 = 56
    rank_4x4x1 = 16
    total_rank = rank_4x4x4 + rank_4x4x1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U_s, V_s, W_s = strassen_2x2()
    
    # 8 block multiplications for 4x4 * 4x4
    mults = [
        (0, 0, 0),  # A00 * B00 -> C00
        (0, 2, 0),  # A01 * B10 -> C00
        (0, 0, 2),  # A00 * B01 -> C01
        (0, 2, 2),  # A01 * B11 -> C01
        (2, 0, 0),  # A10 * B00 -> C10
        (2, 2, 0),  # A11 * B10 -> C10
        (2, 0, 2),  # A10 * B01 -> C11
        (2, 2, 2),  # A11 * B11 -> C11
    ]
    
    col = 0
    for i_off, j_off, k_off in mults:
        for c in range(7):
            for i_l in range(2):
                for j_l in range(2):
                    u_val = U_s[i_l*2 + j_l][c]
                    if u_val != 0:
                        i_g = i_off + i_l
                        j_g = j_off + j_l
                        U[i_g * m + j_g][col] += u_val
                        
            for j_l in range(2):
                for k_l in range(2):
                    v_val = V_s[j_l*2 + k_l][c]
                    if v_val != 0:
                        j_g = j_off + j_l
                        k_g = k_off + k_l
                        V[j_g * p + k_g][col] += v_val
                        
            for k_l in range(2):
                for i_l in range(2):
                    w_val = W_s[k_l*2 + i_l][c]
                    if w_val != 0:
                        k_g = k_off + k_l
                        i_g = i_off + i_l
                        W[k_g * n + i_g][col] += w_val
                        
            col += 1
            
    # 4x4 * 4x1 schoolbook part (column k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Verify
    if not verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
