import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Target rank: 33
    # Structure: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2)
    # Strassen rank 7, Standard rank 4.
    # Total: 3 * (7 + 4) = 33
    
    rank = 33
    
    U = np.zeros((num_A, rank), dtype=np.int8)
    V = np.zeros((num_B, rank), dtype=np.int8)
    W = np.zeros((num_C, rank), dtype=np.int8)
    
    # Strassen factors for 2x2x2
    # U_local[i, j], V_local[j, k], W_local[k, i]
    # Indices: i, j, k in {0, 1}
    
    # M1: (A00+A11)(B00+B11)
    S_U_1 = np.array([[1, 0], [0, 1]], dtype=np.int8)
    S_V_1 = np.array([[1, 0], [0, 1]], dtype=np.int8)
    S_W_1 = np.array([[1, 0], [0, 1]], dtype=np.int8) # C00 += M1, C11 += M1
    
    # M2: (A10+A11)B00
    S_U_2 = np.array([[0, 0], [1, 1]], dtype=np.int8)
    S_V_2 = np.array([[1, 0], [0, 0]], dtype=np.int8)
    S_W_2 = np.array([[0, 1], [0, -1]], dtype=np.int8) # C10 += M2, C11 -= M2
    
    # M3: A00(B01-B11)
    S_U_3 = np.array([[1, 0], [0, 0]], dtype=np.int8)
    S_V_3 = np.array([[0, 1], [0, -1]], dtype=np.int8)
    S_W_3 = np.array([[0, 0], [1, 1]], dtype=np.int8) # C01 += M3, C11 += M3
    
    # M4: A11(B10-B00)
    S_U_4 = np.array([[0, 0], [0, 1]], dtype=np.int8)
    S_V_4 = np.array([[-1, 0], [1, 0]], dtype=np.int8)
    S_W_4 = np.array([[1, 1], [0, 0]], dtype=np.int8) # C00 += M4, C10 += M4
    
    # M5: (A00+A01)B11
    S_U_5 = np.array([[1, 1], [0, 0]], dtype=np.int8)
    S_V_5 = np.array([[0, 0], [0, 1]], dtype=np.int8)
    S_W_5 = np.array([[-1, 0], [1, 0]], dtype=np.int8) # C00 -= M5, C01 += M5
    
    # M6: (A10-A00)(B00+B01)
    S_U_6 = np.array([[-1, 0], [1, 0]], dtype=np.int8)
    S_V_6 = np.array([[1, 1], [0, 0]], dtype=np.int8)
    S_W_6 = np.array([[0, 0], [0, 1]], dtype=np.int8) # C11 += M6
    
    # M7: (A01-A11)(B10+B11)
    S_U_7 = np.array([[0, 1], [0, -1]], dtype=np.int8)
    S_V_7 = np.array([[0, 0], [1, 1]], dtype=np.int8)
    S_W_7 = np.array([[1, 0], [0, 0]], dtype=np.int8) # C00 += M7
    
    S_U = [S_U_1, S_U_2, S_U_3, S_U_4, S_U_5, S_U_6, S_U_7]
    S_V = [S_V_1, S_V_2, S_V_3, S_V_4, S_V_5, S_V_6, S_V_7]
    S_W = [S_W_1, S_W_2, S_W_3, S_W_4, S_W_5, S_W_6, S_W_7]
    
    # Standard 2x1x2 factors
    # A is 2x1, B is 1x2, C is 2x2
    # C[i, k] = A[i, 0] * B[0, k]
    # Rank 4: one for each (i, k)
    
    Std_U = [
        np.array([[1], [0]], dtype=np.int8), # i=0
        np.array([[1], [0]], dtype=np.int8), # i=0
        np.array([[0], [1]], dtype=np.int8), # i=1
        np.array([[0], [1]], dtype=np.int8)  # i=1
    ]
    Std_V = [
        np.array([[1, 0]], dtype=np.int8), # k=0
        np.array([[0, 1]], dtype=np.int8), # k=1
        np.array([[1, 0]], dtype=np.int8), # k=0
        np.array([[0, 1]], dtype=np.int8)  # k=1
    ]
    Std_W = [
        np.array([[1, 0], [0, 0]], dtype=np.int8), # C[0,0]
        np.array([[0, 0], [1, 0]], dtype=np.int8), # C[1,0]
        np.array([[0, 1], [0, 0]], dtype=np.int8), # C[0,1]
        np.array([[0, 0], [0, 1]], dtype=np.int8)  # C[1,1]
    ]
    
    col = 0
    # Iterate over 3 blocks of the output dimension p=6
    # Each block handles 2 columns of B and 2 columns of C
    for b in range(3):
        # Strassen part: 7 terms
        # Maps local indices to global indices
        # Local A: rows 0,1; cols 0,1 (global cols 0,1)
        # Local B: rows 0,1; cols 0,1 (global rows 0,1; global cols 2b, 2b+1)
        # Local C: rows 0,1; cols 0,1 (global rows 2b, 2b+1; global cols 0,1)
        
        for r in range(7):
            # U: local (i, j) -> global (i, j)
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] += S_U[r][i][j]
            
            # V: local (j, k) -> global (j, 2b + k)
            for j in range(2):
                for k in range(2):
                    V[j * p + (2 * b + k), col] += S_V[r][j][k]
            
            # W: local (k, i) -> global (2b + k, i)
            for k in range(2):
                for i in range(2):
                    W[(2 * b + k) * n + i, col] += S_W[r][k][i]
            
            col += 1
            
        # Standard part: 4 terms
        # Maps local indices to global indices
        # Local A: rows 0,1; col 0 (global col 2)
        # Local B: row 0; cols 0,1 (global row 2; global cols 2b, 2b+1)
        # Local C: rows 0,1; cols 0,1 (global rows 2b, 2b+1; global cols 0,1)
        
        for r in range(4):
            # U: local (i, 0) -> global (i, 2)
            for i in range(2):
                U[i * m + 2, col] += Std_U[r][i][0]
            
            # V: local (0, k) -> global (2, 2b + k)
            for k in range(2):
                V[2 * p + (2 * b + k), col] += Std_V[r][0][k]
            
            # W: local (k, i) -> global (2b + k, i)
            for k in range(2):
                for i in range(2):
                    W[(2 * b + k) * n + i, col] += Std_W[r][k][i]
            
            col += 1

    # Verify the decomposition
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Valid triples: a = i*m + j, b = j*p + k, c = k*n + i
    # T should be 1 for valid triples, 0 otherwise.
    
    def verify_decomp(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check invalid triples
        # We can sample a few or check all. 6*18*12 = 1296 triples.
        # 36 valid, 1260 invalid.
        for a in range(num_A):
            for b in range(num_B):
                for c in range(num_C):
                    # Check if valid
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if (i1 == i2 and j1 == j2 and k1 == k2):
                        continue # Valid, already checked
                    
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val) > 1e-9:
                        return False
        
        return True

    if verify_decomp(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_school = np.zeros((num_A, rank_school), dtype=np.int8)
        V_school = np.zeros((num_B, rank_school), dtype=np.int8)
        W_school = np.zeros((num_C, rank_school), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_school[a, col] = 1
                    V_school[b, col] = 1
                    W_school[c, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
