import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Correct Strassen decomposition for 2x2 * 2x2.
    A indices: 0,1,2,3 (row-major)
    B indices: 0,1,2,3 (row-major)
    C indices: 0,1,2,3 (row-major, C[k,i] -> k*2+i)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1 # C10 += M2, C11 -= M2
    
    # M3 = A00(B01+B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = 1
    W[1][2] = 1; W[3][2] = 1 # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1 # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1 # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = 1 # C00 += M6, C01 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1 # C00 -= M7, C11 += M7
    
    return U, V, W

def _strassen_2x2x3_block():
    """
    Decompose 2x2 * 2x3.
    Rank 11.
    """
    rank = 11
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Map Strassen 2x2 B to first 2 cols of 2x3 B
    # Local B: 0,1 (col 0), 2,3 (col 1), 4,5 (col 2)
    # Strassen B: 0,1 (row 0), 2,3 (row 1)
    # We want Strassen B to correspond to local B cols 0 and 1.
    # Strassen B[0] -> Local B[0]
    # Strassen B[1] -> Local B[1]
    # Strassen B[2] -> Local B[2]
    # Strassen B[3] -> Local B[3]
    b_map = [0, 1, 2, 3]
    
    # Strassen C: 0,1,2,3 -> Local C: 0,1,2,3 (rows 0,1 of 3x2 C)
    c_map = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[s][r] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map[s]][r] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map[s]][r] += W_s[s][r]
    
    # Schoolbook for col 2 (local B indices 4, 5)
    col = 7
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*3+2][col] = 1 # B[j, 2] -> index j*3+2
            W[2*2+i][col] = 1 # C[2, i] -> index 2*2+i = 4+i
            col += 1
            
    return U, V, W

def _strassen_2x3x5():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j in {0,1}, k in {0,1} -> Strassen 2x2x2
    U_s, V_s, W_s = _strassen_2x2x2()
    a_map = [0, 1, 3, 4]
    b_map = [0, 1, 5, 6]
    c_map = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0: U[a_map[s]][col] += U_s[s][r]
            if V_s[s][r] != 0: V[b_map[s]][col] += V_s[s][r]
            if W_s[s][r] != 0: W[c_map[s]][col] += W_s[s][r]
        col += 1
        
    # Block 2: j in {0,1}, k in {2,3,4} -> Strassen 2x2x3
    U_b, V_b, W_b = _strassen_2x2x3_block()
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 4, 7, 8, 9]
    c_map2 = [4, 5, 6, 7, 8, 9]
    
    for r in range(11):
        for s in range(4):
            if U_b[s][r] != 0: U[a_map2[s]][col + r] += U_b[s][r]
        for s in range(6):
            if V_b[s][r] != 0: V[b_map2[s]][col + r] += V_b[s][r]
        for s in range(6):
            if W_b[s][r] != 0: W[c_map2[s]][col + r] += W_b[s][r]
    col += 11
    
    # Block 3: j=2, k in {0..4} -> Schoolbook
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank 28
    try:
        result = _strassen_2x3x5()
        T = _build_target_tensor(n, m, p)
        valid, diff = _verify_decomposition(result["U"], result["V"], result["W"], T)
        if valid:
            return result
    except Exception:
        pass
        
    if time.time() > deadline:
        return _schoolbook_decomposition(n, m, p)
        
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
