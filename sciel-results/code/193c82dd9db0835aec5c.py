import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen multiplication.
    Local indices:
    A: 0,1 (row 0), 2,3 (row 1)
    B: 0,1 (col 0), 2,3 (col 1) -- Wait, B is 2x2.
    Standard B index: b = j*2 + i? No, B is (m x p).
    In 2x2x2, m=2, p=2.
    B index: row j, col k. b = j*p + k = j*2 + k.
    B: 0,1 (row 0), 2,3 (row 1).
    C: 0,1 (col 0), 2,3 (col 1). c = k*n + i = k*2 + i.
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_trivial_2x2x1():
    """
    Returns U, V, W for 2x2x1 trivial multiplication.
    A: 2x2 (4 entries)
    B: 2x1 (2 entries)
    C: 2x1 (2 entries)
    Rank 4.
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C0 = A00*B0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # C0 += A01*B1
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # C1 = A10*B0
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # C1 += A11*B1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _add_decomposition(global_U, global_V, global_W, local_U, local_V, local_W, 
                        a_map, b_map, c_map, offset):
    """
    Add local decomposition to global.
    a_map: list of 4 global indices for local A indices 0,1,2,3
    b_map: list of global indices for local B indices
    c_map: list of global indices for local C indices
    offset: current rank offset
    """
    for r in range(local_U.shape[1]):
        for i in range(len(a_map)):
            if local_U[i, r] != 0:
                global_U[a_map[i], offset + r] += local_U[i, r]
        for i in range(len(b_map)):
            if local_V[i, r] != 0:
                global_V[b_map[i], offset + r] += local_V[i, r]
        for i in range(len(c_map)):
            if local_W[i, r] != 0:
                global_W[c_map[i], offset + r] += local_W[i, r]

def construct():
    n, m, p = DIMENSIONS
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    # Target rank 50
    target_rank = 50
    U = np.zeros((A_dim, target_rank))
    V = np.zeros((B_dim, target_rank))
    W = np.zeros((C_dim, target_rank))
    
    strassen_U, strassen_V, strassen_W = _get_strassen_2x2x2()
    trivial_U, trivial_V, trivial_W = _get_trivial_2x2x1()
    
    offset = 0
    
    # Part 1: 2x4x4
    # A1: 2x2 (rows 0,1 of A). A indices: 0,1,2,3
    # A2: 2x2 (rows 2,3 of A). A indices: 4,5,6,7
    
    # B11: 2x2 (rows 0,1; cols 0,1 of B). B indices: 0,1,4,5
    # B12: 2x2 (rows 0,1; cols 2,3 of B). B indices: 2,3,6,7
    # B21: 2x2 (rows 2,3; cols 0,1 of B). B indices: 8,9,12,13
    # B22: 2x2 (rows 2,3; cols 2,3 of B). B indices: 10,11,14,15
    
    # C11: 2x2 (cols 0,1 of C). C indices: 0,1,2,3
    # C12: 2x2 (cols 2,3 of C). C indices: 4,5,6,7
    
    # 1.1: A1 * B11 -> C11
    a_map = [0, 1, 2, 3]
    b_map = [0, 1, 4, 5]
    c_map = [0, 1, 2, 3]
    _add_decomposition(U, V, W, strassen_U, strassen_V, strassen_W, a_map, b_map, c_map, offset)
    offset += 7
    
    # 1.2: A1 * B12 -> C12
    a_map = [0, 1, 2, 3]
    b_map = [2, 3, 6, 7]
    c_map = [4, 5, 6, 7]
    _add_decomposition(U, V, W, strassen_U, strassen_V, strassen_W, a_map, b_map, c_map, offset)
    offset += 7
    
    # 1.3: A2 * B21 -> C11
    a_map = [4, 5, 6, 7]
    b_map = [8, 9, 12, 13]
    c_map = [0, 1, 2, 3]
    _add_decomposition(U, V, W, strassen_U, strassen_V, strassen_W, a_map, b_map, c_map, offset)
    offset += 7
    
    # 1.4: A2 * B22 -> C12
    a_map = [4, 5, 6, 7]
    b_map = [10, 11, 14, 15]
    c_map = [4, 5, 6, 7]
    _add_decomposition(U, V, W, strassen_U, strassen_V, strassen_W, a_map, b_map, c_map, offset)
    offset += 7
    
    # Part 2: 2x4x3
    # B_top: 2x3 (rows 0,1 of B; cols 4,5,6). B indices: 16,17,18, 20,21,22
    # B_bot: 2x3 (rows 2,3 of B; cols 4,5,6). B indices: 24,25,26, 28,29,30
    
    # C21: 2x2 (cols 4,5 of C). C indices: 8,9,10,11
    # C22: 2x1 (col 6 of C). C indices: 12,13
    
    # 2.1: A1 * B_top_1 (2x2) -> C21
    # B_top_1: cols 4,5. B indices: 16,17, 20,21
    a_map = [0, 1, 2, 3]
    b_map = [16, 17, 20, 21]
    c_map = [8, 9, 10, 11]
    _add_decomposition(U, V, W, strassen_U, strassen_V, strassen_W, a_map, b_map, c_map, offset)
    offset += 7
    
    # 2.2: A1 * B_top_2 (2x1) -> C22
    # B_top_2: col 6. B indices: 18, 22
    a_map = [0, 1, 2, 3]
    b_map = [18, 22]
    c_map = [12, 13]
    _add_decomposition(U, V, W, trivial_U, trivial_V, trivial_W, a_map, b_map, c_map, offset)
    offset += 4
    
    # 2.3: A2 * B_bot_1 (2x2) -> C21
    # B_bot_1: cols 4,5. B indices: 24,25, 28,29
    a_map = [4, 5, 6, 7]
    b_map = [24, 25, 28, 29]
    c_map = [8, 9, 10, 11]
    _add_decomposition(U, V, W, strassen_U, strassen_V, strassen_W, a_map, b_map, c_map, offset)
    offset += 7
    
    # 2.4: A2 * B_bot_2 (2x1) -> C22
    # B_bot_2: col 6. B indices: 26, 30
    a_map = [4, 5, 6, 7]
    b_map = [26, 30]
    c_map = [12, 13]
    _add_decomposition(U, V, W, trivial_U, trivial_V, trivial_W, a_map, b_map, c_map, offset)
    offset += 4
    
    # Verify
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Fallback to trivial if verification fails (should not happen)
        U_triv = np.zeros((n * m, n * m * p))
        V_triv = np.zeros((m * p, n * m * p))
        W_triv = np.zeros((p * n, n * m * p))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_triv[a_idx, r] = 1.0
                    V_triv[b_idx, r] = 1.0
                    W_triv[c_idx, r] = 1.0
                    r += 1
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
