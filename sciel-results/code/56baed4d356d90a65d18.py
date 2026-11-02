"""Matrix multiplication tensor decomposition for 4x5 * 5x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Schoolbook decomposition: rank = n*m*p = 100 (guaranteed valid)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def build_tensor(n, m, p):
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    def verify_decomposition(U, V, W, T, tol=1e-6):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        error = np.max(np.abs(T_rec - T))
        return error < tol, error

    # Constructive decomposition for 4x5x5 using Strassen for 4x4x4 block
    # Rank 85
    
    # 1. Strassen 2x2x2 factors
    # U (4x7), V (4x7), W (4x7)
    # Indices: A(i,j) -> i*2+j, B(j,k) -> j*2+k, C(k,i) -> k*2+i
    U_222 = [
        [1, 0, 1, 0, 1, 0, 0], # A11
        [0, 0, 0, 0, 0, -1, 1], # A12
        [0, 1, 0, 1, 0, 1, 0], # A21
        [1, 0, 0, 1, 0, 0, -1]  # A22
    ]
    V_222 = [
        [1, 1, 1, 0, 0, 1, 1], # B11
        [0, 0, -1, -1, 0, 0, 1], # B12
        [0, 0, 0, 1, 1, 1, 0], # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W_222 = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # Verify 2x2x2 Strassen
    U_222_arr = np.array(U_222)
    V_222_arr = np.array(V_222)
    W_222_arr = np.array(W_222)
    T_222_rec = np.einsum('ar,br,cr->abc', U_222_arr, V_222_arr, W_222_arr)
    T_222_true = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_222_true[i*2+j, j*2+k, k*2+i] = 1.0
    if not np.allclose(T_222_rec, T_222_true):
        # Fallback to schoolbook if Strassen factors are wrong
        pass

    # 2. Kronecker product for 4x4x4
    # U_444 = kron(U_222, U_222) ? 
    # The tensor T_444 is the Kronecker product of T_222 with T_222.
    # The factors for T_444 are the Kronecker products of the factors of T_222.
    # U_444 (16x49) = kron(U_222, U_222)
    # V_444 (16x49) = kron(V_222, V_222)
    # W_444 (16x49) = kron(W_222, W_222)
    
    U_444 = np.kron(U_222_arr, U_222_arr)
    V_444 = np.kron(V_222_arr, V_222_arr)
    W_444 = np.kron(W_222_arr, W_222_arr)
    
    # Verify 4x4x4
    T_444_rec = np.einsum('ar,br,cr->abc', U_444, V_444, W_444)
    T_444_true = np.zeros((16,16,16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T_444_true[i*4+j, j*4+k, k*4+i] = 1.0
    if not np.allclose(T_444_rec, T_444_true):
        # Fallback
        pass

    # 3. Schoolbook factors for remaining blocks
    # Block 2: 4x4x1 (A: 4x4, B: 4x1, C: 1x4)
    # Rank 16
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        U_441[i*4+j, idx] = 1
        V_441[j, idx] = 1
        W_441[i, idx] = 1
        
    # Block 3: 4x1x4 (A: 4x1, B: 1x4, C: 4x4)
    # Rank 16
    U_414 = np.zeros((4, 16))
    V_414 = np.zeros((4, 16))
    W_414 = np.zeros((16, 16))
    for idx in range(16):
        i = idx // 4
        k = idx % 4
        U_414[i, idx] = 1
        V_414[k, idx] = 1
        W_414[k*4+i, idx] = 1
        
    # Block 4: 4x1x1 (A: 4x1, B: 1x1, C: 1x4)
    # Rank 4
    U_411 = np.zeros((4, 4))
    V_411 = np.zeros((1, 4))
    W_411 = np.zeros((4, 4))
    for idx in range(4):
        i = idx
        U_411[i, idx] = 1
        V_411[0, idx] = 1
        W_411[i, idx] = 1

    # Assemble global factors
    # Global U: 20 x 85
    # Global V: 25 x 85
    # Global W: 20 x 85
    
    U_global = np.zeros((20, 85))
    V_global = np.zeros((25, 85))
    W_global = np.zeros((20, 85))
    
    # Block 1: 4x4x4 (Strassen)
    # U_444 (16x49) maps local A(i,j) i<4,j<4 to global A(i,j) i<4,j<4
    # Global A index: i*5 + j
    for local_i in range(16):
        i = local_i // 4
        j = local_i % 4
        global_i = i * 5 + j
        U_global[global_i, 0:49] = U_444[local_i, :]
        
    # V_444 (16x49) maps local B(j,k) j<4,k<4 to global B(j,k) j<4,k<4
    # Global B index: j*5 + k
    for local_j in range(16):
        j = local_j // 4
        k = local_j % 4
        global_j = j * 5 + k
        V_global[global_j, 0:49] = V_444[local_j, :]
        
    # W_444 (16x49) maps local C(k,i) k<4,i<4 to global C(k,i) k<4,i<4
    # Global C index: k*4 + i
    for local_k in range(16):
        k = local_k // 4
        i = local_k % 4
        global_k = k * 4 + i
        W_global[global_k, 0:49] = W_444[local_k, :]
        
    # Block 2: 4x4x1 (Schoolbook)
    # U_441 (16x16) maps local A(i,j) i<4,j<4 to global A(i,j) i<4,j<4
    for local_i in range(16):
        i = local_i // 4
        j = local_i % 4
        global_i = i * 5 + j
        U_global[global_i, 49:65] = U_441[local_i, :]
        
    # V_441 (4x16) maps local B(j,k) j<4,k=4 to global B(j,4) j<4
    # Global B index: j*5 + 4
    for local_j in range(4):
        j = local_j
        global_j = j * 5 + 4
        V_global[global_j, 49:65] = V_441[local_j, :]
        
    # W_441 (4x16) maps local C(k,i) k=4,i<4 to global C(4,i) i<4
    # Global C index: 4*4 + i = 16 + i
    for local_i in range(4):
        i = local_i
        global_i = 16 + i
        W_global[global_i, 49:65] = W_441[local_i, :]
        
    # Block 3: 4x1x4 (Schoolbook)
    # U_414 (4x16) maps local A(i,j) i<4,j=4 to global A(i,4) i<4
    # Global A index: i*5 + 4
    for local_i in range(4):
        i = local_i
        global_i = i * 5 + 4
        U_global[global_i, 65:81] = U_414[local_i, :]
        
    # V_414 (4x16) maps local B(j,k) j=4,k<4 to global B(4,k) k<4
    # Global B index: 4*5 + k = 20 + k
    for local_k in range(4):
        k = local_k
        global_k = 20 + k
        V_global[global_k, 65:81] = V_414[local_k, :]
        
    # W_414 (16x16) maps local C(k,i) k<4,i<4 to global C(k,i) k<4,i<4
    for local_k in range(16):
        k = local_k // 4
        i = local_k % 4
        global_k = k * 4 + i
        W_global[global_k, 65:81] = W_414[local_k, :]
        
    # Block 4: 4x1x1 (Schoolbook)
    # U_411 (4x4) maps local A(i,j) i<4,j=4 to global A(i,4) i<4
    for local_i in range(4):
        i = local_i
        global_i = i * 5 + 4
        U_global[global_i, 81:85] = U_411[local_i, :]
        
    # V_411 (1x4) maps local B(j,k) j=4,k=4 to global B(4,4)
    # Global B index: 4*5 + 4 = 24
    V_global[24, 81:85] = V_411[0, :]
        
    # W_411 (4x4) maps local C(k,i) k=4,i<4 to global C(4,i) i<4
    for local_i in range(4):
        i = local_i
        global_i = 16 + i
        W_global[global_i, 81:85] = W_411[local_i, :]

    # Verify
    T = build_tensor(n, m, p)
    valid, err = verify_decomposition(U_global, V_global, W_global, T)
    
    if valid:
        # Convert to lists of fractions
        U_frac = [[Fraction(int(U_global[i, r] * 2)) / 2 for r in range(85)] for i in range(20)]
        V_frac = [[Fraction(int(V_global[i, r] * 2)) / 2 for r in range(85)] for i in range(25)]
        W_frac = [[Fraction(int(W_global[i, r] * 2)) / 2 for r in range(85)] for i in range(20)]
        
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    else:
        return best_result

# EVOLVE-BLOCK-END
