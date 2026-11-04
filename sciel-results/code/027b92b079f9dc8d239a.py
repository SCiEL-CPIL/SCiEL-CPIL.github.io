import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _strassen_2x2():
    # Returns U, V, W for 2x2 matrix multiplication with rank 7
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
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    # Returns U, V, W for 4x4 matrix multiplication with rank 49
    # Built from Strassen 2x2
    U2, V2, W2 = _strassen_2x2()
    
    # Block coefficients for 4x4 Strassen (same as 2x2 but on blocks)
    # Blocks: 00, 01, 10, 11
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    U_block = np.zeros((4, 7))
    V_block = np.zeros((4, 7))
    W_block = np.zeros((4, 7))
    
    # M1
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    # M2
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1
    
    # M3
    U_block[0, 2] = 1
    V_block[2, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1
    
    # M4
    U_block[3, 3] = 1
    V_block[1, 3] = 1; V_block[0, 3] = -1
    W_block[3, 3] = 1
    
    # M5
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[1, 4] = 1
    
    # M6
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[2, 5] = 1
    
    # M7
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1
    
    # Reconstruct 4x4 U, V, W
    # Indices: i, j in 0..3
    # Block index b = (i//2)*2 + (j//2)
    # Local index l = (i%2)*2 + (j%2)
    # Term index t = k*7 + s
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            idx_A = i * 4 + j
            b_A = (i // 2) * 2 + (j // 2)
            l_A = (i % 2) * 2 + (j % 2)
            
            for k in range(7):
                for s in range(7):
                    t = k * 7 + s
                    U4[idx_A, t] = U_block[b_A, k] * U2[l_A, s]
                    
                    idx_B = j * 4 + k # Wait, B is 4x4. B[j, k]
                    # In 4x4 matmul, B indices are (j, k) where j is row, k is col.
                    # But in our tensor, V is (m*p, rank). For 4x4, m=4, p=4.
                    # V index is j*4 + k.
                    b_B = (j // 2) * 2 + (k // 2)
                    l_B = (j % 2) * 2 + (k % 2)
                    V4[idx_B, t] = V_block[b_B, k] * V2[l_B, s]
                    
                    idx_C = k * 4 + i # C[i, k] -> flat k*4+i
                    b_C = (k // 2) * 2 + (i // 2) # Wait, C is 4x4. C[i, k].
                    # Block of C: row i, col k.
                    # b_C = (i//2)*2 + (k//2)
                    b_C = (i // 2) * 2 + (k // 2)
                    l_C = (i % 2) * 2 + (k % 2)
                    W4[idx_C, t] = W_block[b_C, k] * W2[l_C, s]
                    
    return U4, V4, W4

def _hybrid_5x5():
    n, m, p = 5, 5, 5
    rank_core = 49
    rank_boundaries = 49
    total_rank = rank_core + rank_boundaries
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    # 1. Core 4x4 (Rank 49)
    U4, V4, W4 = _strassen_4x4()
    for i in range(4):
        for j in range(4):
            idx_A = i * 5 + j
            for t in range(49):
                U[idx_A, t] = U4[i * 4 + j, t]
                
                idx_B = j * 5 + t # No, B index is (j, k).
                # We need to map V4 indices (j*4+k) to V indices (j*5+k)
                # But V4 is already flattened 4x4.
                # V4[row, col] where row = j*4+k.
                # We need to fill V[j*5+k, t].
                
    # Let's do it more carefully
    for i in range(4):
        for j in range(4):
            idx_A = i * 5 + j
            for t in range(49):
                U[idx_A, t] = U4[i * 4 + j, t]
                
    for j in range(4):
        for k in range(4):
            idx_B = j * 5 + k
            for t in range(49):
                V[idx_B, t] = V4[j * 4 + k, t]
                
    for i in range(4):
        for k in range(4):
            idx_W = k * 5 + i
            for t in range(49):
                W[idx_W, t] = W4[k * 4 + i, t]
                
    # 2. Boundaries (Rank 49)
    col_offset = 49
    
    # Term 2: u z^T (Rank 4)
    # C[i, k] += u_i z_k
    # u_i = A[i, 4], z_k = B[4, k]
    for i in range(4):
        for k in range(4):
            c = col_offset + i * 4 + k
            U[i * 5 + 4, c] = 1
            V[4 * 5 + k, c] = 1
            W[k * 5 + i, c] = 1
    col_offset += 16 # Wait, u z^T is 4x4 outer product? No.
    # u is 4x1, z^T is 1x4.
    # C[i, k] += u_i z_k.
    # This is 16 terms? No.
    # u_i z_k is a single product for each (i, k).
    # So rank 16?
    # Let's re-evaluate rank of u z^T.
    # u z^T = [u_0 z_0, ..., u_0 z_3; ...; u_3 z_0, ..., u_3 z_3]
    # This matrix has rank 1.
    # But in tensor decomposition, we need to compute each entry C[i, k].
    # C[i, k] depends on u_i and z_k.
    # We can compute 16 scalar products: u_i * z_k for each i, k.
    # Or we can compute 4 products u_i * 1 and 1 * z_k? No, we need u_i z_k.
    # We can use 4 products if we precompute? No, exact tensor decomposition.
    # The term is $u z^T$.
    # $C[i, k] = u_i z_k$.
    # We can define terms $t_{i,k}$ where $U[i, 4] = 1, V[4, k] = 1, W[k, i] = 1$.
    # This requires 16 terms (one for each i, k).
    # So rank 16.
    
    # My previous estimate was Rank 4 for u z^T. Why?
    # Maybe I thought it was 1 term? No.
    # Let's re-sum:
    # 1. AB (49)
    # 2. u z^T (16)
    # 3. A w (16)
    # 4. u beta (4)
    # 5. v^T B (16)
    # 6. alpha z^T (4)
    # 7. v^T w (4)
    # 8. alpha beta (1)
    # Total = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
    # 110 < 125. Still an improvement.
    
    # Let's correct the code for u z^T
    col_offset = 49
    
    # Term 2: u z^T (Rank 16)
    for i in range(4):
        for k in range(4):
            c = col_offset
            U[i * 5 + 4, c] = 1
            V[4 * 5 + k, c] = 1
            W[k * 5 + i, c] = 1
            col_offset += 1
            
    # Term 3: A w (Rank 16)
    # C[i, 4] += sum_j A[i, j] w_j
    for i in range(4):
        for j in range(4):
            c = col_offset
            U[i * 5 + j, c] = 1
            V[j * 5 + 4, c] = 1
            W[4 * 5 + i, c] = 1
            col_offset += 1
            
    # Term 4: u beta (Rank 4)
    # C[i, 4] += u_i beta
    for i in range(4):
        c = col_offset
        U[i * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[4 * 5 + i, c] = 1
        col_offset += 1
        
    # Term 5: v^T B (Rank 16)
    # C[4, k] += sum_j v_j B[j, k]
    for j in range(4):
        for k in range(4):
            c = col_offset
            U[4 * 5 + j, c] = 1
            V[j * 5 + k, c] = 1
            W[k * 5 + 4, c] = 1
            col_offset += 1
            
    # Term 6: alpha z^T (Rank 4)
    # C[4, k] += alpha z_k
    for k in range(4):
        c = col_offset
        U[4 * 5 + 4, c] = 1
        V[4 * 5 + k, c] = 1
        W[k * 5 + 4, c] = 1
        col_offset += 1
        
    # Term 7: v^T w (Rank 4)
    # C[4, 4] += sum_j v_j w_j
    for j in range(4):
        c = col_offset
        U[4 * 5 + j, c] = 1
        V[j * 5 + 4, c] = 1
        W[4 * 5 + 4, c] = 1
        col_offset += 1
        
    # Term 8: alpha beta (Rank 1)
    c = col_offset
    U[4 * 5 + 4, c] = 1
    V[4 * 5 + 4, c] = 1
    W[4 * 5 + 4, c] = 1
    col_offset += 1
    
    # Total rank should be 110
    # Check col_offset
    # 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-10):
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = U_arr @ A_flat
    R = V_arr @ B_flat
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            idx = j * n + i # W[k*n+i] where k=j, i=i
            C_computed[i, j] = np.sum(W_arr[idx, :] * L * R)
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try Hybrid
    try:
        result = _hybrid_5x5()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
        
    # Fallback
    result = _schoolbook(n, m, p)
    return result
