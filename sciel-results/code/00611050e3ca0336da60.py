"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _construct_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Splits B into two 4x4 blocks. Each block uses rank-49 Strassen.
    Total rank 98.
    """
    n, m, p = DIMENSIONS # 4, 4, 8
    
    # 2x2 Strassen factors
    # U2, V2, W2 are 7x4 matrices (7 factors, 4 elements each)
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Build 4x4x4 Strassen factors (49 of them)
    # Each factor is a triple of 4x4 matrices (flattened to 16)
    # We will generate these on the fly for each of the 2 blocks
    
    for t in range(2): # 2 blocks
        for s in range(7): # Top level Strassen factor
            for r in range(7): # Scalar level Strassen factor
                col = t * 49 + s * 7 + r
                
                # Compute U4, V4, W4 for this (s,r) pair
                # U4 is 4x4
                U4 = np.zeros(16)
                V4 = np.zeros(16)
                W4 = np.zeros(16)
                
                for I in range(4):
                    for J in range(4):
                        r_block = I // 2
                        c_block = J // 2
                        i_loc = I % 2
                        j_loc = J % 2
                        coeff = U2[s, r_block*2+c_block] * U2[r, i_loc*2+j_loc]
                        U4[I*4+J] = coeff
                        
                for J in range(4):
                    for K in range(4):
                        r_block = J // 2
                        c_block = K // 2
                        j_loc = J % 2
                        k_loc = K % 2
                        coeff = V2[s, r_block*2+c_block] * V2[r, j_loc*2+k_loc]
                        V4[J*4+K] = coeff
                        
                for I in range(4):
                    for K in range(4):
                        r_block = I // 2
                        c_block = K // 2
                        i_loc = I % 2
                        k_loc = K % 2
                        coeff = W2[s, r_block*2+c_block] * W2[r, i_loc*2+k_loc]
                        # W4 index is I, K. W matrix index is K, I (k*n+i)
                        W4[I*4+K] = coeff
                
                # Fill U
                U[:, col] = U4
                
                # Fill V
                # V index is j*p+k. p=8.
                # For block t, k is offset by 4*t
                for j in range(4):
                    for k in range(4):
                        V[j*8 + (k + 4*t), col] = V4[j*4+k]
                        
                # Fill W
                # W index is k*n+i. n=4.
                # For block t, k is offset by 4*t
                for k in range(4):
                    for i in range(4):
                        W[(k + 4*t)*4 + i, col] = W4[i*4+k]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_strassen_rect_4x4_4x8():
    """
    Construct a rank-56 decomposition for 4x4 * 4x8 using Strassen(2x2x2) x rect(2x2x4).
    
    The 4x4x8 multiplication tensor factors as:
    - Split A (4x4) into 2x2 blocks of 2x2 submatrices
    - Split B (4x8) into 2x4 blocks of 2x4 submatrices  
    - Split C (8x4) into 4x2 blocks of 4x2 submatrices
    
    For each pair of 2x2 blocks (I,J) of A and 2x4 block (J,K) of B:
      C[I,K] += A[I,J] * B[J,K]  (2x2 * 2x4 = 2x4)
    
    The 2x2x2 Strassen gives rank 7 for the block-level multiplication.
    The 2x2x4 rectangular multiplication has rank 8 (schoolbook: each output
    element C[i,k] = sum_j A[i,j]*B[j,k], with 4 outputs x 2 terms = 8).
    
    Total rank: 7 * 8 = 56
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # 2x2x2 Strassen factors (7 factors)
    # U2[r, i*2+j], V2[r, j*2+k], W2[r, k*2+i]
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    # 2x2x4 rectangular multiplication factors (8 factors, schoolbook)
    # A is 2x2, B is 2x4, C is 4x2 (output is C[k, i] in trace convention k*n+i)
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # For each (i,j,k): one factor
    # U_rect: (2*2, 8) = (4, 8), indexed by [i*2+j, factor]
    # V_rect: (2*4, 8) = (8, 8), indexed by [j*4+k, factor]
    # W_rect: (4*2, 8) = (8, 8), indexed by [k*2+i, factor]
    
    U_rect = np.zeros((4, 8), dtype=float)
    V_rect = np.zeros((8, 8), dtype=float)
    W_rect = np.zeros((8, 8), dtype=float)
    
    f = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U_rect[i*2+j, f] = 1.0
                V_rect[j*4+k, f] = 1.0
                W_rect[k*2+i, f] = 1.0
                f += 1
    
    # Verify the 2x2x4 decomposition
    # sum_f U_rect[i*2+j, f] * V_rect[j*4+k, f] * W_rect[k*2+i, f] should be 1 for all (i,j,k)
    U3r = U_rect.reshape(2, 2, 8)
    V3r = V_rect.reshape(2, 4, 8)
    W3r = W_rect.reshape(4, 2, 8)
    check = np.einsum('ijf,jkf,kif->ijk', U3r, V3r, W3r)
    assert np.allclose(check, 1.0, atol=1e-10), f"2x2x4 check failed: {check}"
    
    rank = 7 * 8  # 56
    U = np.zeros((n * m, rank), dtype=float)  # (16, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (32, 56)
    
    for s in range(7):   # Strassen factor index
        for r in range(8):  # Rectangular factor index
            col = s * 8 + r
            
            # U: 4x4 matrix, indexed by [i*4+j]
            # i = I*2 + i_loc, j = J*2 + j_loc
            # U[i*4+j, col] = U2[s, I*2+J] * U_rect[i_loc*2+j_loc, r]
            for I in range(2):
                for J in range(2):
                    for i_loc in range(2):
                        for j_loc in range(2):
                            i = I * 2 + i_loc
                            j = J * 2 + j_loc
                            U[i*4+j, col] = U2[s, I*2+J] * U_rect[i_loc*2+j_loc, r]
            
            # V: 4x8 matrix, indexed by [j*8+k]
            # j = J*2 + j_loc, k = K*4 + k_loc (K is 0 or 1, k_loc is 0-3)
            # V[j*8+k, col] = V2[s, J*2+K] * V_rect[j_loc*4+k_loc, r]
            for J in range(2):
                for K in range(2):
                    for j_loc in range(2):
                        for k_loc in range(4):
                            j = J * 2 + j_loc
                            k = K * 4 + k_loc
                            V[j*8+k, col] = V2[s, J*2+K] * V_rect[j_loc*4+k_loc, r]
            
            # W: 8x4 matrix, indexed by [k*4+i]
            # k = K*4 + k_loc, i = I*2 + i_loc
            # W[k*4+i, col] = W2[s, K*2+I] * W_rect[k_loc*2+i_loc, r]
            for K in range(2):
                for I in range(2):
                    for k_loc in range(4):
                        for i_loc in range(2):
                            k = K * 4 + k_loc
                            i = I * 2 + i_loc
                            W[k*4+i, col] = W2[s, K*2+I] * W_rect[k_loc*2+i_loc, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries rank-56 (Strassen x rectangular) first, falls back to rank-98, then schoolbook.
    """
    start_time = time.time()
    deadline = 150.0 - 10.0  # 140s safety margin
    
    # Try rank-56 first
    try:
        result = _construct_strassen_rect_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        # Fallback to schoolbook if out of time
        n, m, p = DIMENSIONS
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
    
    # Try rank-98 Strassen
    try:
        result = _construct_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
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

# EVOLVE-BLOCK-END
