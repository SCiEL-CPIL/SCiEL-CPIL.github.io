import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def _schoolbook_4x4_by_4x9():
    """
    Returns U, V, W for 4x4 by 4x9 multiplication using schoolbook.
    Rank is 4*4*9 = 144.
    U shape: (16, 144)
    V shape: (36, 144)
    W shape: (36, 144)
    """
    n, m, p = 4, 4, 9
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_by_2x2():
    """
    Returns U, V, W for 2x2 by 2x2 multiplication using Strassen.
    Rank is 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    # Indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = -1; W[2, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_by_4x4():
    """
    Returns U, V, W for 4x4 by 4x4 multiplication using Strassen.
    Rank is 49.
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    """
    U_sub, V_sub, W_sub = _strassen_2x2_by_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block indices for 4x4 matrix:
    # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3)
    # 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3)
    # 8: (2,0), 9: (2,1), 10: (2,2), 11: (2,3)
    # 12: (3,0), 13: (3,1), 14: (3,2), 15: (3,3)
    
    # Strassen blocks:
    # A11 = [0,1,4,5]
    # A12 = [2,3,6,7]
    # A21 = [8,9,12,13]
    # A22 = [10,11,14,15]
    
    a11 = [0, 1, 4, 5]
    a12 = [2, 3, 6, 7]
    a21 = [8, 9, 12, 13]
    a22 = [10, 11, 14, 15]
    
    b11 = [0, 1, 4, 5]
    b12 = [2, 3, 6, 7]
    b21 = [8, 9, 12, 13]
    b22 = [10, 11, 14, 15]
    
    c11 = [0, 1, 4, 5]
    c12 = [2, 3, 6, 7]
    c21 = [8, 9, 12, 13]
    c22 = [10, 11, 14, 15]
    
    for r in range(7):
        for k in range(7):
            col = r * 7 + k
            
            # U contributions
            for idx, val in enumerate(U_sub[:, r]):
                if val != 0:
                    # Determine which block this index belongs to
                    if idx < 2:
                        # Top-left block (A11)
                        U[a11[idx], col] += val
                    elif idx < 4:
                        # Top-right block (A12)
                        U[a12[idx - 2], col] += val
                    elif idx < 6:
                        # Bottom-left block (A21)
                        U[a21[idx - 4], col] += val
                    else:
                        # Bottom-right block (A22)
                        U[a22[idx - 6], col] += val
            
            # V contributions
            for idx, val in enumerate(V_sub[:, k]):
                if val != 0:
                    if idx < 2:
                        V[b11[idx], col] += val
                    elif idx < 4:
                        V[b12[idx - 2], col] += val
                    elif idx < 6:
                        V[b21[idx - 4], col] += val
                    else:
                        V[b22[idx - 6], col] += val
            
            # W contributions
            for idx, val in enumerate(W_sub[:, r]):
                if val != 0:
                    if idx < 2:
                        W[c11[idx], col] += val
                    elif idx < 4:
                        W[c12[idx - 2], col] += val
                    elif idx < 6:
                        W[c21[idx - 4], col] += val
                    else:
                        W[c22[idx - 6], col] += val
    
    return U, V, W

def _combine_decompositions(decomps):
    """
    Combines a list of (U, V, W) decompositions into one.
    Each decomposition must have the same U, V, W shapes.
    """
    if not decomps:
        return None
    
    U0, V0, W0 = decomps[0]
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    
    U = np.zeros((U0.shape[0], total_rank))
    V = np.zeros((V0.shape[0], total_rank))
    W = np.zeros((W0.shape[0], total_rank))
    
    col = 0
    for U_d, V_d, W_d in decomps:
        rank_d = U_d.shape[1]
        U[:, col:col+rank_d] = U_d
        V[:, col:col+rank_d] = V_d
        W[:, col:col+rank_d] = W_d
        col += rank_d
    
    return U, V, W

def _verify_decomposition(U, V, W, n=4, m=4, p=9):
    """
    Verifies that the decomposition is correct.
    """
    # Construct the tensor T from the decomposition
    # T[i*m+j, j*p+k, k*n+i] should be 1
    # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    
    # This is expensive, so we check a few random indices
    # Or better, check that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # For efficiency, we can check a subset
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        u_val = U[i*m+j]
        v_val = V[j*p+k]
        w_val = W[k*n+i]
        
        result = np.dot(u_val, v_val * w_val)
        if abs(result - 1.0) > 1e-9:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Strategy:
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # 
    # For A*B1 and A*B2 (4x4 by 4x4), use Strassen (rank 49 each)
    # For A*B3 (4x4 by 4x1), use schoolbook (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    # However, we need to be careful about how we split B.
    # B is 4x9. We split it into columns 0-3, 4-7, 8.
    # B1 = B[:, 0:4] (4x4)
    # B2 = B[:, 4:8] (4x4)
    # B3 = B[:, 8:9] (4x1)
    
    # The decomposition for A*B1:
    # U1, V1, W1 from Strassen 4x4 by 4x4
    # But we need to map the V indices correctly.
    # In the full problem, V has shape (36, rank).
    # For B1, the relevant V indices are j*9 + k for k in 0..3.
    
    # This is getting complex. Let's try a different approach.
    # Instead of splitting B, let's just use the schoolbook for now
    # and see if we can improve.
    
    # Actually, let's try to implement the Strassen-based approach properly.
    
    # For the 4x4 by 4x4 part, we have U, V, W of shapes (16, 49), (16, 49), (16, 49).
    # But in our problem, V has shape (36, rank) because B is 4x9.
    # So we need to embed the 4x4 V into the 4x9 V.
    
    # Let's define the mapping:
    # For B1 (columns 0-3): V1[j, k] -> V_full[j*9 + k, col] for k in 0..3
    # For B2 (columns 4-7): V2[j, k] -> V_full[j*9 + (k+4), col] for k in 0..3
    # For B3 (column 8): V3[j, 0] -> V_full[j*9 + 8, col] for j in 0..3
    
    # Similarly for W:
    # For C1 (columns 0-3): W1[k, i] -> W_full[k*4 + i, col] for k in 0..3
    # For C2 (columns 4-7): W2[k, i] -> W_full[k*4 + i, col] for k in 4..7
    # For C3 (column 8): W3[k, i] -> W_full[k*4 + i, col] for k in 8
    
    # Wait, W has shape (36, rank) where index is k*4 + i.
    # k ranges from 0 to 8.
    # So for C1, k in 0..3, indices are 0..15
    # For C2, k in 4..7, indices are 16..31
    # For C3, k in 8, indices are 32..35
    
    # Let's implement this.
    
    U_strassen, V_strassen, W_strassen = _strassen_4x4_by_4x4()
    
    # For B3 (4x1), use schoolbook
    U_b3 = np.zeros((16, 16))
    V_b3 = np.zeros((36, 16))
    W_b3 = np.zeros((36, 16))
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_b3[i * 4 + j, col] = 1
            V_b3[j * 9 + 8, col] = 1
            W_b3[8 * 4 + i, col] = 1
    
    # Now combine:
    # Part 1: A * B1 (B1 is columns 0-3 of B)
    U1 = np.zeros((16, 49))
    V1 = np.zeros((36, 49))
    W1 = np.zeros((36, 49))
    for r in range(49):
        U1[:, r] = U_strassen[:, r]
        for j in range(4):
            for k in range(4):
                V1[j * 9 + k, r] = V_strassen[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W1[k * 4 + i, r] = W_strassen[k * 4 + i, r]
    
    # Part 2: A * B2 (B2 is columns 4-7 of B)
    U2 = np.zeros((16, 49))
    V2 = np.zeros((36, 49))
    W2 = np.zeros((36, 49))
    for r in range(49):
        U2[:, r] = U_strassen[:, r]
        for j in range(4):
            for k in range(4):
                V2[j * 9 + (k + 4), r] = V_strassen[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W2[(k + 4) * 4 + i, r] = W_strassen[k * 4 + i, r]
    
    # Part 3: A * B3 (B3 is column 8 of B)
    U3 = U_b3
    V3 = V_b3
    W3 = W_b3
    
    # Combine all parts
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    # Verify
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook_4x4_by_4x9()
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
