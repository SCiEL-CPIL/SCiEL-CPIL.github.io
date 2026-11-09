import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def get_strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen, shapes (4, 7)
    Index convention: row index = i*2 + j for (i,j) in matrix
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    U[:, 0] = [1, 0, 0, 1]   # A00+A11
    V[:, 0] = [1, 0, 0, 1]   # B00+B11
    W[:, 0] = [1, 0, 0, 0]   # C00 +1
    
    U[:, 1] = [0, 0, 1, 1]   # A10+A11
    V[:, 1] = [1, 0, 0, 0]   # B00
    W[:, 1] = [0, 0, 1, 0]   # C10 +1
    
    U[:, 2] = [1, 0, 0, 0]   # A00
    V[:, 2] = [0, 1, 0, -1]  # B01-B11
    W[:, 2] = [0, 1, 0, 0]   # C01 +1
    
    U[:, 3] = [0, 0, 0, 1]   # A11
    V[:, 3] = [0, 0, 1, -1]  # B10-B00
    W[:, 3] = [-1, 0, 1, 0]  # C00 -1, C10 +1
    
    U[:, 4] = [1, 1, 0, 0]   # A00+A01
    V[:, 4] = [0, 0, 0, 1]   # B11
    W[:, 4] = [0, 1, 0, 0]   # C01 +1
    
    U[:, 5] = [1, 0, -1, 0]  # A10-A00
    V[:, 5] = [1, 1, 0, 0]   # B00+B01
    W[:, 5] = [1, 0, 0, 0]   # C00 +1
    
    U[:, 6] = [0, 1, 0, -1]  # A01-A11
    V[:, 6] = [0, 0, 1, 1]   # B10+B11
    W[:, 6] = [0, 0, 0, 1]   # C11 +1
    
    return U, V, W

def get_schoolbook(n, m, p):
    """Returns U, V, W for n×m × m×p schoolbook, shapes (n*m, n*m*p), (m*p, n*m*p), (p*n, n*m*p)
    Index convention: U[i*m+j, k], V[j*p+k, k], W[k*n+i, k] for k in range(n*m*p)
    Actually, the index for the rank term is a triple (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    """
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

def construct():
    n, m, p = 4, 5, 5  # A is n×m, B is m×p, C is n×p
    
    # Split the middle dimension m=5 into 2+3
    m1, m2 = 2, 3  # m1 + m2 = 5
    
    # Split the output dimension p=5 into 2+3
    p1, p2 = 2, 3  # p1 + p2 = 5
    
    # A is 4×5 = [A_left (4×2) | A_right (4×3)]
    # B is 5×5 = [B_top (2×5); B_bottom (3×5)]
    #   B_top = [B_tl (2×2) | B_tr (2×3)]
    #   B_bottom = [B_bl (3×2) | B_br (3×3)]
    # C is 4×5 = [C_left (4×2) | C_right (4×3)]
    #   C_left = A_left × B_tl + A_right × B_bl
    #   C_right = A_left × B_tr + A_right × B_br
    
    # Four sub-problems:
    # 1. A_left (4×2) × B_tl (2×2) -> contributes to C_left (4×2): use Strassen, rank 7
    # 2. A_right (4×3) × B_bl (3×2) -> contributes to C_left (4×2): schoolbook, rank 4*3*2=24
    # 3. A_left (4×2) × B_tr (2×3) -> contributes to C_right (4×3): schoolbook, rank 4*2*3=24
    # 4. A_right (4×3) × B_br (3×3) -> contributes to C_right (4×3): schoolbook, rank 4*3*3=36
    
    # Total rank: 7 + 24 + 24 + 36 = 91
    # That's still less than 100!
    
    total_rank = 7 + 24 + 24 + 36
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Sub-problem 1: A_left (4×2) × B_tl (2×2) -> C_left (4×2)
    # A_left indices in A: rows 0-3, cols 0-1 -> U index = i*5 + j for i in [0,4), j in [0,2)
    # B_tl indices in B: rows 0-1, cols 0-1 -> V index = j*5 + k for j in [0,2), k in [0,2)
    # C_left indices in C: rows 0-3, cols 0-1 -> W index = k*4 + i for k in [0,2), i in [0,4)
    
    U1, V1, W1 = get_strassen_2x2x2()
    # U1 is (4,7), V1 is (4,7), W1 is (4,7)
    # Need to map indices
    
    for r in range(7):
        for a_idx in range(4):  # a_idx is row in 2x2 A part
            i, j = a_idx // 2, a_idx % 2  # i in [0,2), j in [0,2)
            # In full A (4x5), this is row i, col j -> U index = i*5 + j
            U[i * 5 + j, col_offset + r] += U1[a_idx, r]
        
        for b_idx in range(4):  # b_idx is row in 2x2 B part
            j, k = b_idx // 2, b_idx % 2  # j in [0,2), k in [0,2)
            # In full B (5x5), this is row j, col k -> V index = j*5 + k
            V[j * 5 + k, col_offset + r] += V1[b_idx, r]
        
        for c_idx in range(4):  # c_idx is row in 2x2 C part
            k, i = c_idx // 2, c_idx % 2  # k in [0,2), i in [0,2)
            # In full C (4x5), this is row i, col k -> W index = k*4 + i
            W[k * 4 + i, col_offset + r] += W1[c_idx, r]
    
    col_offset += 7
    
    # Sub-problem 2: A_right (4×3) × B_bl (3×2) -> C_left (4×2)
    # A_right: rows 0-3, cols 2-4 -> U index = i*5 + (2+j) for i in [0,4), j in [0,3)
    # B_bl: rows 2-4, cols 0-1 -> V index = (2+j)*5 + k for j in [0,3), k in [0,2)
    # C_left: rows 0-3, cols 0-1 -> W index = k*4 + i for k in [0,2), i in [0,4)
    
    U2, V2, W2 = get_schoolbook(4, 3, 2)
    # U2 is (12, 24), V2 is (6, 24), W2 is (8, 24)
    
    for r in range(24):
        # Map U2 indices: row = i*3 + j for i in [0,4), j in [0,3)
        for u_idx in range(12):
            i, j = u_idx // 3, u_idx % 3
            full_u = i * 5 + (2 + j)
            U[full_u, col_offset + r] += U2[u_idx, r]
        
        # Map V2 indices: row = j*2 + k for j in [0,3), k in [0,2)
        for v_idx in range(6):
            j, k = v_idx // 2, v_idx % 2
            full_v = (2 + j) * 5 + k
            V[full_v, col_offset + r] += V2[v_idx, r]
        
        # Map W2 indices: row = k*4 + i for k in [0,2), i in [0,4)
        for w_idx in range(8):
            k, i = w_idx // 4, w_idx % 4
            full_w = k * 4 + i
            W[full_w, col_offset + r] += W2[w_idx, r]
    
    col_offset += 24
    
    # Sub-problem 3: A_left (4×2) × B_tr (2×3) -> C_right (4×3)
    # A_left: rows 0-3, cols 0-1 -> U index = i*5 + j for i in [0,4), j in [0,2)
    # B_tr: rows 0-1, cols 2-4 -> V index = j*5 + (2+k) for j in [0,2), k in [0,3)
    # C_right: rows 0-3, cols 2-4 -> W index = (2+k)*4 + i for k in [0,3), i in [0,4)
    
    U3, V3, W3 = get_schoolbook(4, 2, 3)
    # U3 is (8, 24), V3 is (6, 24), W3 is (12, 24)
    
    for r in range(24):
        # Map U3 indices: row = i*2 + j for i in [0,4), j in [0,2)
        for u_idx in range(8):
            i, j = u_idx // 2, u_idx % 2
            full_u = i * 5 + j
            U[full_u, col_offset + r] += U3[u_idx, r]
        
        # Map V3 indices: row = j*3 + k for j in [0,2), k in [0,3)
        for v_idx in range(6):
            j, k = v_idx // 3, v_idx % 3
            full_v = j * 5 + (2 + k)
            V[full_v, col_offset + r] += V3[v_idx, r]
        
        # Map W3 indices: row = k*4 + i for k in [0,3), i in [0,4)
        for w_idx in range(12):
            k, i = w_idx // 4, w_idx % 4
            full_w = (2 + k) * 4 + i
            W[full_w, col_offset + r] += W3[w_idx, r]
    
    col_offset += 24
    
    # Sub-problem 4: A_right (4×3) × B_br (3×3) -> C_right (4×3)
    # A_right: rows 0-3, cols 2-4 -> U index = i*5 + (2+j) for i in [0,4), j in [0,3)
    # B_br: rows 2-4, cols 2-4 -> V index = (2+j)*5 + (2+k) for j in [0,3), k in [0,3)
    # C_right: rows 0-3, cols 2-4 -> W index = (2+k)*4 + i for k in [0,3), i in [0,4)
    
    U4, V4, W4 = get_schoolbook(4, 3, 3)
    # U4 is (12, 36), V4 is (9, 36), W4 is (12, 36)
    
    for r in range(36):
        # Map U4 indices: row = i*3 + j for i in [0,4), j in [0,3)
        for u_idx in range(12):
            i, j = u_idx // 3, u_idx % 3
            full_u = i * 5 + (2 + j)
            U[full_u, col_offset + r] += U4[u_idx, r]
        
        # Map V4 indices: row = j*3 + k for j in [0,3), k in [0,3)
        for v_idx in range(9):
            j, k = v_idx // 3, v_idx % 3
            full_v = (2 + j) * 5 + (2 + k)
            V[full_v, col_offset + r] += V4[v_idx, r]
        
        # Map W4 indices: row = k*4 + i for k in [0,3), i in [0,4)
        for w_idx in range(12):
            k, i = w_idx // 4, w_idx % 4
            full_w = (2 + k) * 4 + i
            W[full_w, col_offset + r] += W4[w_idx, r]
    
    col_offset += 36
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
