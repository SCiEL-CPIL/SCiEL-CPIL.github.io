"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using explicit loops.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        
        u_2d = u_c.reshape(n, m)
        v_2d = v_c.reshape(m, p)
        w_2d = w_c.reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A0 (1x4) and A12 (2x4).
    A0 x B (4x8): Schoolbook, rank 32.
    A12 x B (4x8): Split A12 into two 2x2 blocks (left/right).
                   Split B into four 2x2 blocks (top/bottom, left/right).
                   8 sub-problems of 2x2x2, each Strassen (rank 7).
                   Total rank 56.
    Grand total: 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2 blocks
    # A rows 1-2, split cols into [0,1] and [2,3]
    # B split into 4 blocks of 2x2: cols [0,1], [2,3], [4,5], [6,7]
    # Actually B is 4x8. Rows 0-1 and 2-3. Cols 0-1, 2-3, 4-5, 6-7.
    
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T # shape (4, 7)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T # shape (4, 7)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]).T # shape (4, 7)
    
    # A blocks: 
    # Block 0: rows 1,2; cols 0,1. Indices in U: (1*4+0)..(1*4+1), (2*4+0)..(2*4+1)
    # Block 1: rows 1,2; cols 2,3. Indices in U: (1*4+2)..(1*4+3), (2*4+2)..(2*4+3)
    
    # B blocks:
    # Block 0: rows 0,1; cols 0,1.
    # Block 1: rows 0,1; cols 2,3.
    # Block 2: rows 2,3; cols 0,1.
    # Block 3: rows 2,3; cols 2,3.
    # Wait, the multiplication is A(2x4) * B(4x8).
    # C(2x8) = A_L(2x2)*B_TL(2x2) + A_R(2x2)*B_TR(2x2) ...
    # Actually:
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    # C[:, 4:6] = A[:, 0:2] * B[0:2, 4:6] + A[:, 2:4] * B[2:4, 4:6]
    # C[:, 6:8] = A[:, 0:2] * B[0:2, 6:8] + A[:, 2:4] * B[2:4, 6:8]
    
    # So for each k_block in 0..3:
    #   Term 1: A_L * B_TL (rows 0-1 of B)
    #   Term 2: A_R * B_BL (rows 2-3 of B)
    
    for k_block in range(4):
        k_start = k_block * 2
        
        # Sub-problem 1: A_L (cols 0-1) * B_TL (rows 0-1, cols k_start..k_start+1)
        # A_L indices: U[1*4+0:2*4+2] -> U[4:6] and U[8:10]
        # B_TL indices: V[0*p+k_start:2*p+k_start+2] -> V[k_start:k_start+2] and V[4+k_start:4+k_start+2]
        
        for t in range(7):
            # U
            # A_L is 2x2.
            # U indices: (1,0), (1,1), (2,0), (2,1) -> 4, 5, 8, 9
            # strassen_U is 4x7.
            # Row 0 of strassen_U corresponds to A[0,0]
            # Row 1 to A[0,1]
            # Row 2 to A[1,0]
            # Row 3 to A[1,1]
            
            U[4, col] += strassen_U[0, t]
            U[5, col] += strassen_U[1, t]
            U[8, col] += strassen_U[2, t]
            U[9, col] += strassen_U[3, t]
            
            # V
            # B_TL is 2x2.
            # V indices: (0, k_start), (0, k_start+1), (1, k_start), (1, k_start+1)
            # -> k_start, k_start+1, p+k_start, p+k_start+1
            V[k_start, col] += strassen_V[0, t]
            V[k_start+1, col] += strassen_V[1, t]
            V[p+k_start, col] += strassen_V[2, t]
            V[p+k_start+1, col] += strassen_V[3, t]
            
            # W
            # C is 2x2.
            # W indices: (k_start, 1), (k_start+1, 1), (k_start, 2), (k_start+1, 2)
            # Convention: W[k*n + i]
            # i=1, k=k_start -> k_start*3 + 1
            # i=1, k=k_start+1 -> (k_start+1)*3 + 1
            # i=2, k=k_start -> k_start*3 + 2
            # i=2, k=k_start+1 -> (k_start+1)*3 + 2
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1
            
        # Sub-problem 2: A_R (cols 2-3) * B_BL (rows 2-3, cols k_start..k_start+1)
        # A_R indices: U[1*4+2:2*4+4] -> U[6:8] and U[10:12]
        # B_BL indices: V[2*p+k_start:4*p+k_start+2] -> V[8+k_start:8+k_start+2] and V[12+k_start:12+k_start+2]
        
        for t in range(7):
            # U
            U[6, col] += strassen_U[0, t]
            U[7, col] += strassen_U[1, t]
            U[10, col] += strassen_U[2, t]
            U[11, col] += strassen_U[3, t]
            
            # V
            V[2*p+k_start, col] += strassen_V[0, t]
            V[2*p+k_start+1, col] += strassen_V[1, t]
            V[3*p+k_start, col] += strassen_V[2, t]
            V[3*p+k_start+1, col] += strassen_V[3, t]
            
            # W
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 88
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
    
    return best_result
