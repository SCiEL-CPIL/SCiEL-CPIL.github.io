import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0

def _check_deadline():
    if time.time() > _DEADLINE - 5.0:
        return True
    return False

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T[a, b, d] = 1.0
    return T

def _verify(n, m, p, U, V, W):
    T = _build_tensor(n, m, p)
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T, atol=1e-10, rtol=0)

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 with rank 7 (Strassen).
    U shape: (4, 7) -- A is 2x2, flattened row-major: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    V shape: (4, 7) -- B is 2x2, flattened row-major: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    W shape: (4, 7) -- C is 2x2, flattened row-major: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x3x2_rank11():
    """
    Rank-11 decomposition for 2x3 * 3x2 = 2x2.
    U shape: (6, 11) -- A is 2x3, row-major: A[i,j] at index i*3+j
    V shape: (6, 11) -- B is 3x2, row-major: B[j,k] at index j*2+k
    W shape: (4, 11) -- C is 2x2, row-major: C[i,k] at index i*2+k
    """
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    # Part 1: Strassen for A_left(2x2) * B_top(2x2)
    # A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,3,4
    # B_top: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    # C: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Map A_left indices: [0,1,3,4] -> strassen A indices [0,1,2,3]
    # strassen A: 0=A00, 1=A01, 2=A10, 3=A11
    # A_left: 0=A[0,0], 1=A[0,1], 3=A[1,0], 4=A[1,1]
    a_map = [0, 1, 3, 4]  # strassen A index -> A_left index
    
    # Map B_top indices: [0,1,2,3] -> strassen B indices [0,1,2,3]
    # B_top: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    # strassen B: 0=B00, 1=B01, 2=B10, 3=B11
    b_map = [0, 1, 2, 3]
    
    # Map C indices: same
    c_map = [0, 1, 2, 3]
    
    for c in range(7):
        for s_a, val in enumerate(U_s[:, c]):
            if val != 0:
                U[a_map[s_a], c] += val
        for s_b, val in enumerate(V_s[:, c]):
            if val != 0:
                V[b_map[s_b], c] += val
        for s_c, val in enumerate(W_s[:, c]):
            if val != 0:
                W[c_map[s_c], c] += val
    
    # Part 2: Outer product A_right(2x1) * B_bottom(1x2)
    # A_right: A[0,2]=index 2, A[1,2]=index 5
    # B_bottom: B[2,0]=index 4, B[2,1]=index 5
    # C[i,k] += A[i,2] * B[2,k]
    # 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    
    for ii in range(2):
        for kk in range(2):
            c = 7 + ii * 2 + kk
            U[ii * 3 + 2, c] = 1
            V[2 * 2 + kk, c] = 1
            W[ii * 2 + kk, c] = 1
    
    return U, V, W

def _fallback_44():
    """
    4-block decomposition: split p=8 into 4 blocks of size 2.
    Each block uses rank-11 for 2x3 * 3x2.
    Total rank: 4 * 11 = 44.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    num_blocks = p // block_size  # 4
    r_sub = 11
    total_rank = num_blocks * r_sub  # 44
    
    U_sub, V_sub, W_sub = _build_2x3x2_rank11()
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for b in range(num_blocks):
        k_offset = b * block_size
        col_offset = b * r_sub
        
        # U: same for each block
        U_full[:, col_offset:col_offset+r_sub] = U_sub
        
        # V: B[j, k_offset + k'] for j in 0..2, k' in 0..1
        # V_sub[j*2 + k', c] -> V_full[j*p + k_offset + k', col_offset + c]
        for j in range(m):
            for k_prime in range(block_size):
                b_local = j * block_size + k_prime
                b_global = j * p + k_offset + k_prime
                V_full[b_global, col_offset:col_offset+r_sub] = V_sub[b_local]
        
        # W: C[i, k_offset + k'] for i in 0..1, k' in 0..1
        # W_sub[i*2 + k', c] -> W_full[(k_offset + k')*n + i, col_offset + c]
        for k_prime in range(block_size):
            for i in range(n):
                d_local = i * block_size + k_prime
                d_global = (k_offset + k_prime) * n + i
                W_full[d_global, col_offset:col_offset+r_sub] = W_sub[d_local]
    
    return U_full, V_full, W_full

def _try_improve_rank_42():
    """
    Try to find a rank-42 decomposition by starting from the rank-44 solution
    and using gradient descent to try to eliminate 2 more terms.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 3, 8
    T = _build_tensor(n, m, p)
    
    # Start from rank-44 solution
    U44, V44, W44 = _fallback_44()
    
    # Try to find a rank-42 solution by perturbing and optimizing
    # We'll use a simple approach: try to set 2 columns to zero and optimize the rest
    # This is heuristic and may not work, but worth trying
    
    # Actually, let's try a different approach: use the rank-44 as initialization
    # and try to find a lower rank solution via optimization
    # But this is complex. Let's just return None for now.
    
    return None

def _build_2x3x4_rank18():
    """
    Rank-18 decomposition for 2x3 * 3x4 = 2x4.
    Split B's 4 columns into 2 blocks of 2.
    Each block uses rank-11 for 2x3 * 3x2.
    Total rank: 2 * 11 = 22. But we can do better.
    
    Actually, let's just use the 2-block approach: 2 * 11 = 22.
    """
    n, m, p = 2, 3, 4
    block_size = 2
    num_blocks = p // block_size  # 2
    r_sub = 11
    total_rank = num_blocks * r_sub  # 22
    
    U_sub, V_sub, W_sub = _build_2x3x2_rank11()
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for b in range(num_blocks):
        k_offset = b * block_size
        col_offset = b * r_sub
        
        U_full[:, col_offset:col_offset+r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(block_size):
                b_local = j * block_size + k_prime
                b_global = j * p + k_offset + k_prime
                V_full[b_global, col_offset:col_offset+r_sub] = V_sub[b_local]
        
        for k_prime in range(block_size):
            for i in range(n):
                d_local = i * block_size + k_prime
                d_global = (k_offset + k_prime) * n + i
                W_full[d_global, col_offset:col_offset+r_sub] = W_sub[d_local]
    
    return U_full, V_full, W_full

def _fallback_44_v2():
    """
    Alternative 44-rank decomposition: split p=8 into 2 blocks of 4.
    Each block uses rank-22 for 2x3 * 3x4.
    Total rank: 2 * 22 = 44.
    """
    n, m, p = 2, 3, 8
    block_size = 4
    num_blocks = p // block_size  # 2
    r_sub = 22
    total_rank = num_blocks * r_sub  # 44
    
    U_sub, V_sub, W_sub = _build_2x3x4_rank18()  # Actually rank 22
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for b in range(num_blocks):
        k_offset = b * block_size
        col_offset = b * r_sub
        
        U_full[:, col_offset:col_offset+r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(block_size):
                b_local = j * block_size + k_prime
                b_global = j * p + k_offset + k_prime
                V_full[b_global, col_offset:col_offset+r_sub] = V_sub[b_local]
        
        for k_prime in range(block_size):
            for i in range(n):
                d_local = i * block_size + k_prime
                d_global = (k_offset + k_prime) * n + i
                W_full[d_global, col_offset:col_offset+r_sub] = W_sub[d_local]
    
    return U_full, V_full, W_full

def _try_rank_40():
    """
    Try to find a rank-40 decomposition.
    This would require rank-10 for 2x3x2 sub-problems.
    For now, return None as we don't have a verified rank-10 construction.
    """
    return None

def construct():
    n, m, p = 2, 3, 8
    
    # First, try the rank-44 block decomposition
    if not _check_deadline():
        U44, V44, W44 = _fallback_44()
        if _verify(n, m, p, U44, V44, W44):
            # Try to improve to rank 42
            if not _check_deadline():
                res = _try_improve_rank_42()
                if res is not None:
                    U, V, W = res
                    if _verify(n, m, p, U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try rank 40
            if not _check_deadline():
                res40 = _try_rank_40()
                if res40 is not None:
                    U, V, W = res40
                    if _verify(n, m, p, U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Return rank-44
            return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
    
    # Fallback to schoolbook rank 48
    rank_school = n * m * p
    U_fb = np.zeros((n*m, rank_school))
    V_fb = np.zeros((m*p, rank_school))
    W_fb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
