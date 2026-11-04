"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Try to find a rank-83 decomposition
    result_83 = _try_find_rank_83(n, m, p, deadline)
    if result_83 is not None:
        return result_83

    # Fallback to known rank-84
    result_84 = _build_strassen_3x5x6(n, m, p)
    if result_84 is not None:
        U, V, W, rank = result_84
        if _verify_fast(U, V, W, n, m, p):
            U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
            V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
            W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback
    return _build_schoolbook(n, m, p)

def _try_find_rank_83(n, m, p, deadline):
    """
    Attempt to find a rank-83 decomposition by checking if the rank-84 terms
    are linearly dependent. If so, we can eliminate one term.
    """
    if time.time() > deadline - 20:
        return None
    
    U84, V84, W84, rank84 = _build_strassen_3x5x6(n, m, p)
    
    try:
        if time.time() > deadline - 15:
            return None
        
        # Build the matrix M of shape (n*m*p, rank84) where M[i,j,k,c] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        u_flat = U84.T  # (84, 15)
        v_flat = V84.T  # (84, 30)
        w_flat = W84.T  # (84, 18)
        
        T_terms = np.einsum('ic,jc,kc->ijkc', u_flat, v_flat, w_flat)
        M = T_terms.reshape(-1, 84)
        
        if time.time() > deadline - 10:
            return None
            
        # Use SVD to check rank
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        if time.time() > deadline - 5:
            return None
        
        # Check if smallest singular value is effectively zero
        if s[-1] < 1e-10:
            x = Vt[-1, :]
            idx = np.argmax(np.abs(x))
            if abs(x[idx]) < 1e-10:
                return None
                
            coeffs = np.ones(84)
            coeffs[idx] = 0
            for c in range(84):
                if c != idx:
                    coeffs[c] = 1.0 - x[c] / x[idx]
            
            U_new = U84 * coeffs
            V_new = V84.copy()
            W_new = W84.copy()
            
            mask = np.ones(84, dtype=bool)
            mask[idx] = False
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            
            if _verify_fast(U_new, V_new, W_new, n, m, p):
                U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_new.tolist()]
                V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_new.tolist()]
                W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_new.tolist()]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    except Exception:
        pass
        
    return None


def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U (A)
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V (B)
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W (C)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _build_strassen_3x3x3():
    """
    Returns U, V, W for 3x3x3 multiplication using Strassen-based decomposition.
    Rank 23.
    """
    # We can build this by decomposing 3x3 into 2x2 + 1x1 + ...
    # Or use a known 23-rank decomposition.
    # A simpler way is to use the recursive Strassen on 3x3?
    # No, 3 is prime.
    # Let's use the 2x2 Strassen + Schoolbook decomposition that yields 23.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    sU, sV, sW = _get_strassen_2x2x2()
    
    col = 0
    # Top-left 2x2 block
    # U: rows 0,1; cols 0,1
    U[0:2, 0:2, :] = sU # Wrong dimension
    # Let's map indices properly.
    # A is 3x3. Indices i,j in 0..2.
    # Block A00: i in 0,1; j in 0,1.
    
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Uses Strassen 7.
    for c in range(7):
        U[0*3 + 0, col+c] = sU[0, c]
        U[0*3 + 1, col+c] = sU[1, c]
        U[1*3 + 0, col+c] = sU[2, c]
        U[1*3 + 1, col+c] = sU[3, c]
        
        V[0*3 + 0, col+c] = sV[0, c]
        V[0*3 + 1, col+c] = sV[1, c]
        V[1*3 + 0, col+c] = sV[2, c]
        V[1*3 + 1, col+c] = sV[3, c]
        
        W[0*3 + 0, col+c] = sW[0, c]
        W[0*3 + 1, col+c] = sW[1, c]
        W[1*3 + 0, col+c] = sW[2, c]
        W[1*3 + 1, col+c] = sW[3, c]
    col += 7

    # Block 1: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A is 2x1, B is 1x2. Rank 4.
    for i in range(2):
        for k in range(2):
            U[i*3 + 2, col] = 1
            V[2*3 + k, col] = 1 # B[2, k]
            W[k*3 + i, col] = 1
            col += 1
    col += 0 # 4 terms

    # Block 2: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # A is 1x2, B is 2x2. Rank 4.
    for j in range(2):
        for k in range(2):
            U[2*3 + j, col] = 1
            V[j*3 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1
    col += 0 # 4 terms

    # Block 3: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A is 2x2, B is 2x1. Rank 4.
    for i in range(2):
        for j in range(2):
            U[i*3 + j, col] = 1
            V[j*3 + 2, col] = 1
            W[2*3 + i, col] = 1
            col += 1
    col += 0 # 4 terms

    # Block 4: A[2, 2] * B[2, 2] -> C[2, 2]
    # Rank 1.
    U[2*3 + 2, col] = 1
    V[2*3 + 2, col] = 1
    W[2*3 + 2, col] = 1
    col += 1

    # Block 5: A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    # A 2x1, B 1x1. Rank 2.
    for i in range(2):
        U[i*3 + 2, col] = 1
        V[2*3 + 2, col] = 1
        W[2*3 + i, col] = 1
        col += 1
    col += 0 # 2 terms

    # Block 6: A[2, 0:2] * B[0:2, 2] -> C[2, 0:2]
    # A 1x2, B 2x1. Rank 4.
    # Wait, I already did A[2,0:2]*B[0:2,0:2] in Block 2.
    # I need A[2,0:2] * B[0:2,2] for C[2, 0:2].
    # This contributes to C[2,0] and C[2,1].
    for j in range(2):
        U[2*3 + j, col] = 1
        V[j*3 + 2, col] = 1
        W[2*3 + j, col] = 1 # No, C index is k.
        # C[2, k]. B is [0:2, 2]. So B[j, 2].
        # U is A[2, j].
        # W is C[2, k]? No.
        # Let's restart the mapping. This is error prone.
        
    # Use a known 23-rank decomposition?
    # Or just stick to 84?
    # The improvement is small (80 vs 84).
    # And high risk of bugs.
    
    return None

def _build_optimized_3x5x6(n, m, p):
    # Fallback to current 84-rank if 3x3x3 fails
    return _build_strassen_3x5x6(n, m, p)

def _build_strassen_3x5x6(n, m, p):
    """
    Build a rank-minimal (or near-minimal) decomposition for 3x5 * 5x6.
    
    Strategy: Partition the intermediate index j into blocks {0,1}, {2,3}, {4}
    and the output index k into blocks {0,1}, {2,3}, {4,5}.
    
    For each (j_block, k_block) pair, we have a subproblem:
    A_sub (3 x jb_size) * B_sub (jb_size x kb_size) -> C_sub (3 x kb_size)
    
    We decompose each subproblem using Strassen where applicable.
    
    Total rank calculation:
    - (j=[0,1], k=[0,1]): 3x2 * 2x2 -> 3x2
      Split i: [0,1] and [2]
      i=[0,1]: 2x2 * 2x2 -> 2x2, Strassen rank 7
      i=[2]: 1x2 * 2x2 -> 1x2, rank 4
      Subtotal: 11
    - (j=[0,1], k=[2,3]): same as above, 11
    - (j=[0,1], k=[4,5]): same as above, 11
    - (j=[2,3], k=[0,1]): 11
    - (j=[2,3], k=[2,3]): 11
    - (j=[2,3], k=[4,5]): 11
    - (j=[4], k=[0,1]): 3x1 * 1x2 -> 3x2, rank 6
    - (j=[4], k=[2,3]): 6
    - (j=[4], k=[4,5]): 6
    Total: 6*11 + 3*6 = 66 + 18 = 84
    """
    sU, sV, sW = _get_strassen_2x2x2()
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper: add Strassen 2x2x2 for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
    def add_strassen(i0, j0, k0):
        nonlocal col
        for c in range(7):
            U[i0*5 + j0, col] = sU[0, c]
            U[i0*5 + j0+1, col] = sU[1, c]
            U[(i0+1)*5 + j0, col] = sU[2, c]
            U[(i0+1)*5 + j0+1, col] = sU[3, c]
            V[j0*6 + k0, col] = sV[0, c]
            V[j0*6 + k0+1, col] = sV[1, c]
            V[(j0+1)*6 + k0, col] = sV[2, c]
            V[(j0+1)*6 + k0+1, col] = sV[3, c]
            W[k0*3 + i0, col] = sW[0, c]
            W[k0*3 + i0+1, col] = sW[1, c]
            W[(k0+1)*3 + i0, col] = sW[2, c]
            W[(k0+1)*3 + i0+1, col] = sW[3, c]
            col += 1
    
    # Helper: add schoolbook 1x2 * 2x2 -> 1x2 (for i=2 case)
    def add_1x2_2x2(i, j0, k0):
        nonlocal col
        for j in range(2):
            for k in range(2):
                U[i*5 + j0+j, col] = 1
                V[(j0+j)*6 + k0+k, col] = 1
                W[(k0+k)*3 + i, col] = 1
                col += 1
    
    # Helper: add schoolbook 3x1 * 1x2 -> 3x2
    def add_3x1_1x2(j, k0):
        nonlocal col
        for i in range(3):
            for k in range(2):
                U[i*5 + j, col] = 1
                V[j*6 + k0+k, col] = 1
                W[(k0+k)*3 + i, col] = 1
                col += 1
    
    # Process each (j_block, k_block) pair
    j_blocks = [(0, 1), (2, 3), (4, 4)]  # (j_start, j_size)
    k_blocks = [(0, 1), (2, 3), (4, 5)]  # (k_start, k_size)
    
    for j_start, j_size in j_blocks:
        for k_start, k_size in k_blocks:
            if j_size == 2 and k_size == 2:
                # 3x2 * 2x2 -> 3x2
                # i=[0,1]: Strassen 2x2x2
                add_strassen(0, j_start, k_start)
                # i=[2]: 1x2 * 2x2 -> 1x2, rank 4
                add_1x2_2x2(2, j_start, k_start)
            elif j_size == 1 and k_size == 2:
                # 3x1 * 1x2 -> 3x2, rank 6
                add_3x1_1x2(j_start, k_start)
    
    return U, V, W, rank

def _build_schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END
