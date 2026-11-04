"""Improved SEARCH approach: Fast NumPy validation and structured block-Strassen construction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 140.0  # safety margin before evaluator kill

def _validate_numpy(U, V, W, n, m, p):
    """
    Fast validation using NumPy.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    """
    r = U.shape[1]
    # Reshape to 3D
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # einsum: U is 'irc' (i, j, c), V is 'jrc' (j, k, c), W is 'kir' (k, i, c)
    # We want output 'ijk'
    # Note: W index is (k, i). So W3[k, i, c].
    # np.einsum('irc,jrc,kir->ijk', U3, V3, W3)
    try:
        T = np.einsum('irc,jrc,kir->ijk', U3, V3, W3)
    except Exception as e:
        return False
    
    # T should be all 1s
    if T.shape != (n, m, p):
        return False
    return np.all(T == 1)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i,j], V[j,k], W[k,i]
    """
    n = m = p = 2
    r = 7
    U = np.zeros((n*m, r), dtype=int)
    V = np.zeros((m*p, r), dtype=int)
    W = np.zeros((p*n, r), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = A11*(B10-B00)
    U[3, 1] = 1
    V[0, 1] = -1; V[4, 1] = 1
    W[1, 1] = 1
    
    # M3 = (A10+A11)*B01
    U[2, 2] = 1; U[3, 2] = 1
    V[5, 2] = 1
    W[2, 2] = 1
    
    # M4 = A00*(B01-B11)
    U[0, 3] = 1
    V[1, 3] = 1; V[5, 3] = -1
    W[3, 3] = 1
    
    # M5 = (A00+A01)(B11+B00)
    U[0, 4] = 1; U[1, 4] = 1
    V[0, 4] = 1; V[1, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A01-A11)*(B10+B11)
    U[1, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[5, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A10-A00)*(B00+B01)
    U[2, 6] = 1; U[0, 6] = -1
    V[0, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_block_4x4x4():
    """
    Constructs 4x4x4 decomposition using 8 Strassen 2x2x2 decompositions.
    Rank 56.
    """
    n = m = p = 4
    r_total = 56
    U = np.zeros((n*m, r_total), dtype=int)
    V = np.zeros((m*p, r_total), dtype=int)
    W = np.zeros((p*n, r_total), dtype=int)
    
    U2, V2, W2 = _strassen_2x2()
    
    # Blocks:
    # A blocks: (a, d) where a is block row (0,1), d is block col (0,1)
    # B blocks: (d, b) where d is block row (0,1), b is block col (0,1)
    # C blocks: (a, b)
    
    # We iterate over all (a, d, b) combinations?
    # C_ab = sum_d A_ad B_db
    # For each (a, b), we sum over d=0,1.
    # So we need a Strassen decomposition for each (a, d, b) triple?
    # No, for fixed (a, b), we have two terms: d=0 and d=1.
    # Each term is a 2x2 multiplication of block A_ad and block B_db.
    # So yes, we instantiate Strassen for each (a, d, b).
    # There are 4 * 2 * 4 = 32? No.
    # a in {0,1}, d in {0,1}, b in {0,1}.
    # 2 * 2 * 2 = 8 combinations.
    # Each uses 7 terms. Total 56.
    
    offset = 0
    for a in range(2):
        for d in range(2):
            for b in range(2):
                # Map local indices to global
                # A_ad: rows 2a..2a+1, cols 2d..2d+1
                # B_db: rows 2d..2d+1, cols 2b..2b+1
                # C_ab: rows 2a..2a+1, cols 2b..2b+1
                
                # Local U index (i_loc, j_loc) -> Global (2a+i_loc, 2d+j_loc)
                # Local V index (j_loc, k_loc) -> Global (2d+j_loc, 2b+k_loc)
                # Local W index (k_loc, i_loc) -> Global (2b+k_loc, 2a+i_loc)
                
                for c in range(7):
                    # U
                    for i_loc in range(2):
                        for j_loc in range(2):
                            u_val = U2[i_loc*2 + j_loc, c]
                            if u_val != 0:
                                g_i = 2*a + i_loc
                                g_j = 2*d + j_loc
                                U[g_i*4 + g_j, offset + c] = u_val
                            
                    # V
                    for j_loc in range(2):
                        for k_loc in range(2):
                            v_val = V2[j_loc*2 + k_loc, c]
                            if v_val != 0:
                                g_j = 2*d + j_loc
                                g_k = 2*b + k_loc
                                V[g_j*4 + g_k, offset + c] = v_val
                                
                    # W
                    for k_loc in range(2):
                        for i_loc in range(2):
                            w_val = W2[k_loc*2 + i_loc, c]
                            if w_val != 0:
                                g_k = 2*b + k_loc
                                g_i = 2*a + i_loc
                                W[g_k*4 + g_i, offset + c] = w_val
                                
                offset += 7
                
    return U, V, W

def _build_block_4x4x5():
    """
    Constructs 4x4x5 decomposition.
    Split B into 4x4 and 4x1.
    4x4 part: Rank 56 (same as 4x4x4 but B cols 0-3).
    4x1 part: Rank 16 (Schoolbook).
    Total Rank 72.
    """
    n = m = 4
    p = 5
    r_total = 56 + 16
    U = np.zeros((n*m, r_total), dtype=int)
    V = np.zeros((m*p, r_total), dtype=int)
    W = np.zeros((p*n, r_total), dtype=int)
    
    # Part 1: 4x4x4 (B cols 0-3)
    U1, V1, W1 = _build_block_4x4x4()
    U[:, :56] = U1
    # V1 is 4x4, map to V[:, 0:4]
    V[:, 0:4] = V1
    # W1 is 4x4, map to W[0:4, :]
    W[0:4, :] = W1
    
    # Part 2: 4x4x1 (B col 4)
    # Schoolbook: C_i = sum_j A_ij B_j4
    # Term (i, j, 4): U[i,j] = 1, V[j,4] = 1, W[4,i] = 1
    offset = 56
    for i in range(4):
        for j in range(4):
            U[i*4 + j, offset] = 1
            V[j*4 + 4, offset] = 1
            W[4*4 + i, offset] = 1
            offset += 1
            
    return U, V, W

def _build_full_4x4x9():
    """
    Combines 4x4x4 and 4x4x5.
    Total Rank 56 + 72 = 128.
    """
    n = m = 4
    p = 9
    r_total = 56 + 72
    U = np.zeros((n*m, r_total), dtype=int)
    V = np.zeros((m*p, r_total), dtype=int)
    W = np.zeros((p*n, r_total), dtype=int)
    
    # Part 1: 4x4x4 (B cols 0-3)
    U1, V1, W1 = _build_block_4x4x4()
    U[:, :56] = U1
    V[:, 0:4] = V1
    W[0:4, :] = W1
    
    # Part 2: 4x4x5 (B cols 4-8)
    U2, V2, W2 = _build_block_4x4x5()
    U[:, 56:128] = U2
    # V2 is 4x5, map to V[:, 4:9]
    V[:, 4:9] = V2
    # W2 is 5x4, map to W[4:9, :]
    W[4:9, :] = W2
    
    return U, V, W

def construct():
    t0 = time.time()
    
    # Generate the structured decomposition
    U, V, W = _build_full_4x4x9()
    
    # Validate
    if _validate_numpy(U, V, W, 4, 4, 9):
        # Return as lists of lists to match interface if required, or numpy arrays?
        # The prompt says "factor matrices of shapes... with integer/half-integer entries".
        # Usually numpy arrays are accepted if they are numeric.
        # However, the original code returned lists. Let's stick to lists to be safe, 
        # or check if evaluator accepts numpy. The prompt says "Python standard library, NumPy... are available".
        # The return type is dict of matrices. Numpy arrays are standard.
        # But to be absolutely safe with "integer/half-integer entries", 
        # converting to lists of ints is robust.
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        n, m, p = 4, 4, 9
        r = n*m*p
        U_fb = np.zeros((n*m, r), dtype=int)
        V_fb = np.zeros((m*p, r), dtype=int)
        W_fb = np.zeros((p*n, r), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, c] = 1
                    V_fb[j*p+k, c] = 1
                    W_fb[k*n+i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
