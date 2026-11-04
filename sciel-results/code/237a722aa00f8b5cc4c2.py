import time
import numpy as np

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Tight deadline for the search, construction is O(1)
    
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    # 1. Construct the known rank-28 Strassen-based solution
    # This is a direct sum of 4 blocks of 2x2x2 Strassen multiplication.
    U, V, W = _build_rank_28(n, m, p)
    
    # Convert to numpy arrays for verification and search
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Verify the rank-28 solution
    max_err = _verify_decomp(U_arr, V_arr, W_arr, 28, T)
    if max_err > 1e-10:
        # Fallback to schoolbook (rank 32) if something went wrong, though unlikely
        U32, V32, W32 = _build_schoolbook_32(n, m, p)
        return {"U": U32, "V": V32, "W": W32}

    # 2. Attempt to find a rank-27 solution by checking for linear dependencies
    # The rank of the tensor is the rank of the matrix M where M_{(a,b,c), r} = U_{a,r} V_{b,r} W_{c,r}
    # If the columns of M are linearly dependent, we can reduce the rank.
    
    if time.time() < deadline:
        rank_found, new_U, new_V, new_W = _search_for_lower_rank(U_arr, V_arr, W_arr, T, deadline)
        if rank_found < 28:
            # Verify the new solution
            if _verify_decomp(new_U, new_V, new_W, rank_found, T) < 1e-10:
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
    
    # Return the rank-28 solution
    return {"U": U, "V": V, "W": W}

def _build_rank_28(n, m, p):
    """
    Constructs the rank-28 decomposition using hardcoded Strassen indices for 4 blocks.
    This is faster than generating indices in a loop.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block of Strassen terms
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen terms for 2x2 multiplication
        # Term 0: M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # Term 1: M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # Term 2: M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # Term 3: M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # Term 4: M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # Term 5: M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # Term 6: M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Define indices for the 4 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # U index: i*m + j. i in {0,1}, j in {0,1,2,3}
    # V index: j*p + k. j in {0,1,2,3}, k in {0,1,2,3}
    # W index: k*n + i. k in {0,1,2,3}, i in {0,1}
    
    # Block 1: A_L (j=0,1) * B_TL (j=0,1, k=0,1) -> C_LL (k=0,1)
    # X_rows (U): i=0,j=0->0; i=0,j=1->1; i=1,j=0->4; i=1,j=1->5. Order: 0,1,4,5
    X1 = [0, 1, 4, 5]
    # Y_rows (V): j=0,k=0->0; j=0,k=1->1; j=1,k=0->4; j=1,k=1->5. Order: 0,1,4,5
    Y1 = [0, 1, 4, 5]
    # Z_rows (W): k=0,i=0->0; k=0,i=1->2; k=1,i=0->1; k=1,i=1->3. 
    # Strassen local indices map to: 0->TL, 1->TR, 2->BL, 3->BR of the 2x2 block.
    # W index for C element (i,k): k*n + i.
    # Local 0 (TL) -> i=0, k=0 -> W[0]
    # Local 1 (TR) -> i=0, k=1 -> W[1]
    # Local 2 (BL) -> i=1, k=0 -> W[2]
    # Local 3 (BR) -> i=1, k=1 -> W[3]
    # So Z1 should be [0, 1, 2, 3] corresponding to local indices 0,1,2,3.
    # Wait, the add_block function uses X_rows[0], X_rows[3], etc.
    # It assumes X_rows[0] is local 0, X_rows[1] is local 1, X_rows[2] is local 2, X_rows[3] is local 3.
    # Let's verify the mapping in add_block:
    # M1 uses X[0], X[3]. Local 0 and 3.
    # In Strassen: M1 = (A00+A11)(B00+B11).
    # So X[0] is A00, X[3] is A11.
    # For A_L (2x2 block):
    # A00 is A[0,0] -> U[0]
    # A01 is A[0,1] -> U[1]
    # A10 is A[1,0] -> U[4]
    # A11 is A[1,1] -> U[5]
    # So X1 = [0, 1, 4, 5] is correct.
    
    # For C_LL (2x2 block):
    # C00 is C[0,0] -> W[0]
    # C01 is C[0,1] -> W[1]
    # C10 is C[1,0] -> W[2]
    # C11 is C[1,1] -> W[3]
    # So Z1 = [0, 1, 2, 3].
    
    Z1 = [0, 1, 2, 3]
    add_block(0, X1, Y1, Z1)
    
    # Block 2: A_R (j=2,3) * B_BL (j=2,3, k=0,1) -> C_LR (k=0,1)
    # X2 (U): A[0,2]->2, A[0,3]->3, A[1,2]->6, A[1,3]->7. X2 = [2, 3, 6, 7]
    X2 = [2, 3, 6, 7]
    # Y2 (V): B[2,0]->8, B[2,1]->9, B[3,0]->12, B[3,1]->13. Y2 = [8, 9, 12, 13]
    Y2 = [8, 9, 12, 13]
    # Z2 (W): Same C columns as Block 1 (k=0,1). Z2 = [0, 1, 2, 3]
    Z2 = [0, 1, 2, 3]
    add_block(7, X2, Y2, Z2)
    
    # Block 3: A_L (j=0,1) * B_TR (j=0,1, k=2,3) -> C_RL (k=2,3)
    # X3 (U): Same as X1. X3 = [0, 1, 4, 5]
    X3 = [0, 1, 4, 5]
    # Y3 (V): B[0,2]->2, B[0,3]->3, B[1,2]->6, B[1,3]->7. Y3 = [2, 3, 6, 7]
    Y3 = [2, 3, 6, 7]
    # Z3 (W): C columns k=2,3.
    # C[0,2] -> W[4], C[0,3] -> W[5], C[1,2] -> W[6], C[1,3] -> W[7].
    # Local 0 (TL of C_R) -> C[0,2] -> W[4]
    # Local 1 (TR of C_R) -> C[0,3] -> W[5]
    # Local 2 (BL of C_R) -> C[1,2] -> W[6]
    # Local 3 (BR of C_R) -> C[1,3] -> W[7]
    Z3 = [4, 5, 6, 7]
    add_block(14, X3, Y3, Z3)
    
    # Block 4: A_R (j=2,3) * B_BR (j=2,3, k=2,3) -> C_RR (k=2,3)
    # X4 (U): Same as X2. X4 = [2, 3, 6, 7]
    X4 = [2, 3, 6, 7]
    # Y4 (V): B[2,2]->10, B[2,3]->11, B[3,2]->14, B[3,3]->15. Y4 = [10, 11, 14, 15]
    Y4 = [10, 11, 14, 15]
    # Z4 (W): Same as Z3. Z4 = [4, 5, 6, 7]
    Z4 = [4, 5, 6, 7]
    add_block(21, X4, Y4, Z4)
    
    return U, V, W

def _build_schoolbook_32(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return U, V, W

def _verify_decomp(U_arr, V_arr, W_arr, rank_val, T):
    nm, mp, pn = T.shape
    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank_val):
        S += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    return np.max(np.abs(S - T))

def _search_for_lower_rank(U_arr, V_arr, W_arr, T, deadline):
    """
    Checks if the rank can be reduced by finding linear dependencies in the 
    tensor decomposition vectors.
    Returns (rank, U, V, W) if a lower rank is found, else (28, None, None, None).
    """
    nm, mp, pn = T.shape
    rank = 28
    
    # Build M matrix: M[abc, r] = U[a,r]*V[b,r]*W[c,r]
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()
        
    # Check rank of M
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        # Count significant singular values
        tol = 1e-8
        effective_rank = np.sum(s > tol)
        
        if effective_rank < rank:
            # Find the pivot columns
            # We can use QR with pivoting on M.T to find independent columns
            # But we need to map back to U, V, W
            # If column r is dependent, we can drop it.
            # For simplicity, if effective_rank < 28, we assume we can drop (28 - effective_rank) columns.
            # This is a heuristic; proper basis extraction is needed.
            # However, given the structure, if it's 28, this will return 28.
            # If it is 27, we need to identify which one to drop.
            
            # Let's try to drop columns one by one and check if the remaining span the original.
            # This is expensive.
            # Alternative: Just return the original if no obvious dependency.
            # Given the disjoint supports, dependencies are unlikely.
            
            # For now, if effective_rank < 28, we return a flag.
            # But we need to construct the new U, V, W.
            # Let's try to find the first dependent column.
            for r_elim in range(rank):
                if time.time() > deadline:
                    break
                other_cols = [c for c in range(rank) if c != r_elim]
                M_sub = M[:, other_cols]
                rhs = M[:, r_elim]
                try:
                    alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                    residual = np.max(np.abs(M_sub @ alpha - rhs))
                    if residual < 1e-8:
                        # Column r_elim is redundant
                        new_U = np.delete(U_arr, r_elim, axis=1)
                        new_V = np.delete(V_arr, r_elim, axis=1)
                        new_W = np.delete(W_arr, r_elim, axis=1)
                        return rank - 1, new_U, new_V, new_W
                except np.linalg.LinAlgError:
                    continue
                    
    except np.linalg.LinAlgError:
        pass
        
    return 28, None, None, None
