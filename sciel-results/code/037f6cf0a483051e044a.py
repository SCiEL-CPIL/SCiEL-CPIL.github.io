"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round a matrix to half-integers."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _als_decompose(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """Alternating least squares to refine U, V, W."""
    prev_err = float('inf')
    for it in range(max_iters):
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, V, optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
    n = 2
    m = 2
    p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U indices: i*m+j
    # V indices: j*p+k
    # W indices: k*n+i
    
    # Term 0: M1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = -1
    W[1*2+0, 0] = -1; W[1*2+1, 0] = 1
    
    # Term 1: M2
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    
    # Term 2: M3
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+0, 2] = 1
    
    # Term 3: M4
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[1*2+1, 3] = 1
    
    # Term 4: M5
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[1*2+0, 4] = 1
    
    # Term 5: M6
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[0*2+1, 5] = 1
    
    # Term 6: M7
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[1*2+1, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using Strassen blocks (rank 49)."""
    n = 4
    m = 4
    p = 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    # Map global indices to local block indices
    # Global i = 2*I + i_loc
    # Global j = 2*J + j_loc
    # Global k = 2*K + k_loc
    
    for I in range(2):
        for J in range(2):
            for i_loc in range(2):
                for j_loc in range(2):
                    i_glob = 2*I + i_loc
                    j_glob = 2*J + j_loc
                    idx_u = i_glob * 4 + j_glob
                    
                    for r_str in range(7):
                        # Determine if this block (I,J) is involved in Strassen term r_str
                        # We need to check the coefficients in U2
                        # U2[i_loc*2+j_loc, r_str]
                        val_u = U2[i_loc*2+j_loc, r_str]
                        if val_u != 0:
                            for r_inner in range(7):
                                r_glob = r_str * 7 + r_inner
                                U[idx_u, r_glob] = val_u * U2[i_loc*2+j_loc, r_inner] # Wait, U2 is for 2x2
                                # Actually, U2[i_loc*2+j_loc, r_inner] is the coefficient for the inner 2x2 multiplication
                                # But we are using the SAME Strassen structure for the inner blocks.
                                # So U[idx_u, r_glob] = U2[i_loc*2+j_loc, r_inner] is correct?
                                # No, U2 is the U for the 2x2 multiplication.
                                # The inner multiplication is also 2x2.
                                # So yes, we reuse U2.
                                # But we already set val_u = U2[i_loc*2+j_loc, r_str].
                                # And we are iterating r_inner.
                                # This logic is flawed.
                                
                                # Let's restart the logic.
                                # The global tensor is a Kronecker product of the block Strassen tensor and the inner Strassen tensor.
                                # T_global = T_block \otimes T_inner
                                # U_global = U_block \otimes U_inner
                                # V_global = V_block \otimes V_inner
                                # W_global = W_block \otimes W_inner
                                
                                # U_block indices: (I, J). U_block[I*2+J, r_str]
                                # U_inner indices: (i_loc, j_loc). U_inner[i_loc*2+j_loc, r_inner]
                                # U_global indices: (I*2+i_loc, J*2+j_loc).
                                # U_global[(I*2+i_loc)*4 + (J*2+j_loc), r_str*7 + r_inner] = U_block[I*2+J, r_str] * U_inner[i_loc*2+j_loc, r_inner]
                                
                                pass
    
    # Let's implement the Kronecker product properly.
    U_block, V_block, W_block = _strassen_2x2()
    U_inner, V_inner, W_inner = _strassen_2x2()
    
    U = np.kron(U_block, U_inner)
    V = np.kron(V_block, V_inner)
    W = np.kron(W_block, W_inner)
    
    # Check shapes
    # U_block: (4, 7), U_inner: (4, 7) -> U: (16, 49)
    # V_block: (4, 7), V_inner: (4, 7) -> V: (16, 49)
    # W_block: (4, 7), W_inner: (4, 7) -> W: (16, 49)
    
    # The indices in the task are:
    # U: i*m+j
    # V: j*p+k
    # W: k*n+i
    
    # The Kronecker product ordering must match.
    # np.kron(A, B) where A is (4,7) and B is (4,7).
    # Result (16, 49).
    # Row index: i_block * 4 + i_inner.
    # i_block = 2*I + i_loc? No, i_block is 0..3.
    # In 2x2 Strassen, i is 0..1.
    # So i_block corresponds to the block index I?
    # Yes, if we define the 2x2 block matrix indices as I, J.
    # Then the global row index is I*2 + i_loc.
    # np.kron(A, B) row index is i_A * n_B + i_B.
    # Here n_B = 4 (size of B's first dim).
    # So row index = i_block * 4 + i_inner.
    # i_block is 0..3? No, U_block has 4 rows (2x2 matrix entries).
    # So i_block is the index of the entry in the 2x2 block matrix.
    # i_block = I*2 + i_loc? No, I is 0..1.
    # The 2x2 block matrix has entries (0,0), (0,1), (1,0), (1,1).
    # Indices 0, 1, 2, 3.
    # So i_block = I*2 + J? No, I is row, J is col.
    # In U, index is i*m+j.
    # So i_block = I*2 + J? No, I is the block row.
    # Let's assume the 2x2 block matrix indices are flattened as row-major.
    # Then i_block = I*2 + J? No, I is 0..1, J is 0..1.
    # Flattened index = I*2 + J.
    # So i_block = I*2 + J.
    # Then global index = i_block * 4 + i_inner.
    # i_inner = i_loc * 2 + j_loc.
    # So global index = (I*2 + J) * 4 + (i_loc * 2 + j_loc).
    # = (2I+J)*4 + 2i_loc + j_loc.
    # = 8I + 4J + 2i_loc + j_loc.
    
    # Task index: i_glob * 4 + j_glob.
    # i_glob = 2I + i_loc.
    # j_glob = 2J + j_loc.
    # Task index = (2I + i_loc)*4 + (2J + j_loc) = 8I + 4i_loc + 2J + j_loc.
    
    # Mismatch!
    # np.kron gives 8I + 4J + 2i_loc + j_loc.
    # Task wants 8I + 4i_loc + 2J + j_loc.
    
    # We need to permute the rows of U and V.
    # U rows correspond to (I, J, i_loc, j_loc).
    # We need to reorder them to (I, i_loc, J, j_loc).
    
    # Let's construct U manually to avoid permutation errors.
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for I in range(2):
        for J in range(2):
            for r_str in range(7):
                u_block_val = U_block[I*2+J, r_str]
                if u_block_val == 0: continue
                
                for i_loc in range(2):
                    for j_loc in range(2):
                        u_inner_val = U_inner[i_loc*2+j_loc, r_str] # Wait, r_inner?
                        # No, U_inner is for the inner 2x2 multiplication.
                        # The inner multiplication uses the same Strassen structure.
                        # So we iterate r_inner.
                        pass
                        
    # Correct Logic:
    # U_global[I, J, i_loc, j_loc, r_str, r_inner] = U_block[I, J, r_str] * U_inner[i_loc, j_loc, r_inner]
    
    for I in range(2):
        for J in range(2):
            for i_loc in range(2):
                for j_loc in range(2):
                    i_glob = 2*I + i_loc
                    j_glob = 2*J + j_loc
                    idx_u = i_glob * 4 + j_glob
                    
                    for r_str in range(7):
                        u_block_val = U_block[I*2+J, r_str]
                        if u_block_val == 0: continue
                        
                        for r_inner in range(7):
                            u_inner_val = U_inner[i_loc*2+j_loc, r_inner]
                            if u_inner_val == 0: continue
                            
                            r_glob = r_str * 7 + r_inner
                            U[idx_u, r_glob] = u_block_val * u_inner_val
                            
    # Similarly for V
    for J in range(2):
        for K in range(2):
            for j_loc in range(2):
                for k_loc in range(2):
                    j_glob = 2*J + j_loc
                    k_glob = 2*K + k_loc
                    idx_v = j_glob * 4 + k_glob
                    
                    for r_str in range(7):
                        v_block_val = V_block[J*2+K, r_str]
                        if v_block_val == 0: continue
                        
                        for r_inner in range(7):
                            v_inner_val = V_inner[j_loc*2+k_loc, r_inner]
                            if v_inner_val == 0: continue
                            
                            r_glob = r_str * 7 + r_inner
                            V[idx_v, r_glob] = v_block_val * v_inner_val
                            
    # Similarly for W
    for K in range(2):
        for I in range(2):
            for k_loc in range(2):
                for i_loc in range(2):
                    k_glob = 2*K + k_loc
                    i_glob = 2*I + i_loc
                    idx_w = k_glob * 4 + i_glob
                    
                    for r_str in range(7):
                        w_block_val = W_block[K*2+I, r_str]
                        if w_block_val == 0: continue
                        
                        for r_inner in range(7):
                            w_inner_val = W_inner[k_loc*2+i_loc, r_inner]
                            if w_inner_val == 0: continue
                            
                            r_glob = r_str * 7 + r_inner
                            W[idx_w, r_glob] = w_block_val * w_inner_val
                            
    return U, V, W

def _build_4x4x1():
    """Build 4x4x1 decomposition (rank 16)."""
    n = 4
    m = 4
    p = 1
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(4):
        for j in range(4):
            r = i*4 + j
            U[i*4+j, r] = 1
            V[j*1+0, r] = 1
            W[0*4+i, r] = 1
            
    return U, V, W

def _build_4x4x5():
    """Combine 4x4x4 and 4x4x1 to get 4x4x5 (rank 65)."""
    n = 4
    m = 4
    p = 5
    rank = 65
    
    U1, V1, W1 = _build_4x4x4_strassen()
    U2, V2, W2 = _build_4x4x1()
    
    # Pad U2, V2, W2 to p=5
    U = np.zeros((16, 65))
    V = np.zeros((20, 65))
    W = np.zeros((20, 65))
    
    # First 49 columns
    U[:, :49] = U1
    V[:16, :49] = V1
    W[:16, :49] = W1
    
    # Last 16 columns
    U[:, 49:] = U2
    V[16:20, 49:] = V2 # V2 is 4x16, map to j=0..3, k=4
    # V index is j*p+k. j=0..3, k=4.
    # V2 indices are j*1+0.
    # So V2[j, r] -> V[j*5+4, r]
    for j in range(4):
        for r in range(16):
            V[j*5+4, 49+r] = V2[j, r]
            
    # W2 indices are k*4+i. k=0.
    # W2[i, r] -> W[0*4+i, r] = W[i, r]
    W[:4, 49:] = W2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build rank 65 solution
    U, V, W = _build_4x4x5()
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = 65
    
    # Try to reduce rank using ALS
    for target_rank in range(64, 48, -1):
        if time.time() > deadline - 30:
            break
            
        # Initialize with the 65 rank solution, but truncated/randomized
        U_init = np.random.randn(16, target_rank) * 0.1
        V_init = np.random.randn(20, target_rank) * 0.1
        W_init = np.random.randn(20, target_rank) * 0.1
        
        # Or use the first target_rank columns of the 65 rank solution?
        # That might be better.
        U_init = U[:, :target_rank]
        V_init = V[:, :target_rank]
        W_init = W[:, :target_rank]
        
        U_ref, V_ref, W_ref = _als_decompose(U_init, V_init, W_init, n, m, p, target_rank, max_iters=50, tol=1e-12)
        
        U_rounded = _round_matrix_to_half_int(U_ref)
        V_rounded = _round_matrix_to_half_int(V_ref)
        W_rounded = _round_matrix_to_half_int(W_ref)
        
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            best_rank = target_rank
            break
            
    return best_result
# EVOLVE-BLOCK-END
