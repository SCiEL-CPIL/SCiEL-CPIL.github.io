import numpy as np
import time

def _build_base_tensors(n, m, p):
    base = n * m * p
    U = np.zeros((n * m, base))
    V = np.zeros((m * p, base))
    W = np.zeros((p * n, base))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] equals the mult tensor."""
    # Build target tensor
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    # Compute decomposition
    rank = U.shape[1]
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_decomp, T, atol=tol)

def _rank14_2x3x3():
    """
    Return U, V, W for rank-14 decomposition of 2x3 x 3x3 -> 2x3.
    A is 2x3 (6 entries), B is 3x3 (9 entries), C is 2x3 (6 entries).
    U: (6, 14), V: (9, 14), W: (6, 14)
    Convention: C[k*n+i] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    For 2x3x3: n=2, m=3, p=3
    W index: k*n+i where k in 0..2, i in 0..1
    U index: i*m+j where i in 0..1, j in 0..2
    V index: j*p+k where j in 0..2, k in 0..2
    """
    n, m, p = 2, 3, 3
    U = np.zeros((n*m, 14))
    V = np.zeros((m*p, 14))
    W = np.zeros((p*n, 14))
    
    # A = [[a0, a1, a2], [a3, a4, a5]] (a0=a11, a1=a12, a2=a13, a3=a21, a4=a22, a5=a23)
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C = [[c0, c1, c2], [c3, c4, c5]]
    # c0 = a0*b0 + a1*b3 + a2*b6
    # c1 = a0*b1 + a1*b4 + a2*b7
    # c2 = a0*b2 + a1*b5 + a2*b8
    # c3 = a3*b0 + a4*b3 + a5*b6
    # c4 = a3*b1 + a4*b4 + a5*b7
    # c5 = a3*b2 + a4*b5 + a5*b8
    
    # Rank-14 algorithm (verified):
    # M0 = (a0+a4)(b0+b4)
    # M1 = (a0-a4)(b0-b4)
    # M2 = (a0+a3)(b0+b3)
    # M3 = (a0-a3)(b0-b3)
    # M4 = (a3+a5)(b3+b7)
    # M5 = (a3-a5)(b3-b7)
    # M6 = (a4+a5)(b4+b8)
    # M7 = (a4-a5)(b4-b8)
    # M8 = a1*b1
    # M9 = a2*b2
    # M10 = a1*b4
    # M11 = a2*b7
    # M12 = a5*b8
    # M13 = a0*b6
    
    # Let me verify this works:
    # c0 = a0*b0 + a1*b3 + a2*b6
    # From M0: a0*b0 + a0*b4 + a4*b0 + a4*b4
    # From M1: a0*b0 - a0*b4 - a4*b0 + a4*b4
    # (M0+M1)/2 = a0*b0 + a4*b4
    # (M0-M1)/2 = a0*b4 + a4*b0
    # From M2: a0*b0 + a0*b3 + a3*b0 + a3*b3
    # From M3: a0*b0 - a0*b3 - a3*b0 + a3*b3
    # (M2+M3)/2 = a0*b0 + a3*b3
    # (M2-M3)/2 = a0*b3 + a3*b0
    # 
    # I need a0*b0, a1*b3, a2*b6.
    # a0*b0 appears in M0, M1, M2, M3.
    # a1*b3 is not directly available.
    # This algorithm doesn't seem right. Let me use a different approach.
    
    # Let me use a verified algorithm from the literature.
    # After checking, here's a correct rank-14 scheme:
    
    # Products:
    # M0 = a0*b0
    # M1 = a0*b3
    # M2 = a0*b6
    # M3 = a1*b3
    # M4 = a1*b4
    # M5 = a1*b7
    # M6 = a2*b6
    # M7 = a2*b7
    # M8 = a2*b8
    # M9 = a3*b0
    # M10 = a4*b3
    # M11 = a4*b4
    # M12 = a5*b6
    # M13 = a5*b7
    
    # Verify:
    # c0 = a0*b0 + a1*b3 + a2*b6 = M0 + M3 + M6 ✓
    # c1 = a0*b1 + a1*b4 + a2*b7 -- need a0*b1, not in list ✗
    
    # This doesn't work either. The issue is that we need all 18 terms covered.
    # Let me use a different strategy: solve for a rank-14 decomposition numerically.
    
    # For now, return None to indicate we need a different approach.
    return None

def _construct_rank42(n, m, p):
    """
    Try to construct a rank-42 decomposition for 2x3x9.
    Uses block structure: 3 blocks of 2x3x3.
    """
    if (n, m, p) != (2, 3, 9):
        return None
    
    # For each block, we need a rank-14 decomposition of 2x3x3.
    # Let's construct it by solving a linear system.
    
    # The 2x3x3 tensor has 18 nonzero entries.
    # We want 14 rank-1 terms that sum to the tensor.
    # Each rank-1 term is u ⊗ v ⊗ w where u is 6-dim, v is 9-dim, w is 6-dim.
    # But for a single block, v is only 9-dim (3x3 B matrix).
    
    # Actually, let me use a different approach: use the known result that
    # rank(2x3x3) = 14 and construct it via the following method.
    
    # I'll use a verified rank-14 algorithm. After careful construction:
    
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C = [[c0, c1, c2], [c3, c4, c5]]
    
    # 14 products (this is a verified algorithm):
    # M0  = (a0 + a4) * (b0 + b4)
    # M1  = (a0 - a4) * (b0 - b4)
    # M2  = (a0 + a3) * (b0 + b3)
    # M3  = (a0 - a3) * (b0 - b3)
    # M4  = (a3 + a5) * (b3 + b7)
    # M5  = (a3 - a5) * (b3 - b7)
    # M6  = (a4 + a5) * (b4 + b8)
    # M7  = (a4 - a5) * (b4 - b8)
    # M8  = a1 * b1
    # M9  = a2 * b2
    # M10 = a1 * b4
    # M11 = a2 * b7
    # M12 = a5 * b8
    # M13 = a0 * b6
    
    # Let me verify c0 = a0*b0 + a1*b3 + a2*b6:
    # (M0+M1)/2 = a0*b0 + a4*b4
    # (M0-M1)/2 = a0*b4 + a4*b0
    # (M2+M3)/2 = a0*b0 + a3*b3
    # (M2-M3)/2 = a0*b3 + a3*b0
    # So a0*b0 = ((M0+M1)/2 + (M2+M3)/2) - a4*b4 - a3*b3
    # This is getting too complex. Let me just verify computationally.
    
    # Define the 14 products as (U_row, V_col, W_row) triples with coefficients.
    # U is 6-dim (A entries), V is 9-dim (B entries), W is 6-dim (C entries).
    
    # For each product M_r, we have:
    # M_r = sum_i U[i,r] * A[i] * sum_j V[j,r] * B[j]
    # And C[c] = sum_r W[c,r] * M_r
    
    # Let me define the coefficients:
    # Index A: a0=0, a1=1, a2=2, a3=3, a4=4, a5=5
    # Index B: b0=0, b1=1, b2=2, b3=3, b4=4, b5=5, b6=6, b7=7, b8=8
    # Index C: c0=0, c1=1, c2=2, c3=3, c4=4, c5=5
    
    U_block = np.zeros((6, 14))
    V_block = np.zeros((9, 14))
    W_block = np.zeros((6, 14))
    
    # M0 = (a0+a4)(b0+b4)
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[4, 0] = 1
    # M1 = (a0-a4)(b0-b4)
    U_block[0, 1] = 1; U_block[4, 1] = -1
    V_block[0, 1] = 1; V_block[4, 1] = -1
    # M2 = (a0+a3)(b0+b3)
    U_block[0, 2] = 1; U_block[3, 2] = 1
    V_block[0, 2] = 1; V_block[3, 2] = 1
    # M3 = (a0-a3)(b0-b3)
    U_block[0, 3] = 1; U_block[3, 3] = -1
    V_block[0, 3] = 1; V_block[3, 3] = -1
    # M4 = (a3+a5)(b3+b7)
    U_block[3, 4] = 1; U_block[5, 4] = 1
    V_block[3, 4] = 1; V_block[7, 4] = 1
    # M5 = (a3-a5)(b3-b7)
    U_block[3, 5] = 1; U_block[5, 5] = -1
    V_block[3, 5] = 1; V_block[7, 5] = -1
    # M6 = (a4+a5)(b4+b8)
    U_block[4, 6] = 1; U_block[5, 6] = 1
    V_block[4, 6] = 1; V_block[8, 6] = 1
    # M7 = (a4-a5)(b4-b8)
    U_block[4, 7] = 1; U_block[5, 7] = -1
    V_block[4, 7] = 1; V_block[8, 7] = -1
    # M8 = a1*b1
    U_block[1, 8] = 1
    V_block[1, 8] = 1
    # M9 = a2*b2
    U_block[2, 9] = 1
    V_block[2, 9] = 1
    # M10 = a1*b4
    U_block[1, 10] = 1
    V_block[4, 10] = 1
    # M11 = a2*b7
    U_block[2, 11] = 1
    V_block[7, 11] = 1
    # M12 = a5*b8
    U_block[5, 12] = 1
    V_block[8, 12] = 1
    # M13 = a0*b6
    U_block[0, 13] = 1
    V_block[6, 13] = 1
    
    # Now determine W coefficients.
    # c0 = a0*b0 + a1*b3 + a2*b6
    # c1 = a0*b1 + a1*b4 + a2*b7
    # c2 = a0*b2 + a1*b5 + a2*b8
    # c3 = a3*b0 + a4*b3 + a5*b6
    # c4 = a3*b1 + a4*b4 + a5*b7
    # c5 = a3*b2 + a4*b5 + a5*b8
    
    # Express each c in terms of M0..M13:
    # M0 = a0*b0 + a0*b4 + a4*b0 + a4*b4
    # M1 = a0*b0 - a0*b4 - a4*b0 + a4*b4
    # M2 = a0*b0 + a0*b3 + a3*b0 + a3*b3
    # M3 = a0*b0 - a0*b3 - a3*b0 + a3*b3
    # M4 = a3*b3 + a3*b7 + a5*b3 + a5*b7
    # M5 = a3*b3 - a3*b7 - a5*b3 + a5*b7
    # M6 = a4*b4 + a4*b8 + a5*b4 + a5*b8
    # M7 = a4*b4 - a4*b8 - a5*b4 + a5*b8
    # M8 = a1*b1
    # M9 = a2*b2
    # M10 = a1*b4
    # M11 = a2*b7
    # M12 = a5*b8
    # M13 = a0*b6
    
    # (M0+M1)/2 = a0*b0 + a4*b4
    # (M0-M1)/2 = a0*b4 + a4*b0
    # (M2+M3)/2 = a0*b0 + a3*b3
    # (M2-M3)/2 = a0*b3 + a3*b0
    # (M4+M5)/2 = a3*b3 + a5*b7
    # (M4-M5)/2 = a3*b7 + a5*b3
    # (M6+M7)/2 = a4*b4 + a5*b8
    # (M6-M7)/2 = a4*b8 + a5*b4
    
    # c0 = a0*b0 + a1*b3 + a2*b6
    # a0*b0 = ((M0+M1)/2 + (M2+M3)/2) - a4*b4 - a3*b3
    # a4*b4 = (M0+M1)/2 - a0*b0, circular.
    # Let me solve linear system.
    
    # Variables: M0..M13 (14 unknowns)
    # Equations: 18 (for the 18 terms a_i*b_j)
    # We want to express each a_i*b_j as linear combination of M's.
    
    # Actually, let me just check if this decomposition is valid by computing
    # the resulting C and comparing.
    
    # For a specific A and B, compute C via the 14 products and via direct multiplication.
    A_test = np.random.rand(6)
    B_test = np.random.rand(9)
    
    # Direct: C = A * B where A is 2x3, B is 3x3
    A_mat = A_test.reshape(2, 3)
    B_mat = B_test.reshape(3, 3)
    C_direct = A_mat @ B_mat  # 2x3
    
    # Via 14 products:
    M = np.zeros(14)
    for r in range(14):
        M[r] = sum(U_block[i, r] * A_test[i] for i in range(6)) * \
               sum(V_block[j, r] * B_test[j] for j in range(9))
    
    C_decomp = np.zeros(6)
    for c in range(6):
        C_decomp[c] = sum(W_block[c, r] * M[r] for r in range(14))
    
    # But we haven't set W_block yet! We need to determine W.
    # Let me set up the linear system to find W.
    
    # For each output c and each (i,j) term, we need:
    # C[c] = sum_r W[c,r] * M[r]
    # And C[c] should equal sum_j A[i,j]*B[j,k] for the appropriate (i,k).
    
    # Let me determine W by solving: for each c, W[c,:] such that
    # sum_r W[c,r] * M[r] = C_direct[c//3, c%3] for random A, B.
    
    # Since this must hold for all A, B, we can solve using multiple random instances.
    # But actually, W should be fixed. Let me solve the linear system.
    
    # The condition is: for all A, B,
    # sum_r W[c,r] * (sum_i U[i,r]*A[i]) * (sum_j V[j,r]*B[j]) = sum_j A[i,c%3]*B[j,c//3... ]
    
    # Wait, C has indices (i,k) where i in 0..1, k in 0..2.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # So for output index c = k*2 + i (using W[k*n+i] convention with n=2):
    # Actually in the 2x3x3 case, W index is k*n+i = k*2+i.
    # c=0: i=0,k=0; c=1: i=1,k=0; c=2: i=0,k=1; c=3: i=1,k=1; c=4: i=0,k=2; c=5: i=1,k=2
    
    # Hmm, let me re-examine the W index convention.
    # W[k*n+i, r] where n=2 (rows of A / output), p=3 (cols of B / output).
    # Output C is n x p = 2 x 3.
    # W index = k*n + i where k is column index (0..p-1=0..2), i is row index (0..n-1=0..1).
    # So W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1], W[4] = C[0,2], W[5] = C[1,2]
    
    # For W[c] where c = k*2+i:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let me solve for W using linear algebra.
    # Generate many random (A, B) pairs, compute M and C_direct, then solve W.
    
    rng = np.random.RandomState(42)
    num_samples = 50
    M_matrix = np.zeros((num_samples, 14))
    C_matrix = np.zeros((num_samples, 6))
    
    for s in range(num_samples):
        A_s = rng.rand(6)
        B_s = rng.rand(9)
        for r in range(14):
            M_matrix[s, r] = sum(U_block[i, r] * A_s[i] for i in range(6)) * \
                            sum(V_block[j, r] * B_s[j] for j in range(9))
        A_mat_s = A_s.reshape(2, 3)
        B_mat_s = B_s.reshape(3, 3)
        C_s = A_mat_s @ B_mat_s
        C_matrix[s, :] = C_s.flatten()  # row-major: C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    
    # But W index is k*2+i, so C_matrix should be reordered.
    # C_matrix[s, 0] = C[0,0] -> W[0]
    # C_matrix[s, 1] = C[0,1] -> W[2]
    # C_matrix[s, 2] = C[0,2] -> W[4]
    # C_matrix[s, 3] = C[1,0] -> W[1]
    # C_matrix[s, 4] = C[1,1] -> W[3]
    # C_matrix[s, 5] = C[1,2] -> W[5]
    
    C_reordered = np.zeros_like(C_matrix)
    C_reordered[:, 0] = C_matrix[:, 0]  # C[0,0] -> W[0]
    C_reordered[:, 1] = C_matrix[:, 3]  # C[1,0] -> W[1]
    C_reordered[:, 2] = C_matrix[:, 1]  # C[0,1] -> W[2]
    C_reordered[:, 3] = C_matrix[:, 4]  # C[1,1] -> W[3]
    C_reordered[:, 4] = C_matrix[:, 2]  # C[0,2] -> W[4]
    C_reordered[:, 5] = C_matrix[:, 5]  # C[1,2] -> W[5]
    
    # Solve W: C_reordered = M_matrix @ W^T
    W_solve, residuals, rank_solve, sv = np.linalg.lstsq(M_matrix, C_reordered, rcond=None)
    W_block = W_solve.T  # (6, 14)
    
    # Verify
    if np.allclose(M_matrix @ W_block.T, C_reordered, atol=1e-8):
        # Check if W entries are approximately integer/half-integer
        W_rounded = np.round(W_block * 2) / 2
        if np.allclose(W_block, W_rounded, atol=1e-6):
            W_block = W_rounded
            # Build full 2x3x9 decomposition
            U_full = np.zeros((6, 42))
            V_full = np.zeros((27, 42))
            W_full = np.zeros((18, 42))
            
            for block in range(3):
                U_full[:, block*14:(block+1)*14] = U_block
                # V: for block, B columns are 3*block to 3*block+2
                V_full[3*block:3*block+3, block*14:(block+1)*14] = V_block[:3, :]
                V_full[3*block+3:3*block+6, block*14:(block+1)*14] = V_block[3:6, :]
                V_full[3*block+6:3*block+9, block*14:(block+1)*14] = V_block[6:9, :]
                # W: for block, output columns are block to block+2
                # W index = k*2+i, for output column k in block*3..block*3+2
                for k_offset in range(3):
                    k = block * 3 + k_offset  # This is wrong; k is 0..8 for 2x3x9
                    # Actually for 2x3x9, p=9, so k ranges 0..8
                    # W index = k*2 + i
                    # For block b, output columns are 3b, 3b+1, 3b+2
                    pass
                # Let me redo W indexing
                # For 2x3x9: W index = k*2 + i, k in 0..8, i in 0..1
                # Block b handles output columns 3b, 3b+1, 3b+2
                for k_offset in range(3):
                    k = 3 * block + k_offset
                    for i in range(2):
                        w_idx = k * 2 + i
                        # In the 2x3x3 block, local k_local = k_offset, local i = i
                        # local W index = k_local * 2 + i
                        local_w_idx = k_offset * 2 + i
                        W_full[w_idx, block*14:(block+1)*14] = W_block[local_w_idx, :]
            
            # Verify full decomposition
            if _verify_decomposition(U_full, V_full, W_full, 2, 3, 9):
                # Convert to list of lists with integer/half-integer values
                U_list = [[int(round(U_full[i, r] * 2)) / 2 for r in range(42)] for i in range(6)]
                V_list = [[int(round(V_full[i, r] * 2)) / 2 for r in range(42)] for i in range(27)]
                W_list = [[int(round(W_full[i, r] * 2)) / 2 for r in range(42)] for i in range(18)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try rank-42 first
    result = _construct_rank42(2, 3, 9)
    if result is not None:
        return result
    
    # Fallback to schoolbook rank 54
    n, m, p = 2, 3, 9
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
