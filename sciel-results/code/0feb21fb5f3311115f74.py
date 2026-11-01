import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Try to find a rank-27 decomposition for M_{2,4,4}.
    # Strategy: Use the block structure but look for algebraic identities
    # that allow combining terms.
    #
    # A (2x4) = [A_L | A_R], each 2x2
    # B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]], each 2x2
    # C (2x4) = [C_L | C_R], each 2x2
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    #
    # The 4 independent 2x2*2x2 products give rank 28 via Strassen.
    # To get rank 27, we need to find one linear dependency among the 28
    # rank-1 terms, or use a different algebraic structure.
    #
    # Alternative: Try to find a direct rank-27 decomposition by
    # setting up the tensor equations and solving.
    
    # First, let's try the straightforward rank-28 construction and verify it.
    # Then attempt to reduce.
    
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_indices, v_indices, w_indices, col_idx):
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff

    # Strassen 2x2 terms (same as before)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(left_right, local_idx):
        if left_right == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]

    def map_v_b(block, local_idx):
        row_off = {'TL': 0, 'TR': 0, 'BL': 2, 'BR': 2}[block]
        col_off = {'TL': 0, 'TR': 2, 'BL': 0, 'BR': 2}[block]
        r = local_idx // 2
        c = local_idx % 2
        return (row_off + r) * 4 + (col_off + c)

    def map_w_c(left_right, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        global_c = c if left_right == 'L' else 2 + c
        return global_c * 2 + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            u_global = [(map_u_a(a_side, idx), coeff) for idx, coeff in u_local]
            v_global = [(map_v_b(b_block, idx), coeff) for idx, coeff in v_local]
            w_global = [(map_w_c(c_side, idx), coeff) for idx, coeff in w_local]
            add_term(u_global, v_global, w_global, start_col)
            start_col += 1
        return start_col

    col = apply_strassen('L', 'TL', 'L', 0)
    col = apply_strassen('R', 'BL', 'L', col)
    col = apply_strassen('L', 'TR', 'R', col)
    col = apply_strassen('R', 'BR', 'R', col)
    
    # Verify the rank-28 solution
    def verify(U_mat, V_mat, W_mat):
        A = np.arange(n * m).reshape(n, m).astype(float) + 1
        B = np.arange(m * p).reshape(m, p).astype(float) + 2
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U_mat.shape[1]):
                    for jj in range(m):
                        acc += U_mat[ii * m + jj, cc] * V_mat[jj * p + kk, cc] * W_mat[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf)
    
    if not verify(U, V, W):
        # Fallback: return naive rank-32
        U32 = np.zeros((n*m, 32), dtype=int)
        V32 = np.zeros((m*p, 32), dtype=int)
        W32 = np.zeros((p*n, 32), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U32[i*m+j, idx] = 1
                    V32[j*p+k, idx] = 1
                    W32[k*n+i, idx] = 1
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
    
    # Try to find a rank-27 decomposition by checking if any column
    # is linearly dependent on others in a way that allows elimination.
    # This is complex, so let's try a different algebraic approach.
    #
    # Alternative idea: Use a different factorization of the tensor.
    # The tensor M_{2,4,4} can be written as:
    # T[i,j,k] = delta_{i,j} * delta_{j,k} ... no, that's not right.
    # T[i,j,k] = 1 if the indices are consistent with matrix mult.
    # Actually T_{i,j,k} = 1 for all valid (i,j,k) where i in [0,1], j in [0,3], k in [0,3]
    # The multiplication tensor has T[i,j,k] = 1 if j is the shared index.
    # More precisely, for C[i,k] = sum_j A[i,j]*B[j,k], the tensor is:
    # T[i,j,k] = 1 (it's the structure tensor of matrix multiplication)
    
    # Let me try to search for a rank-27 solution using a different strategy.
    # Key idea: In the 2x4x4 case, note that A has only 2 rows.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3.
    # 
    # We can try to use the fact that for fixed i, C[i,:] = A[i,:] @ B.
    # This is a 1x4 vector times 4x4 matrix = 1x4 vector.
    # The bilinear complexity of 1x4 * 4x4 is 16 (each output needs 4 mults, 4 outputs).
    # But we have 2 such rows, and they share the B matrix.
    # 
    # Total naive: 2 * 4 * 4 = 32.
    # 
    # Can we do better by sharing multiplications between the two rows of A?
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # For each k, we compute 2 outputs from 4+4=8 inputs (but B is shared).
    # The bilinear complexity of computing [A[0,0]*B[0,k], A[1,0]*B[0,k], A[0,1]*B[1,k], A[1,1]*B[1,k], ...]
    # is 8 per k (since A[i,j]*B[j,k] are all independent bilinear terms).
    # Total: 4 * 8 = 32. No savings from sharing B.
    #
    # So the structure doesn't obviously allow < 28. Let's keep rank 28.
    
    if time.time() - _t0 > _deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
