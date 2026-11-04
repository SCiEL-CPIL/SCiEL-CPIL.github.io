"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes C = A @ B."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With trace convention: W[k*n + i, r]
    rank = U.shape[1]
    # For each term r, contribution to C[i,k] is:
    # sum_{i,j} U[i*m+j, r] * A[i,j] * sum_{j,k} V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Actually, the tensor decomposition is:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    # Wait, that's not right. Let me think again.
    # The tensor T[i,j,k] = delta (for matrix mult) is decomposed as:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r]) * W[i,k,r]
    # But with trace convention W[k*n+i], so W[i,k] = W[k*n+i]
    
    # Let's just check with random matrices
    np.random.seed(42)
    A = np.random.randint(-3, 3, size=(n, m))
    B = np.random.randint(-3, 3, size=(m, p))
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p), dtype=int)
    for r in range(rank):
        # u_part[i,j] = U[i*m+j, r]
        # v_part[j,k] = V[j*p+k, r]
        # w_part[i,k] = W[k*n+i, r]
        u_part = U.reshape(n, m, rank)[:, :, r]
        v_part = V.reshape(m, p, rank)[:, :, r]
        w_part = W.reshape(p, n, rank)[:, :, r]  # W[k*n+i] -> w_part[k, i]
        # C[i,k] += (sum_j A[i,j]*u_part[i,j]) * (sum_j B[j,k]*v_part[j,k]) * w_part[k, i]
        a_scaled = A * u_part
        b_scaled = B * v_part
        # C[i,k] += a_scaled[i,:] dot b_scaled[:,k] * w_part[k,i]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += np.dot(a_scaled[i, :], b_scaled[:, k]) * w_part[k, i]
    
    return np.array_equal(C_expected, C_computed)


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use 2x2 block structure with Kronecker product decomposition.
    # A is 4x4: 2x2 blocks of size 2x2. A[ai][bj] where ai,bj in {0,1}
    #   A[i,j] with i=2*ai+ai2, j=2*bj+bj2, ai2,bj2 in {0,1}
    # B is 4x6: 2x3 blocks of size 2x2. B[bj][ck] where bj in {0,1}, ck in {0,1,2}
    #   B[j,k] with j=2*bj+bj2, k=2*ck+ck2, bj2,ck2 in {0,1}
    # C is 4x6: 2x3 blocks of size 2x2. C[ai][ck] where ai in {0,1}, ck in {0,1,2}
    #   C[i,k] with i=2*ai+ai2, k=2*ck+ck2, ai2,ck2 in {0,1}
    #
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]  (block matrix multiplication)
    # Each block product is 2x2 * 2x2 -> 2x2, decomposed via Strassen (rank 7).
    #
    # Outer structure: 2x2 * 2x3 -> 2x3 block multiplication
    # Use 11 outer terms: 7 (Strassen for cols 0,1) + 4 (schoolbook for col 2)
    # Each outer term involves one 2x2*2x2 block product (rank 7)
    # Total rank: 11 * 7 = 77
    
    sU, sV, sW = _strassen_2x2()
    
    total_rank = 77
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col_idx = 0
    
    # Part 1: Strassen for C0, C1 (7 outer terms)
    for t in range(7):
        for inner_t in range(7):
            col = col_idx
            col_idx += 1
            
            # U part
            for ai in range(2):
                for bj in range(2):
                    coeff_outer = sU[ai*2+bj, t]
                    if coeff_outer != 0:
                        for ai2 in range(2):
                            for bj2 in range(2):
                                coeff_inner = sU[ai2*2+bj2, inner_t]
                                if coeff_inner != 0:
                                    u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                                    U[u_row, col] += coeff_outer * coeff_inner
            # V part
            for bj in range(2):
                for ck in range(2):
                    coeff_outer = sV[bj*2+ck, t]
                    if coeff_outer != 0:
                        for bj2 in range(2):
                            for ck2 in range(2):
                                coeff_inner = sV[bj2*2+ck2, inner_t]
                                if coeff_inner != 0:
                                    v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                                    V[v_row, col] += coeff_outer * coeff_inner
            # W part
            for ai in range(2):
                for ck in range(2):
                    coeff_outer = sW[ck*2+ai, t]
                    if coeff_outer != 0:
                        for ai2 in range(2):
                            for ck2 in range(2):
                                coeff_inner = sW[ck2*2+ai2, inner_t]
                                if coeff_inner != 0:
                                    w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                                    W[w_row, col] += coeff_outer * coeff_inner

    # Part 2: Schoolbook for C2 (4 outer terms)
    for ai in range(2):
        for bj in range(2):
            for inner_t in range(7):
                col = col_idx
                col_idx += 1
                
                # U: block (ai, bj)
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2+bj2, inner_t]
                        if coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_inner
                            
                # V: block (bj, 2)
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2+ck2, inner_t]
                        if coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*2 + ck2)
                            V[v_row, col] += coeff_inner
                            
                # W: block (ai, 2)
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2+ai2, inner_t]
                        if coeff_inner != 0:
                            w_row = (2*2 + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_inner

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
