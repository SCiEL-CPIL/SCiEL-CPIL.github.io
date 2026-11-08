"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-189 decomposition for 6x6 matrix multiplication.
    
    Strategy: decompose 6x6 into 3x3 blocks of 2x2 sub-matrices,
    apply Strassen's algorithm (rank 7) to each 2x2 block multiplication.
    Total rank = 27 block multiplications * 7 Strassen terms = 189.
    
    Indexing convention:
    - U: shape (n*m, rank), U[i*m + j, r] multiplies into the (i,j) entry of A
    - V: shape (m*p, rank), V[j*p + k, r] multiplies into the (j,k) entry of B  
    - W: shape (p*n, rank), W[k*n + i, r] contributes to the (i,k) entry of C
    
    This gives C[i,k] = sum_r U[i*m+k... wait, let me re-read the convention.
    
    The convention is: W[k*n+i, r] - so W is indexed by output column k and row i.
    The decomposition satisfies:
      C[i,k] = sum_r (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j'} V[j'*p+k, r] * B[j',k])
    
    Actually, the standard tensor decomposition for matrix mult A(n,m) * B(m,p) = C(n,p):
      C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    where the sum is over r (and implicitly the intermediate index j is summed within U and V).
    
    More precisely, the bilinear form is:
      C[i,k] = sum_{r=1}^{rank} W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Wait, that's not right either. The standard decomposition is:
      C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    
    No - the correct form is:
      C[i,k] = sum_{r} W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Actually I think the convention means the decomposition is:
      C[i,k] = sum_r U[i*m + ?, r] ... 
    
    Let me just use the standard form where:
      C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    But actually for the evaluator, what matters is that the product of the three
    factorized forms reconstructs the matrix product. Let me use the straightforward
    approach where for each Strassen term within each block position, we set the
    appropriate entries.
    """
    _t0 = time.time()
    _deadline = _t0 + 145.0
    
    n, m, p = DIMENSIONS  # all 6
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    # Strassen 2x2 patterns:
    # For multiplying A(2x2) * B(2x2) = C(2x2):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M6
    #
    # Each pattern: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (local_row, local_col) -> value (for A submatrix)
    # v_coeffs: dict from (local_row, local_col) -> value (for B submatrix)
    # w_coeffs: dict from (local_row, local_col) -> value (for C submatrix output)
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11): contributes to C00(+1), C11(+1)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00: contributes to C10(+1), C11(-1)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11): contributes to C01(+1), C11(-1)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
        # M3 = A11(B10-B00): contributes to C00(-1), C10(+1)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
        # M4 = (A00+A01)B11: contributes to C01(+1)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        # M5 = (A10-A00)(B00+B01): contributes to C00(+1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
        # M6 = (A01-A11)(B10+B11): contributes to C11(+1)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        off_I = I * block_size
        for J in range(blocks_per_dim):
            off_J = J * block_size
            for K in range(blocks_per_dim):
                off_K = K * block_size
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # U[i*m + j, r]: A's (I*2+pi, J*2+pj) entry
                    for (pi, pj), val in u_coeffs.items():
                        r = off_I + pi
                        c = off_J + pj
                        U[r * m + c][col_idx] += val
                    
                    # V[j*p + k, r]: B's (J*2+pi, K*2+pj) entry
                    for (pi, pj), val in v_coeffs.items():
                        r = off_J + pi
                        c = off_K + pj
                        V[r * p + c][col_idx] += val
                    
                    # W[k*n + i, r]: C's (I*2+pi, K*2+pj) entry
                    for (pi, pj), val in w_coeffs.items():
                        i = off_I + pi
                        k = off_K + pj
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                
                if time.time() > _deadline:
                    return {"U": U, "V": "V", "W": W}
            if time.time() > _deadline:
                return {"U": U, "V": V, "W": W}
        if time.time() > _deadline:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
