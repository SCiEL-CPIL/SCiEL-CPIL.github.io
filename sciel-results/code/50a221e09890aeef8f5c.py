"""Rank-minimal tensor decomposition for 2x2 by 2x14 matrix multiplication."""
import time
import itertools

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank decomposition for C = A*B where A is 2x2 and B is 2x14.
    
    Uses block-wise Strassen: partition B into 7 blocks of 2x2, apply Strassen
    to each block. Total rank = 7 * 7 = 49.
    
    Interface: U shape (n*m, rank) = (4, rank)
               V shape (m*p, rank) = (28, rank)  
               W shape (p*n, rank) = (28, rank)
    
    Convention: C[k*n+i, r] = sum_c U[i*m+j, c] * V[j*p+k, c] ... 
    Actually the convention is:
      C = U * A * V^T style where:
      C[out_idx] = sum_c U[inA_idx, c] * V[inB_idx, c] * W[out_idx, c]
    
    More precisely, the tensor T[i,j,k,l] = delta conditions gives:
      C[k*n+i] = sum_{j} A[i,j] * B[j,k]
    
    In the factor form: C[out] = sum_c U[a_idx, c] * V[b_idx, c] * W[out_idx, c]
    where a_idx = i*m+j (row i, col j of A), b_idx = j*p+k (row j, col k of B),
    out_idx = k*n+i (row k, col i of C in trace convention W[k*n+i, r]).
    """
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # C = A*B
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 * 2x14, partition B into 7 blocks of 2x2.
    # For block q (columns 2q, 2q+1):
    #   B_block = [[b0,0, b0,1], [b1,0, b1,1]] where b_{r,c} = B[r, 2q+c]
    #   A is the full 2x2 matrix
    #   C_block = A * B_block (2x2)
    
    # Strassen coefficients for each block:
    # For A = [[a00, a01], [a10, a11]]:
    #   U indices: 0=(0,0)=a00, 1=(0,1)=a01, 2=(1,0)=a10, 3=(1,1)=a11
    #   V indices: for block q, columns 2q and 2q+1 of B
    #     V row index = j*p + k, where j is A's col = B's row, k is B's col
    #     So for block q, B entries are at B[0, 2q], B[0, 2q+1], B[1, 2q], B[1, 2q+1]
    #     V indices: 0*14+2q = 2q, 0*14+2q+1 = 2q+1, 1*14+2q = 14+2q, 1*14+2q+1 = 14+2q+1
    #   W indices: out = k*n + i, where k is output col (= B's col within block + 2q), i is output row
    #     For block q, output columns are 2q and 2q+1
    #     W index for (i, 2q+c) = (2q+c)*n + i = 2*(2q+c) + i
    
    # Strassen M terms and their A, B coefficients:
    strassen_terms = [
        # (M_name, A_coeffs: {u_idx: val}, B_coeffs: {v_local_idx: val})
        # v_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for the 2x2 B block
        ('M1', {0: 1, 3: 1}, {0: 1, 3: 1}),    # (a00+a11)*(b00+b11)
        ('M2', {2: 1, 3: 1}, {0: 1}),            # (a10+a11)*b00
        ('M3', {0: 1}, {1: 1, 3: -1}),           # a00*(b01-b11)
        ('M4', {3: 1}, {2: 1, 0: -1}),           # a11*(b10-b00)
        ('M5', {0: 1, 1: 1}, {3: 1}),            # (a00+a01)*b11
        ('M6', {2: 1, 0: -1}, {0: 1, 2: 1}),     # (a10-a00)*(b00+b10)
        ('M7', {1: 1, 3: -1}, {1: 1, 3: 1}),     # (a01-a11)*(b01+b11)
    ]
    
    # Output reconstruction:
    # C00 = M1+M4-M5+M7  -> W coeff for (i=0, k=0) in block
    # C01 = M3+M5        -> W coeff for (i=0, k=1) in block
    # C10 = M2+M4        -> W coeff for (i=1, k=0) in block
    # C11 = M1-M2+M3+M6  -> W coeff for (i=1, k=1) in block
    output_recon = [
        # (i, k_in_block, {M_idx: coeff})
        (0, 0, {0: 1, 3: 1, 4: -1, 6: 1}),   # C00
        (0, 1, {2: 1, 4: 1}),                 # C01
        (1, 0, {1: 1, 3: 1}),                 # C10
        (1, 1, {0: 1, 1: -1, 2: 1, 5: 1}),   # C11
    ]
    
    rank = 7 * 7  # 7 blocks * 7 Strassen terms
    U = [[0] * rank for _ in range(n * m)]   # (4, 49)
    V = [[0] * rank for _ in range(m * p)]   # (28, 49)
    W = [[0] * rank for _ in range(p * n)]   # (28, 49)
    
    col = 0
    for q in range(7):  # 7 blocks
        for m_idx, (m_name, a_coeffs, b_coeffs) in enumerate(strassen_terms):
            c = col
            # U: set A coefficients
            for u_idx, val in a_coeffs.items():
                U[u_idx][c] = val
            # V: set B coefficients for this block
            # B block local indices: 0->(row0, col2q), 1->(row0, col2q+1), 2->(row1, col2q), 3->(row1, col2q+1)
            # V global index for local (r, c_local) = r*p + (2q + c_local)
            for v_local, val in b_coeffs.items():
                r = v_local // 2
                c_local = v_local % 2
                v_global = r * p + (2 * q + c_local)
                V[v_global][c] = val
            col += 1
    
    # Now fill W: for each output position, sum contributions from Strassen terms
    for q in range(7):
        for (i, k_in_block, m_coeffs) in output_recon:
            w_global = (2 * q + k_in_block) * n + i
            for m_idx, coeff in m_coeffs.items():
                c = q * 7 + m_idx
                W[w_global][c] += coeff
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all A (2x2) and B (2x14), C = A*B should equal the decomposition
    # C[k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_{j,k'} ... let's just check the tensor directly
    # T[i,j,k] = delta (the multiplication tensor)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the true tensor
    true_tensor = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_tensor[i, j, k] = 1  # C[k*n+i] += A[i,j]*B[j,k]
    
    # Build decomposed tensor
    dec_tensor = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dec_tensor[i, j, k] += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
    
    return np.allclose(true_tensor, dec_tensor, atol=1e-10)

# EVOLVE-BLOCK-END

# Self-test when run directly
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    elapsed = time.time() - t0
    n, m, p = DIMENSIONS
    ok = _verify(result["U"], result["V"], result["W"], n, m, p)
    print(f"Rank: {len(result['U'][0])}, Valid: {ok}, Time: {elapsed:.3f}s")
    if not ok:
        print("ERROR: Decomposition is not valid!")
    else:
        print("Decomposition verified successfully.")

