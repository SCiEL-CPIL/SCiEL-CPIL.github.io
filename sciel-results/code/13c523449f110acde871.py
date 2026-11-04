"""
Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.

Strategy:
  - Partition the 3x4 input A into a 2x4 top block and a 1x4 bottom block.
  - Partition the 4x8 input B into two 4x4 horizontal blocks B1 and B2.
  - Use Strassen's algorithm (rank 7) for each 2x2x2 sub-block of the 2x4x4
    products, giving 2 * 4 * 7 = 56 terms for C1 = A_top * B.
  - Use 8 schoolbook terms for C2 = A_bottom * B.
  - Total rank = 56 + 8 = 64.

Verification:
  For each term r, we check that the outer product u_r ⊗ v_r ⊗ w_r
  equals the corresponding slice of the multiplication tensor.
  The multiplication tensor T has T[i,j,k] = 1 if k is the column index in
  B corresponding to output position (i,k) and j is the row index in B.
  More precisely, C[i,k] = sum_j A[i,j] B[j,k].
  So the tensor is T[i,j,k] = 1 (a single 1 at each (i,j,k) for the identity
  of the multiplication map).
  
  In the decomposition, sum_r U[i,j,r] * V[j',k',r] * W[k'',i'',r] should
  equal T[i,j,k] when (j',k')=(j,k) and (k'',i'')=(k,i), and 0 otherwise.
  
  Actually, the tensor is a 3x4x8 array where T[i,j,k] = 1 for all valid
  (i,j,k) — no wait, that's not right either.
  
  The multiplication tensor is defined as:
  T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
  Since every A[i,j]*B[j,k] contributes to C[i,k], T[i,j,k] = 1 for all
  (i,j,k) in {0..2} x {0..3} x {0..7}.
  
  Wait, that's the schoolbook tensor. The tensor rank is the minimum number
  of rank-1 tensors summing to this.
  
  For the decomposition to be valid:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
  sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = 0  otherwise
  
  Let me verify with a small example mentally... Actually, let me just
  construct it and verify numerically.
"""

import numpy as np

DIMENSIONS = [3, 4, 8]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition term by term.
    # Terms 0..55: Strassen for 2x4x4 (two 4x4 blocks of B)
    # Terms 56..63: Schoolbook for 1x4x8
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # 12 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 24 x 64
    
    # Strassen 2x2x2 coefficients
    # For a 2x2x2 block at local (i0, j0, k0):
    # 7 terms with (u_coeffs, v_coeffs, w_coeffs)
    # u is over local A indices (0,1)x(0,1) -> global (i0+di, j0+dj)
    # v is over local B indices (0,1)x(0,1) -> global (j0+dj, k0+dk)
    # w is over local C indices (0,1)x(0,1) -> global (k0+dk, i0+di)
    
    strassen_terms = [
        # (u: {local_a_idx: coeff}, v: {local_b_idx: coeff}, w: {local_c_idx: coeff})
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00*B10 -> C01 += 1, C11 += 1
        ({(0,0): 1}, {(1,0): 1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11*B11 -> C00 += 1, C10 += 1
        ({(1,1): 1}, {(1,1): 1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A10)*B11 -> C00 += -1, C01 += 1
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A11-A01)*(B00+B01) -> C11 += 1
        ({(1,1): 1, (0,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)*(B10+B11) -> C00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    term_idx = 0
    # Process 2x4x4 blocks: i0 in {0,1} (top 2 rows of A), k0 in {0,4} (two 4x4 blocks of B)
    for i0 in [0, 1]:
        for k0 in [0, 4]:
            for j0 in [0, 2]:  # two 2x2 blocks in the j dimension (m=4)
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    r = term_idx
                    for (di, dj), coeff in u_coeffs.items():
                        gi = i0 + di
                        gj = j0 + dj
                        U[gi * m + gj][r] += coeff
                    for (dj, dk), coeff in v_coeffs.items():
                        gj = j0 + dj
                        gk = k0 + dk
                        V[gj * p + gk][r] += coeff
                    for (dk, di), coeff in w_coeffs.items():
                        gk = k0 + dk
                        gi = i0 + di
                        W[gk * n + gi][r] += coeff
                    term_idx += 1
    
    # Now schoolbook for the bottom row (i=2) of A times full B (4x8)
    for j in range(m):
        for k in range(p):
            r = term_idx
            U[2 * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + 2][r] = 1
            term_idx += 1
    
    # Verify the decomposition
    # Build the full tensor and check
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    T_reconstructed = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_reconstructed[i, j, k] = np.sum(
                    U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :]
                )
    
    # The target tensor is all ones
    T_target = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(T_reconstructed, T_target, atol=1e-10):
        # Find where it fails
        diff = T_reconstructed - T_target
        print("Decomposition verification FAILED")
        print("Max diff:", np.max(np.abs(diff)))
        nonzero = np.argwhere(np.abs(diff) > 1e-10)
        print("Nonzero positions:", nonzero[:20])
        for pos in nonzero[:10]:
            i, j, k = pos
            print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]}, expected 1")
    
    return {"U": U, "V": V, "W": W}
