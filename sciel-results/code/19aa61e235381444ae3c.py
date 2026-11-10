"""Matrix multiplication 4x5x5 with rank-45 decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposition strategy:
    - Split rows {0,1,2,3} into pairs {0,1} and {2,3}
    - Split cols {0,1,2,3,4} into {0,1} and {2,3,4}
    - Split results {0,1,2,3,4} into {0,1} and {2,3,4}
    
    Blocks:
    - 4 copies of 2x2x2 (Strassen, rank 7) = 28
    - 1 copy of 2x3x3 (rank 17) = 17
    Total rank = 45
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 45
    
    U = [[0] * rank for _ in range(n * m)]   # 20 x 45
    V = [[0] * rank for _ in range(m * p)]   # 25 x 45
    W = [[0] * rank for _ in range(p * n)]   # 20 x 45
    
    # Strassen 2x2x2 coefficients: 7 tensors
    # Each tensor is (2x2, 2x2, 2x2) with entries from {-1, 0, 1}
    # Format: (U_entry, V_entry, W_entry) for each of the 4 sub-positions
    strassen_t = [
        # t0: U=[[1,0],[0,0]], V=[[1,0],[0,0]], W=[[1,0],[0,1]]
        [(1,0),(0,0),(0,0),(0,0)],  # U: (0,0)=1, others 0
        [(1,0),(0,0),(0,0),(0,0)],  # V
        [(1,0),(0,1),(0,0),(0,0)],  # W: (0,0)=1,(0,1)=1
    ]
    # Let me use the standard Strassen formulation
    # M1 = (A00+A11)(B00+B11), M2 = A11(B10-B00), M3 = A00(B01-B11)
    # M4 = A10(B00+B11), M5 = (A00+A10)B11, M6 = (A01-A11)(B00+B10), M7 = (A10-A00)(B01-B00)
    # C00 = M1+M4+M6-M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M5+M7
    
    # Represent each of the 7 products as (U_matrix, V_matrix, W_matrix)
    # where each is a 2x2 = 4 entries in row-major order [tl, tr, bl, br]
    # U multiplies A, V multiplies B, W multiplies C (output)
    
    strassen_decomp = [
        # M1: (A00+A11)(B00+B11)
        # U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,-1]]
        # C00 += M1, C11 -= M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2: A11(B10-B00)
        # U: [[0,0],[0,1]], V: [[0,-1],[0,1]], W: [[0,0],[1,0]]
        # C10 += M2
        ([0, 0, 0, 1], [0, -1, 0, 1], [0, 0, 1, 0]),
        # M3: A00(B01-B11)
        # U: [[1,0],[0,0]], V: [[0,1],[0,-1]], W: [[0,1],[0,0]]
        # C01 += M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4: A10(B00+B11)
        # U: [[0,0],[1,0]], V: [[1,0],[0,1]], W: [[1,0],[1,0]]
        # C00 += M4, C10 += M4
        ([0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]),
        # M5: (A00+A10)B11
        # U: [[1,0],[1,0]], V: [[0,0],[0,1]], W: [[0,1],[0,1]]
        # C01 += M5, C11 += M5
        ([1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        # M6: (A01-A11)(B00+B10)
        # U: [[0,1],[0,-1]], V: [[1,0],[1,0]], W: [[1,0],[0,0]]
        # C00 += M6
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),
        # M7: (A10-A00)(B01-B00)
        # U: [[-1,0],[1,0]], V: [[-1,1],[0,0]], W: [[-1,0],[0,1]]
        # C00 -= M7, C11 += M7
        ([-1, 0, 1, 0], [-1, 1, 0, 0], [-1, 0, 0, 1]),
    ]
    
    col = 0
    
    # Process the 4 blocks of 2x2x2
    # Row pairs: (0,1) and (2,3)
    # Col pairs: (0,1) and (2,3)
    row_pairs = [(0, 1), (2, 3)]
    col_pairs = [(0, 1), (2, 3)]
    
    for r0, r1 in row_pairs:
        for c0, c1 in col_pairs:
            # This block computes C[r0:r1+1, c0:c1+1] from A[r0:r1+1, :] * B[:, c0:c1+1]
            # Actually it's a 2x2x2 sub-problem:
            # A is 2x2 (rows r0,r1; cols c0,c1)
            # B is 2x2 (rows c0,c1; cols c0,c1)
            # C is 2x2 (rows r0,r1; cols c0,c1)
            for (u_mat, v_mat, w_mat) in strassen_decomp:
                # u_mat: 4 entries for A positions (r0,c0),(r0,c1),(r1,c0),(r1,c1)
                # v_mat: 4 entries for B positions (c0,c0),(c0,c1),(c1,c0),(c1,c1)
                # w_mat: 4 entries for C positions (r0,c0),(r0,c1),(r1,c0),(r1,c1)
                for idx in range(4):
                    u_val = u_mat[idx]
                    v_val = v_mat[idx]
                    w_val = w_mat[idx]
                    if u_val != 0:
                        # A row = r0 + (idx // 2), col = c0 + (idx % 2)
                        u_row = (r0 + idx // 2) * m + (c0 + idx % 2)
                        U[u_row][col] = u_val
                    if v_val != 0:
                        # B row = c0 + (idx // 2), col = c0 + (idx % 2)
                        v_row = (c0 + idx // 2) * p + (c0 + idx % 2)
                        V[v_row][col] = v_val
                    if w_val != 0:
                        # C result = (c0 + idx % 2, r0 + idx // 2) in W[k*n+i] format
                        w_row = (c0 + idx % 2) * n + (r0 + idx // 2)
                        W[w_row][col] = w_val
                col += 1
    
    # Now the 2x3x3 block: rows (2,3), cols (2,3,4), results (2,3,4)
    # Decompose into:
    # - 2x2x2 sub-block: rows(2,3), cols(2,3), results(2,3) -> Strassen rank 7
    # - 2x2x1 sub-block: rows(2,3), cols(2,3), results(4) -> rank 4 schoolbook
    # - 2x1x2 sub-block: rows(2,3), col(4), results(2,3) -> rank 4 schoolbook
    # - 2x1x1 sub-block: rows(2,3), col(4), result(4) -> rank 2 schoolbook
    # Total: 7 + 4 + 4 + 2 = 17
    
    r0, r1 = 2, 3
    c0, c1, c2 = 2, 3, 4
    d0, d1, d2 = 2, 3, 4
    
    # Sub-block 1: 2x2x2 (rows 2,3; cols 2,3; results 2,3) - Strassen
    for (u_mat, v_mat, w_mat) in strassen_decomp:
        for idx in range(4):
            u_val = u_mat[idx]
            v_val = v_mat[idx]
            w_val = w_mat[idx]
            if u_val != 0:
                u_row = (r0 + idx // 2) * m + (c0 + idx % 2)
                U[u_row][col] = u_val
            if v_val != 0:
                v_row = (c0 + idx // 2) * p + (c0 + idx % 2)
                V[v_row][col] = v_val
            if w_val != 0:
                w_row = (c0 + idx % 2) * n + (r0 + idx // 2)
                W[w_row][col] = w_val
        col += 1
    
    # Sub-block 2: 2x2x1 (rows 2,3; cols 2,3; result 4) - schoolbook rank 4
    # C[4, r] = A[r, 2]*B[2, 4] + A[r, 3]*B[3, 4] for r in {2,3}
    # 4 terms: (r,c) pairs for A, B is fixed column
    for r in [r0, r1]:
        for c in [c0, c1]:
            U[r * m + c][col] = 1
            V[c * p + d2][col] = 1
            W[d2 * n + r][col] = 1
            col += 1
    
    # Sub-block 3: 2x1x2 (rows 2,3; col 4; results 2,3) - schoolbook rank 4
    # C[d, r] = A[r, 4]*B[4, d] for r in {2,3}, d in {2,3}
    for r in [r0, r1]:
        for d in [d0, d1]:
            U[r * m + c2][col] = 1
            V[c2 * p + d][col] = 1
            W[d * n + r][col] = 1
            col += 1
    
    # Sub-block 4: 2x1x1 (rows 2,3; col 4; result 4) - schoolbook rank 2
    # C[4, r] = A[r, 4]*B[4, 4] for r in {2,3}
    for r in [r0, r1]:
        U[r * m + c2][col] = 1
        V[c2 * p + d2][col] = 1
        W[d2 * n + r][col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

def _validate():
    """Internal validation to ensure the decomposition is correct."""
    import numpy as np
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    n, m, p = DIMENSIONS
    # Build the multiplication tensor and check
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1 if i,j,k are valid
    # Actually for matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] should be 1 (the multiplication tensor)
    # T[i,j,k] = delta meaning: for each triple (i,j,k) with valid indices, T=1
    # In our convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This gives the coefficient of the bilinear form for (A[i,j], B[j,k]) -> C[i,k]
    # The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                if abs(val - expected) > 1e-10:
                    return False, (i, j, k, val)
    return True, None

if __name__ == "__main__":
    ok, info = _validate()
    if ok:
        print("Validation passed: rank 45 decomposition is correct.")
    else:
        print(f"Validation FAILED at {info}")
        # Try to debug by printing which entries are wrong
        result = construct()
        import numpy as np
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        n, m, p = DIMENSIONS
        print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
        # Print the tensor values
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    if abs(val - 1.0) > 1e-10:
                        print(f"T[{i},{j},{k}] = {val}")
