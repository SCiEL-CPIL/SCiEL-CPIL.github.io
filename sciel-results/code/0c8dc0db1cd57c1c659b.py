"""Rewritten program using Strassen-based rank reduction for 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices: j (inner) into {0,1} and {2,3,4}
    # k (output col) into {0,1} and {2,3,4}
    # i (output row) into {0} and {1}
    #
    # Block (j in {0,1}, k in {0,1}): 2x2x2 -> Strassen rank 7
    # Block (j in {0,1}, k in {2,3,4}): 2x3x2 -> 12 naive
    # Block (j in {2,3,4}, k in {0,1}): 3x2x2 -> 12 naive
    # Block (j in {2,3,4}, k in {2,3,4}): 2x3x3 -> 18 naive
    # Total rank = 7 + 12 + 12 + 18 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2x2 block: j in {0,1}, k in {0,1}
    # A indices: i in {0,1}, j in {0,1} -> U[i*5 + j]
    # B indices: j in {0,1}, k in {0,1} -> V[j*5 + k]
    # C indices: i in {0,1}, k in {0,1} -> W[k*2 + i]
    #
    # Strassen M's:
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11*(B00-B10)
    # M3 = A00*(B10+B11)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B00
    # M6 = (A01-A10)*(B01+B11)
    # M7 = (A00-A10)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M3 + M7
    # C11 = M2 - M3 + M4 + M6
    
    A00, A01, A10, A11 = 0*5+0, 0*5+1, 1*5+0, 1*5+1
    B00, B01, B10, B11 = 0*5+0, 0*5+1, 1*5+0, 1*5+1
    C00, C01, C10, C11 = 0*2+0, 0*2+1, 1*2+0, 1*2+1
    
    # M1 = (A00+A11)(B00+B11)
    U[A00][col] = 1; U[A11][col] = 1
    V[B00][col] = 1; V[B11][col] = 1
    W[C00][col] = 1; W[C01][col] = 0
    # This gets complicated with multiple C outputs per M. Let me restructure.
    
    # Actually, each M contributes to multiple C entries. In tensor form, each
    # rank-1 term has one U row, one V row, one W row pattern.
    # But Strassen uses sums, so each M is ONE rank-1 term in the tensor,
    # and the C entries are linear combinations of M's.
    #
    # The tensor T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # where T[i,j,k] = A[i,j]*B[j,k]
    #
    # For Strassen, each M_r is a product of linear forms in A and B.
    # So U[:,r] encodes the linear form in A, V[:,r] encodes linear form in B,
    # and W[:,r] encodes which C entry gets this product (with coefficient).
    #
    # But a single M contributes to MULTIPLE C entries. In the rank-1 decomposition,
    # each term is rank-1, so W can only have one non-zero entry (or one with coefficient).
    # Wait, no. W[k*n+i, r] is a vector of length p*n = 10. For each rank term r,
    # W[:,r] is a vector. The contribution to C[i,k] is W[k*n+i, r] * (U part) * (V part).
    # So W[:,r] can have multiple non-zero entries!
    
    # Let me redo: for each M, U part is the A-coefficients, V part is B-coefficients,
    # W part has entries for each C[i,k] with the coefficient from Strassen.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper: index for U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i  # W[k*n+i, r] as specified
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C01 += 0, C10 += 0, C11 += 0
    # Wait: C00 = M1+M4-M5-M7, C01=M5+M6, C10=M3+M7, C11=M2-M3+M4+M6
    # M1 contributes: C00 += 1
    U[u_idx(0,0)][col] = 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1  # C00
    col += 1
    
    # M2 = A11*(B00-B10) -> C11 += 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    V[v_idx(1,0)][col] = -1
    W[w_idx(1,1)][col] = 1  # C11
    col += 1
    
    # M3 = A00*(B10+B11) -> C10 += 1, C11 += -1
    U[u_idx(0,0)][col] = 1
    V[v_idx(1,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(1,0)][col] = 1  # C10
    W[w_idx(1,1)][col] = -1 # C11
    col += 1
    
    # M4 = A11*(B01-B11) -> C00 += 1, C11 += 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = -1
    W[w_idx(0,0)][col] = 1  # C00
    W[w_idx(1,1)][col] = 1  # C11
    col += 1
    
    # M5 = (A00+A01)*B00 -> C00 += -1, C01 += 1
    U[u_idx(0,0)][col] = 1
    U[u_idx(0,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    W[w_idx(0,0)][col] = -1 # C00
    W[w_idx(0,1)][col] = 1  # C01
    col += 1
    
    # M6 = (A01-A10)*(B01+B11) -> C01 += 1, C11 += 1
    U[u_idx(0,1)][col] = 1
    U[u_idx(1,0)][col] = -1
    V[v_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,1)][col] = 1  # C01
    W[w_idx(1,1)][col] = 1  # C11
    col += 1
    
    # M7 = (A00-A10)*(B10+B11) -> C00 += -1, C10 += 1
    U[u_idx(0,0)][col] = 1
    U[u_idx(1,0)][col] = -1
    V[v_idx(1,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = -1 # C00
    W[w_idx(1,0)][col] = 1  # C10
    col += 1
    
    # Now naive terms for remaining blocks
    # j in {0,1}, k in {2,3,4}: i in {0,1}
    for i in range(2):
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(i, k)][col] = 1
                col += 1
    
    # j in {2,3,4}, k in {0,1}: i in {0,1}
    for i in range(2):
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(i, k)][col] = 1
                col += 1
    
    # j in {2,3,4}, k in {2,3,4}: i in {0,1}
    for i in range(2):
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(i, k)][col] = 1
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import numpy as np
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor and verify
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    n, m, p = DIMENSIONS
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(U_arr.shape[1]):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            C_computed[i, k] = total
    
    err = np.max(np.abs(C_computed - C_expected))
    print(f"Max error: {err}")
    assert err < 1e-10, "Decomposition is incorrect!"
    print("Verification passed.")
    print(f"Rank: {U_arr.shape[1]}")

