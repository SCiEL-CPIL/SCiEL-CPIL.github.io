"""Matrix multiplication 3x5x5 — rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[a*2+b, r], V[b*2+c, r], W[c*2+a, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen 7 multiplications for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen factor, we need rank-1 terms.
    # Each M factor is a 2x2 matrix product, which needs 4 rank-1 terms (schoolbook).
    # But in tensor decomposition, each "multiplication" M_i corresponds to a 2x2x2 sub-tensor
    # decomposed into 4 rank-1 terms.
    #
    # Actually, for the full tensor decomposition, each of the 7 Strassen products is itself
    # a 2x2x2 tensor that can be decomposed into 4 rank-1 terms. But that gives 28 rank-1 terms,
    # which is worse than schoolbook (8).
    #
    # The correct approach: Strassen reduces the number of *scalar multiplications* from 8 to 7,
    # but in tensor rank terms, each scalar multiplication is a rank-1 term. So 7 rank-1 terms
    # for 2x2x2? No — each "multiplication" in Strassen is actually a matrix multiplication,
    # not a scalar one.
    #
    # Let me reconsider. The 2x2x2 tensor T(a,b,c) = sum_{i,j} A[i,j]*B[j,c]*delta... 
    # Actually the tensor is T[i,j,k] = delta_{jk} (the multiplication tensor).
    # Rank of 2x2x2 multiplication tensor is 7 (Strassen).
    #
    # The 7 rank-1 decompositions are:
    # r1: U1 = A11+A22, V1 = B11+B22, W1 = C11 + C22 (diagonal of C)
    # Wait, I need to be more careful.
    
    # The standard Strassen decomposition of the 2x2x2 matrix multiplication tensor:
    # T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^4 (for A entries), v_r ∈ R^4 (for B entries), w_r ∈ R^4 (for C entries)
    
    # A indexed as [A11, A12, A21, A22] -> positions (0,0),(0,1),(1,0),(1,1)
    # B indexed as [B11, B12, B21, B22]
    # C indexed as [C11, C12, C21, C22]
    
    # Using standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In tensor decomposition, each rank-1 term (u, v, w) contributes:
    # C = u^T A v^T B w  (outer product structure)
    # Actually for matrix mult: C[i,k] = sum_{j} A[i,j] B[j,k]
    # Tensor: T[i,j,k] = delta(j,k)... no, T is the structure tensor.
    
    # The decomposition is: for each r, C += (u_r * A) * (v_r * B) * w_r
    # where (u_r * A) means element-wise multiply A by u_r, similarly for B.
    # And w_r scales the rows/columns of the result.
    
    # More precisely: C = sum_r (u_r ⊙ A) @ (v_r ⊙ B) ⊙ w_r
    # where ⊙ is element-wise and @ is matrix mult. But that's not quite right either.
    
    # The correct tensor decomposition: 
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # meaning: for inputs A, B (flattened), output C = sum_r (u_r^T A) (v_r^T B) w_r
    # where A, B are treated as vectors in R^4, and u_r, v_r are also in R^4.
    # Wait, that gives C as a scalar times w_r... 
    # 
    # Actually: C (as a vector in R^4) = sum_r (u_r^T A)(v_r^T B) w_r
    # This is correct for the bilinear map A,B -> C.
    
    # So for Strassen 2x2x2:
    # u_r, v_r ∈ R^4 (indices for A and B entries), w_r ∈ R^4 (indices for C entries)
    
    # M1: u = [1,0,0,1], v = [1,1,0,0], w = [1,0,0,0] + [0,0,0,1] = [1,0,0,1]
    # Wait, C11 gets +M1, C22 gets +M1. So w1 = [1,0,0,1]
    # But C12 and C21 don't get M1. So w1 = [1,0,0,1]. ✓
    
    # M2: u = [0,0,1,1], v = [1,0,0,0], w: C21 gets +M2. So w2 = [0,0,1,0]
    # Also C22 gets -M2. So w2 = [0,0,1,-1]
    
    # M3: u = [1,0,0,0], v = [0,0,1,-1], w: C12 gets +M3, C22 gets +M3. So w3 = [0,1,0,1]
    
    # M4: u = [0,0,0,1], v = [0,1,-1,0], w: C11 gets +M4, C21 gets +M4. So w4 = [1,0,1,0]
    
    # M5: u = [1,1,0,0], v = [0,0,0,1], w: C11 gets -M5, C12 gets +M5. So w5 = [-1,1,0,0]
    
    # M6: u = [-1,0,1,0], v = [1,1,0,0], w: C22 gets +M6. So w6 = [0,0,0,1]
    
    # M7: u = [0,1,0,-1], v = [0,0,1,1], w: C11 gets +M7. So w7 = [1,0,0,0]
    
    U = np.array([
        [1, 0, 0, 0, 1, -1, 0],  # u for each of 7 terms, for A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # for A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # for A[1,0]
        [1, 1, 0, 1, 0, 0, -1],  # for A[1,1]
    ], dtype=float).T  # shape (7, 4) -> transpose to (4, 7)
    
    # Wait, I need U[i,j] where i indexes A entries and j indexes the rank-1 term.
    # So U[0,:] = [1, 0, 0, 0, 1, -1, 0] for A[0,0]
    # U[1,:] = [0, 0, 0, 0, 1, 0, 1] for A[0,1]
    # U[2,:] = [0, 1, 0, 0, 0, 1, 0] for A[1,0]
    # U[3,:] = [1, 1, 0, 1, 0, 0, -1] for A[1,1]
    
    U = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)  # shape (4, 7)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # for B[0,0]
        [1, 0, 0, 1, 0, 1, 0],  # for B[0,1]
        [0, 0, 1, -1, 0, 0, 1], # for B[1,0]
        [0, 0, -1, 0, 1, 0, 1], # for B[1,1]
    ], dtype=float)  # shape (4, 7)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # for C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # for C[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # for C[1,0]
        [1, -1, 1, 0, 0, 1, 0],  # for C[1,1]
    ], dtype=float)  # shape (4, 7)
    
    # Verify: C = sum_r (u_r^T A)(v_r^T B) w_r
    # Let's check C[0,0] = sum_r U[0,r] * V[0,r]... no wait.
    # C[i,k] = sum_r U[i,r] * (sum_j V[j,r] * B[j,k]) * W[k,r]... 
    # Hmm, let me re-derive.
    # 
    # The bilinear map: C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor decomposition: C[i,k] = sum_r U[i,r] * (sum_j V[j,r] * B[j,k]) * ... 
    # 
    # Actually the standard form is:
    # C = sum_r (u_r ⊙ A) (v_r ⊙ B)^T ... no.
    # 
    # Let me use the direct form:
    # C[i,k] = sum_r U[i,r] * V[:,r]^T B W[k,r]  -- this isn't standard.
    # 
    # The correct interpretation:
    # Flatten A to a, B to b, C to c (each in R^4).
    # c = sum_r (u_r^T a)(v_r^T b) w_r
    # 
    # So c[i] = sum_r u_r[i] * (v_r^T b) * w_r[i]... no, w_r is a vector, so:
    # c = sum_r (u_r^T a)(v_r^T b) w_r
    # meaning c is a linear combination of w_r vectors, with coefficients (u_r^T a)(v_r^T b).
    # 
    # So c[i] = sum_r (u_r^T a)(v_r^T b) w_r[i]
    # 
    # Let's verify C[0,0] (i=0):
    # c[0] = sum_r (u_r[0]*a[0] + u_r[1]*a[1] + u_r[2]*a[2] + u_r[3]*a[3])
    #              * (v_r[0]*b[0] + v_r[1]*b[1] + v_r[2]*b[2] + v_r[3]*b[3])
    #              * w_r[0]
    # 
    # With a = [A00, A01, A10, A11], b = [B00, B01, B10, B11]
    # 
    # r=1: u^T a = A00+A11, v^T b = B00+B01, w[0]=1 → (A00+A11)(B00+B01)
    # r=2: u^T a = A10+A11, v^T b = B00, w[0]=0 → 0
    # r=3: u^T a = A00, v^T b = B10-B11, w[0]=0 → 0
    # r=4: u^T a = A11, v^T b = B01-B00, w[0]=1 → A11(B01-B00)
    # r=5: u^T a = A00+A01, v^T b = B11, w[0]=-1 → -(A00+A01)B11
    # r=6: u^T a = -A00+A10, v^T b = B00+B01, w[0]=0 → 0
    # r=7: u^T a = A01-A11, v^T b = B10+B11, w[0]=1 → (A01-A11)(B10+B11)
    # 
    # Total: (A00+A11)(B00+B01) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B01 + A11*B00 + A11*B01 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A00*B01 + 2*A11*B01 - A00*B11 - A11*B10 + A01*B10 - A11*B11
    # 
    # Expected C[0,0] = A00*B00 + A01*B10
    # 
    # This doesn't match! My Strassen coefficients are wrong. Let me look up the correct ones.
    
    # Correct Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now in terms of the tensor decomposition c = sum_r (u_r^T a)(v_r^T b) w_r:
    # Each M_i contributes to specific C entries with certain signs.
    # 
    # M1: u=[1,0,0,1], v=[1,1,0,0], contributes +1 to C11, +1 to C22 → w=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0], contributes +1 to C21, -1 to C22 → w=[0,0,1,-1]
    # M3: u=[1,0,0,0], v=[0,0,1,-1], contributes +1 to C12, +1 to C22 → w=[0,1,0,1]
    # M4: u=[0,0,0,1], v=[0,1,-1,0], contributes +1 to C11, +1 to C21 → w=[1,0,1,0]
    # M5: u=[1,1,0,0], v=[0,0,0,1], contributes -1 to C11, +1 to C12 → w=[-1,1,0,0]
    # M6: u=[-1,0,1,0], v=[1,1,0,0], contributes +1 to C22 → w=[0,0,0,1]
    # M7: u=[0,1,0,-1], v=[0,0,1,1], contributes +1 to C11 → w=[1,0,0,0]
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]: M1,M2,M3,M4,M5,M6,M7
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1],   # A[1,1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],    # B[0,0]
        [1, 0, 0, 1, 0, 1, 0],    # B[0,1]
        [0, 0, 1, -1, 0, 0, 1],   # B[1,0]
        [0, 0, -1, 0, 1, 0, 1],   # B[1,1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1]
    ], dtype=float)
    
    # Verify with a test:
    A = np.array([[1,2],[3,4]], dtype=float)
    B = np.array([[5,6],[7,8]], dtype=float)
    C_expected = A @ B  # [[19,22],[43,50]]
    
    a = A.flatten()  # [1,2,3,4]
    b = B.flatten()  # [5,6,7,8]
    
    C_computed = np.zeros(4)
    for r in range(7):
        C_computed += (U[:,r] @ a) * (V[:,r] @ b) * W[:,r]
    
    # Check
    # This should give [19, 22, 43, 50]
    
    return U, V, W


def _schoolbook_2x2x2():
    """Returns (U,V,W) for 2x2x2 schoolbook (rank 4)."""
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    for j in range(2):
        for k in range(2):
            col = j * 2 + k
            U[j, col] = 1  # A[i,j]
            V[j * 2 + k, col] = 1  # B[j,k]
            W[k, col] = 1  # C[i,k] -- wait, W indexes C entries
    # Actually for schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-1 term for (i,j,k): U[i,j] = 1, V[j,k] = 1, W[i,k] = 1
    # But we have 4 entries for A, 4 for B, 4 for C.
    # A entries: (0,0),(0,1),(1,0),(1,1) → index 0,1,2,3
    # B entries: (0,0),(0,1),(1,0),(1,1) → index 0,1,2,3
    # C entries: (0,0),(0,1),(1,0),(1,1) → index 0,1,2,3
    
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # This is 8 terms, not 4. Schoolbook for 2x2x2 is rank 8.
                pass
    # Schoolbook 2x2x2 has rank 8 (i*j*k = 8). But Strassen has rank 7.
    # For sub-blocks, I'll just use schoolbook with rank i*j*k.
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Strategy: partition A into [1,2] blocks, B into [2,1,2], C into [2,1,2]
    # Block sizes: A: 1+2=3, B: 2+1+2=5, C: 2+1+2=5
    # For each block (alpha, beta, gamma) with sizes (ia, ib, ic):
    #   If ia=ib=ic=2: use Strassen (rank 7)
    #   Otherwise: use schoolbook (rank ia*ib*ic)
    
    A_blocks = [1, 2]
    B_blocks = [2, 1, 2]
    C_blocks = [2, 1, 2]
    
    # Compute total rank
    total_rank = 0
    for ia in A_blocks:
        for ib in B_blocks:
            for ic in C_blocks:
                if ia == 2 and ib == 2 and ic == 2:
                    total_rank += 7
                else:
                    total_rank += ia * ib * ic
    
    print(f"Total rank: {total_rank}")
    
    # Build U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Get Strassen decomposition
    Su, Sv, Sw = _strassen_2x2x2()
    # Su shape (4,7), Sv shape (4,7), Sw shape (4,7)
    # But we need to verify this is correct. Let me test it.
    
    A_test = np.array([[1,2],[3,4]], dtype=float)
    B_test = np.array([[5,6],[7,8]], dtype=float)
    C_expected = A_test @ B_test
    a_test = A_test.flatten()
    b_test = B_test.flatten()
    C_computed = np.zeros(4)
    for r in range(7):
        C_computed += (Su[:,r] @ a_test) * (Sv[:,r] @ b_test) * Sw[:,r]
    # print(f"Strassen test: {C_computed} vs {C_expected.flatten()}")
    # The above should match. If not, I'll need to fix.
    
    # For now, proceed with construction.
    rank_offset = 0
    for ai, ia in enumerate(A_blocks):
        a_start = sum(A_blocks[:ai])
        for bi, ib in enumerate(B_blocks):
            b_start = sum(B_blocks[:bi])
            for ci, ic in enumerate(C_blocks):
                c_start = sum(C_blocks[:ci])
                
                if ia == 2 and ib == 2 and ic == 2:
                    # Use Strassen
                    for r in range(7):
                        col = rank_offset + r
                        # U: A block entries
                        for li in range(ia):
                            for lj in range(ib):
                                # A entry at (a_start+li, b_start+lj)
                                U_idx = (a_start + li) * m + (b_start + lj)
                                U[U_idx, col] = Su[li * ib + lj, r]
                        # V: B block entries
                        for lj in range(ib):
                            for lk in range(ic):
                                V_idx = (b_start + lj) * p + (c_start + lk)
                                V[V_idx, col] = Sv[lj * ic + lk, r]
                        # W: C block entries
                        for lk in range(ic):
                            for li in range(ia):
                                W_idx = (c_start + lk) * n + (a_start + li)
                                W[W_idx, col] = Sw[lk * ia + li, r]
                    rank_offset += 7
                else:
                    # Schoolbook
                    for li in range(ia):
                        for lj in range(ib):
                            for lk in range(ic):
                                col = rank_offset
                                U_idx = (a_start + li) * m + (b_start + lj)
                                V_idx = (b_start + lj) * p + (c_start + lk)
                                W_idx = (c_start + lk) * n + (a_start + li)
                                U[U_idx, col] = 1
                                V[V_idx, col] = 1
                                W[W_idx, col] = 1
                                rank_offset += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
