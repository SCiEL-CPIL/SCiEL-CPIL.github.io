"""Matrix multiplication 2x3 * 3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 110  # safety margin under 120s

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 8)
        C_expected = A @ B
        C_computed = np.zeros((2, 8))
        for c in range(U_np.shape[1]):
            U_mat = U_np[:, c].reshape(2, 3)
            V_mat = V_np[:, c].reshape(3, 8)
            W_vec = W_np[:, c]
            prod = U_mat @ V_mat
            for i in range(2):
                for k in range(8):
                    C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
        return np.allclose(C_computed, C_expected, atol=1e-9)

    # Kronecker product decomposition:
    # Split middle dim 3 into (2+1).
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x8) where B is 2x4 blocks of 2x2.
    #   Outer 2x4 mult: rank 7 (Strassen). Inner 2x2 mult: rank 4 (standard).
    #   Kronecker rank: 7*4 = 28.
    # Part 2: A[:,2] (2x1) * B[2,:] (1x8) where B is 1x4 blocks of 2x2.
    #   Each block: outer product 2x2, rank 1. 4 blocks: rank 4.
    # Total rank: 28 + 4 = 32.
    rank = 32
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]

    # Part 1: Kronecker of Strassen-2x4 (rank 7) with standard-2x2 (rank 4)
    # Strassen for 2x4: A is 2x2, B is 2x4 (4 columns), C is 2x4.
    # 7 multiplications M1..M7, each scalar. For each M, we do 2x2 standard mult (4 terms).
    # Total: 7*4 = 28 terms.
    
    # Strassen 2x2 * 2x4:
    # A = [[a00, a01], [a10, a11]] (2x2)
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]] (2x4)
    # C = A * B (2x4)
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b01+b11)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b01-b11)
    # M7 = (a01-a11)*(b00+b10)
    # C00 = M1 - M3
    # C01 = M4 + M5
    # C02 = M1 - M3  (wait, this isn't right for 2x4)
    
    # Actually for 2x4, we need a proper Strassen-like algorithm. Let me use
    # the fact that 2x4 = two 2x2 multiplications sharing the A matrix.
    # C[:,0:2] = A * B[:,0:2] (rank 7 via Strassen)
    # C[:,2:4] = A * B[:,2:4] (rank 7 via Strassen)
    # But we can share some computations. However, for simplicity and correctness,
    # let's just use two independent Strassen 2x2x2 (rank 7 each) = rank 14.
    # Then Kronecker with 2x2 standard (rank 4): 14*4 = 56. That's worse.
    
    # Better: use the Kronecker product structure properly.
    # The 2x4 multiplication tensor T_{2x4} has rank 7 (Strassen).
    # The 2x2 multiplication tensor T_{2x2} has rank 4 (standard).
    # T_{2x4} ⊗ T_{2x2} has rank 7*4 = 28.
    # This corresponds to: A (2x2) * B (2x4 with 2x2 blocks) = C (2x4 with 2x2 blocks)
    # i.e., A (2x2) * B (2x8) = C (2x8) where B and C are viewed as 2x4 of 2x2 blocks.
    
    # Let me implement this by composing: for each of the 7 Strassen multipliers
    # in the 2x4 algorithm, we multiply a scalar by a 2x2 matrix (inner product).
    # Wait, no. The Kronecker product means:
    # For each Strassen term M_j (scalar multiplier in outer 2x4),
    # we need to compute M_j * (inner 2x2 product).
    # But M_j is already a product of (linear combo of A) * (linear combo of B_outer).
    # The inner part is: each "entry" of A_outer is a 2x2 matrix (actually just a scalar here since A is 2x2),
    # and each "entry" of B_outer is a 2x2 matrix (a 2x2 block of B).
    
    # Hmm, let me reconsider. The Kronecker product of tensors:
    # T_outer: (2x2) x (2x4) -> (2x4), rank 7
    # T_inner: (2x2) x (2x2) -> (2x2), rank 4
    # T_outer ⊗ T_inner: (2x2 ⊗ 2x2) x (2x4 ⊗ 2x2) -> (2x4 ⊗ 2x2), rank 7*4=28
    # (2x2 ⊗ 2x2) = 4x4, but our A is only 2x2, so we're using a sub-tensor.
    
    # Actually, I think the correct interpretation is:
    # A is 2x2 (no inner structure, so inner dim is 1x1 effectively)
    # B is 2x4 of 2x2 blocks = 2x8
    # C is 2x4 of 2x2 blocks = 2x8
    # So it's (2x2) x (2x8) -> (2x8)
    # The "outer" structure is 2x4 (treating each 2x2 block as scalar)
    # The "inner" structure is 2x2 (the block multiplication)
    # But A doesn't have inner structure! A is just 2x2 scalars.
    # So the inner multiplication is: scalar * (2x2 matrix) = 2x2 matrix.
    # That's just scaling, rank 1 per block.
    
    # I think I'm overcomplicating this. Let me just implement the 2x4 Strassen
    # directly with 2x2 blocks in B and C.
    
    # For 2x4 Strassen (A is 2x2, B is 2x4, C is 2x4):
    # M1 = (A00+A11)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01+B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B01-B11)
    # M7 = (A01-A11)*(B00+B10)
    # C00 = M1 - M3
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    # But this is for 2x2! For 2x4, we need a different algorithm.
    
    # Let me use a different approach: decompose 2x4 as 2x2 + 2x2 with shared A.
    # C[:,0:2] = A * B[:,0:2] : rank 7 (Strassen)
    # C[:,2:4] = A * B[:,2:4] : rank 7 (Strassen)
    # Total: rank 14 for the outer 2x4 part.
    # With 2x2 blocks in B and C: each scalar becomes a 2x2 block.
    # So each of the 14 terms involves a 2x2 matrix multiplication.
    # Rank: 14 * 4 = 56. Too high.
    
    # The key insight from the directive: use rank-7 for outer 2x4 (not 14).
    # This requires a true rank-7 algorithm for 2x4 matrix multiplication.
    # Such an algorithm exists (it's the Strassen algorithm applied to 2x4).
    
    # Let me look up the rank-7 algorithm for 2x4:
    # A (2x2) * B (2x4) = C (2x4)
    # M1 = (A00+A11)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01+B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B01-B11)
    # M7 = (A01-A11)*(B00+B10)
    # C00 = M1 - M3
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    # Wait, this only gives C as 2x2, not 2x4.
    
    # I think the rank-7 for 2x4 uses a different set of multipliers.
    # Let me just use two Strassen 2x2 (rank 7 each) = rank 14 for the outer part,
    # and accept rank 14*4 + 4 = 60 total. That's worse than 36.
    
    # Alternative: the directive might mean something different. Let me re-read.
    # "treating the 3x8 multiplication as a 2x4 matrix multiplication where each entry is a 2x2 block"
    # So the "2x4" has entries that are 2x2 blocks.
    # The 2x4 matrix is: [[B[0,0:2], B[0,2:4], B[0,4:6], B[0,6:8]],
    #                      [B[1,0:2], B[1,2:4], B[1,4:6], B[1,6:8]]]
    # Each entry is a 2x2 block.
    # A is 2x2 (scalars, no block structure).
    # C is 2x4 of 2x2 blocks = 2x8.
    
    # So it's: (2x2 scalars) * (2x4 of 2x2 blocks) = (2x4 of 2x2 blocks)
    # The outer multiplication is 2x4 (scalar), rank 7.
    # The inner operation is: scalar * 2x2 block = 2x2 block (just scaling, rank 1).
    # Total rank: 7 * 1 = 7? That can't be right either.
    
    # I think the correct interpretation is that we're using the Kronecker product
    # of the 2x4 tensor (rank 7) with the 2x2 tensor (rank 4), giving rank 28.
    # This corresponds to: A is 2x2x2 (but we only use 2x2x1), B is 2x4x2, C is 2x4x2.
    # Since A has no inner structure, we're effectively computing:
    # For each of the 7 outer multipliers, we do a 2x2 matrix multiply (rank 4).
    # But the outer multiplier already involves products of A entries and B block entries.
    
    # Let me just implement it concretely. For the 2x4 outer structure:
    # A = [[a0, a1], [a2, a3]] (indices: A[0,0]=a0, A[0,1]=a1, A[1,0]=a2, A[1,1]=a3)
    # B_outer = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bij is a 2x2 block
    # C_outer = A * B_outer (2x4 of 2x2 blocks)
    
    # Strassen for 2x4 (rank 7):
    # M1 = (a0+a3)*(B00+B10)
    # M2 = (a2+a3)*B00
    # M3 = a0*(B10-B00)
    # M4 = a3*(B01+B11)
    # M5 = (a0+a1)*B11
    # M6 = (a2-a0)*(B01-B11)
    # M7 = (a1-a3)*(B00+B10)
    # C00 = M1 - M3
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    # But this only covers 2x2 output! We need 2x4.
    
    # I realize now that a rank-7 algorithm for 2x4 doesn't simply extend Strassen.
    # The rank of 2x4 matrix multiplication is actually 14 (two independent 2x2 Strassen).
    # So the directive's claim of rank 28 = 7*4 might be incorrect, or I'm misunderstanding.
    
    # Let me just stick with the current rank-36 solution and try to verify it works.
    # Actually, let me try a different approach: use the existing rank-36 code but verify it.
    
    # Re-implementing the rank-36 construction:
    rank = 36
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]

    def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
        U[a_r0 * 3 + b_c0][base] += 1; V[b_c0 * 8 + b_k0][base] += 1
        U[a_r1 * 3 + b_c1][base] += 1; V[b_c1 * 8 + b_k0 + 1][base] += 1
        W[b_k0 * 2 + a_r0][base] += 1
        W[(b_k0 + 1) * 2 + a_r1][base] += 1
        U[a_r1 * 3 + b_c0][base + 1] += 1; V[b_c0 * 8 + b_k0][base + 1] += 1
        U[a_r1 * 3 + b_c1][base + 1] += 1
        W[b_k0 * 2 + a_r1][base + 1] += 1
        U[a_r0 * 3 + b_c0][base + 2] += 1; V[b_c1 * 8 + b_k0][base + 2] += 1
        V[b_c0 * 8 + b_k0][base + 2] += -1
        W[b_k0 * 2 + a_r0][base + 2] += -1
        W[b_k0 * 2 + a_r1][base + 2] += 1
        U[a_r1 * 3 + b_c1][base + 3] += 1; V[b_c0 * 8 + b_k0 + 1][base + 3] += 1
        V[b_c1 * 8 + b_k0 + 1][base + 3] += 1
        W[b_k0 * 2 + a_r0][base + 3] += 1
        W[(b_k0 + 1) * 2 + a_r0][base + 3] += 1
        U[a_r0 * 3 + b_c0][base + 4] += 1; V[b_c1 * 8 + b_k0 + 1][base + 4] += 1
        U[a_r0 * 3 + b_c1][base + 4] += 1
        W[(b_k0 + 1) * 2 + a_r0][base + 4] += 1
        U[a_r1 * 3 + b_c0][base + 5] += 1; V[b_c0 * 8 + b_k0 + 1][base + 5] += 1
        U[a_r0 * 3 + b_c0][base + 5] += -1
        V[b_c1 * 8 + b_k0 + 1][base + 5] += -1
        W[(b_k0 + 1) * 2 + a_r1][base + 5] += 1
        U[a_r0 * 3 + b_c1][base + 6] += 1; V[b_c0 * 8 + b_k0][base + 6] += 1
        U[a_r1 * 3 + b_c1][base + 6] += -1
        V[b_c1 * 8 + b_k0][base + 6] += 1
        W[b_k0 * 2 + a_r0][base + 6] += -1
        W[(b_k0 + 1) * 2 + a_r1][base + 6] += -1

    for half in range(2):
        bk = half * 4
        base = half * 18
        add_strassen_2x2x2(base, 0, 1, 0, 1, bk)
        add_strassen_2x2x2(base + 7, 0, 1, 0, 1, bk + 2)
        for dk in range(4):
            c = base + 14 + dk
            U[0 * 3 + 2][c] += 1
            U[1 * 3 + 2][c] += 1
            V[2 * 8 + bk + dk][c] += 1
            W[(bk + dk) * 2 + 0][c] += 1
            W[(bk + dk) * 2 + 1][c] += 1

    if time.time() > deadline:
        return schoolbook()

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    return schoolbook()
# EVOLVE-BLOCK-END
