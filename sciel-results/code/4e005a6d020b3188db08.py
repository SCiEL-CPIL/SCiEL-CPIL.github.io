"""Matrix multiplication 4x4x6 - tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 115  # safety margin
    
    # Strategy: Block decomposition with shared Strassen terms.
    # A (4x4) split into 2x2 blocks: A[i1][j1] for i1,j1 in {0,1}
    # B (4x6) split into 2x2 blocks: B[j1][k1] for j1 in {0,1}, k1 in {0,1,2}
    # C (4x6) split into 2x2 blocks: C[i1][k1] for i1 in {0,1}, k1 in {0,1,2}
    #
    # C[i1][k1] = A[i1][0]*B[0][k1] + A[i1][1]*B[1][k1]
    #
    # For each fixed i1, we need to compute A[i1][0]*B[0][k1] + A[i1][1]*B[1][k1] for k1=0,1,2
    # This is: X * P_k1 + Y * Q_k1 where X=A[i1][0], Y=A[i1][1], P_k1=B[0][k1], Q_k1=B[1][k1]
    #
    # Using Strassen-like sharing: for computing X*P + Y*Q for multiple (P,Q) pairs,
    # we can precompute combinations of X and Y.
    #
    # Key idea: Use 7 Strassen-style products for the 2x2x2 case, but share them
    # across the 3 column-blocks.
    #
    # For a single 2x2 * 2x2 product, Strassen uses 7 terms.
    # For X*P + Y*Q, we can use at most 14 terms (7+7), but with sharing we can do better.
    #
    # Actually, let's use a direct approach:
    # X*P + Y*Q can be computed with 8 rank-1 tensors (schoolbook for each 2x2 product).
    # But we can do better: 
    #   M1 = (X+Y)(P+Q) = X*P + X*Q + Y*P + Y*Q
    #   M2 = (X-Y)(P-Q) = X*P - X*Q - Y*P + Y*Q
    #   M1 + M2 = 2*X*P + 2*Y*Q
    #   So X*P + Y*Q = (M1 + M2)/2
    # That's only 2 "products" but each product is a 2x2*2x2 = 4 rank-1 tensors.
    # Total: 8 rank-1 tensors. Same as schoolbook!
    #
    # Better: Use the full Strassen for each individual product but share A-side combos.
    # 
    # Let me just use a clean block approach with rank 48 (6 blocks * 8) and see if I can reduce.
    #
    # Actually, let me try: for each i1 (2 choices), compute all three C[i1][k1] blocks.
    # C[i1][k1] = A[i1][0]*B[0][k1] + A[i1][1]*B[1][k1]
    #
    # Define for each i1:
    #   S1 = A[i1][0] + A[i1][1]  (2x2 matrix)
    #   S2 = A[i1][0] - A[i1][1]  (2x2 matrix)
    #   T1_k1 = B[0][k1] + B[1][k1]  (2x2 matrix)
    #   T2_k1 = B[0][k1] - B[1][k1]  (2x2 matrix)
    #
    # Then: A[i1][0]*B[0][k1] + A[i1][1]*B[1][k1] = (S1*T1_k1 + S2*T2_k1)/2
    #
    # So C[i1][k1] = (S1*T1_k1 + S2*T2_k1)/2
    #
    # Each S1*T1_k1 is a 2x2*2x2 product = 4 rank-1 tensors (schoolbook)
    # Each S2*T2_k1 is a 2x2*2x2 product = 4 rank-1 tensors
    # Total per (i1, k1): 8 rank-1 tensors
    # Total: 2 * 3 * 8 = 48 rank-1 tensors
    #
    # But we can use Strassen for each 2x2*2x2 product to get 7 instead of 4?
    # No wait - Strassen gives 7 scalar multiplications for a 2x2 product, which means
    # the 2x2x2 tensor has rank 7. So each 2x2*2x2 product needs 7 rank-1 tensors.
    # Total: 2 * 3 * 2 * 7 = 84. That's worse!
    #
    # The issue: Strassen's 7 is for computing the full 2x2 product. In our tensor
    # decomposition, a "rank-1 tensor" corresponds to one scalar multiplication.
    # So a 2x2*2x2 product indeed needs 7 rank-1 tensors (not 4).
    #
    # Wait, I think I'm confusing myself. Let me be very precise.
    #
    # The tensor T for 2x2x2 matrix mult has rank 7 (Strassen). This means:
    # C[a,c] = sum_{r=1}^{7} (sum_{b} A[a,b]*u_r[a,b]) * (sum_{b} B[b,c]*v_r[b,c]) * w_r[a,c]
    #
    # So each 2x2*2x2 product = 7 rank-1 tensors in our decomposition.
    #
    # For our problem: 6 C-blocks, each is sum of 2 such products = 14 rank-1 tensors each.
    # Total: 6 * 14 = 84. This is what the current program does.
    #
    # To do better, we need to share rank-1 tensors across blocks.
    #
    # Key insight: The A-side combinations can be shared!
    # For i1=0: we use A[0][0] and A[0][1]
    # For i1=1: we use A[1][0] and A[1][1]
    # These are different, so no sharing between i1 values.
    #
    # But within a fixed i1, the three k1 values share the same A blocks.
    # Can we share Strassen terms across k1 values?
    #
    # For fixed i1, C[i1][k1] = A[i1][0]*B[0][k1] + A[i1][1]*B[1][k1]
    # Let X = A[i1][0], Y = A[i1][1], P_k1 = B[0][k1], Q_k1 = B[1][k1]
    #
    # Using the S1, S2, T1, T2 approach:
    # C[i1][k1] = (S1*T1_k1 + S2*T2_k1)/2
    #
    # S1*T1_k1 is a 2x2*2x2 product. If we use Strassen for it, we get 7 terms.
    # But S1 is fixed for all k1! The Strassen decomposition of S1*T1_k1 uses
    # linear combinations of S1's entries. Since S1 is the same for all k1,
    # the A-side linear forms are the same!
    #
    # This means we can share the A-side structure. In tensor rank terms,
    # we still need separate rank-1 tensors for each (S1*T1_k1) product,
    # but the U vectors are the same across k1 for a given Strassen term.
    #
    # Actually, in the tensor decomposition, each rank-1 tensor is independent.
    # Sharing doesn't reduce the rank unless we can combine terms.
    #
    # Let me try a different approach: direct rank-1 construction.
    #
    # For the full 4x4x6 problem, let me try to find a decomposition with rank < 84.
    #
    # Lower bound: The 4x4x6 multiplication tensor has at least max(16, 24, 24) = 24 rank.
    # Known results suggest rank around 30-40 for such rectangular cases.
    #
    # Let me try a construction based on:
    # 1. Split B into B_left (4x4) and B_right (4x2)
    # 2. For A*B_left (4x4x4): use rank-7 Strassen -> but wait, that's for the 
    #    full 4x4 product, which needs 7*4=28 rank-1 tensors? No...
    #
    # I think the confusion is: Strassen's rank-7 result for 2x2x2 means the 2x2x2
    # multiplication tensor has rank 7. For 4x4x4, by block Strassen, the rank is 
    # at most 7*4 = 28 (using Strassen on 2x2 blocks, each block product needs 7 terms).
    # But the actual rank of 4x4x4 is known to be at most 23 (from literature).
    #
    # For simplicity, let me just implement the block Strassen approach cleanly
    # and verify it works. Rank 84 is a valid upper bound.
    
    rank_terms = []  # list of (u_dict, v_dict, w_dict)
    
    for i1 in range(2):
        for k1 in range(3):
            # C[i1][k1] = A[i1][0]*B[0][k1] + A[i1][1]*B[1][k1]
            # Use Strassen for each 2x2*2x2 product
            
            # First product: A[i1][0] * B[0][k1]
            # Strassen terms for X * Y (2x2 * 2x2):
            # X = A[i1][0], Y = B[0][k1]
            # M1 = (X00+X11)(Y00+Y11)
            # M2 = (X10+X11)Y00
            # M3 = X00(Y01-Y11)
            # M4 = X11(Y10-Y00)
            # M5 = (X00+X01)Y11
            # M6 = (X10-X00)(Y00+Y01)
            # M7 = (X01-X11)(Y10+Y11)
            # Z00 = M1 + M4 - M5 + M7
            # Z01 = M3 + M5
            # Z10 = M2 + M4
            # Z11 = M1 - M2 + M3 + M6
            
            x_base = i1 * 2  # row offset in A for block [i1][0]
            y_base = 0 * 2    # col offset in A / row offset in B for block [0][k1]
            z_base = k1 * 2  # col offset in B / col offset in C
            
            # A[i1][0] has rows x_base, x_base+1 and cols 0,1
            # B[0][k1] has rows 0,1 and cols z_base, z_base+1
            
            # Strassen for X * Y:
            terms_prod1 = [
                # M1
                ({(x_base, 0): 1, (x_base+1, 1): 1},
                 {(0, z_base): 1, (1, z_base+1): 1},
                 {(0, z_base): 1, (1, z_base+1): 1}),
                # M2
                ({(x_base+1, 0): 1, (x_base+1, 1): 1},
                 {(0, z_base): 1},
                 {(1, z_base): 1, (1, z_base+1): -1}),
                # M3
                ({(x_base, 0): 1},
                 {(0, z_base+1): 1, (1, z_base+1): -1},
                 {(0, z_base+1): 1, (1, z_base+1): 1}),
                # M4
                ({(x_base+1, 1): 1},
                 {(1, z_base): 1, (0, z_base): -1},
                 {(0, z_base): 1, (1, z_base): 1}),
                # M5
                ({(x_base, 0): 1, (x_base, 1): 1},
                 {(1, z_base+1): 1},
                 {(0, z_base): -1, (0, z_base+1): 1}),
                # M6
                ({(x_base+1, 0): 1, (x_base, 0): -1},
                 {(0, z_base): 1, (0, z_base+1): 1},
                 {(1, z_base+1): 1}),
                # M7
                ({(x_base, 1): 1, (x_base+1, 1): -1},
                 {(1, z_base): 1, (1, z_base+1): 1},
                 {(0, z_base): 1}),
            ]
            
            for u_d, v_d, w_d in terms_prod1:
                rank_terms.append((u_d, v_d, w_d))
            
            # Second product: A[i1][1] * B[1][k1]
            # X = A[i1][1], Y = B[1][k1]
            # A[i1][1] has rows x_base, x_base+1 and cols 2,3
            # B[1][k1] has rows 2,3 and cols z_base, z_base+1
            
            x_base2 = i1 * 2
            y_base2 = 1 * 2
            
            terms_prod2 = [
                # M1
                ({(x_base2, 2): 1, (x_base2+1, 3): 1},
                 {(2, z_base): 1, (3, z_base+1): 1},
                 {(0, z_base): 1, (1, z_base+1): 1}),
                # M2
                ({(x_base2+1, 2): 1, (x_base2+1, 3): 1},
                 {(2, z_base): 1},
                 {(1, z_base): 1, (1, z_base+1): -1}),
                # M3
                ({(x_base2, 2): 1},
                 {(2, z_base+1): 1, (3, z_base+1): -1},
                 {(0, z_base+1): 1, (1, z_base+1): 1}),
                # M4
                ({(x_base2+1, 3): 1},
                 {(3, z_base): 1, (2, z_base): -1},
                 {(0, z_base): 1, (1, z_base): 1}),
                # M5
                ({(x_base2, 2): 1, (x_base2, 3): 1},
                 {(3, z_base+1): 1},
                 {(0, z_base): -1, (0, z_base+1): 1}),
                # M6
                ({(x_base2+1, 2): 1, (x_base2, 2): -1},
                 {(2, z_base): 1, (2, z_base+1): 1},
                 {(1, z_base+1): 1}),
                # M7
                ({(x_base2, 3): 1, (x_base2+1, 3): -1},
                 {(3, z_base): 1, (3, z_base+1): 1},
                 {(0, z_base): 1}),
            ]
            
            for u_d, v_d, w_d in terms_prod2:
                rank_terms.append((u_d, v_d, w_d))
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][r] = coeff
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    U_mat = np.array(U, dtype=float).T
    V_mat = np.array(V, dtype=float).T
    W_mat = np.array(W, dtype=float).T
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U_mat[r].reshape(n, m)
        v_vec = V_mat[r].reshape(m, p)
        w_vec = W_mat[r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                c = 0.0
                for j in range(m):
                    c += u_vec[i, j] * v_vec[j, k]
                C_computed[i, k] += c * w_vec[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook rank
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
# EVOLVE-BLOCK-END
