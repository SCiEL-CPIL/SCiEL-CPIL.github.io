"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 decomposition of 3x3 matrix multiplication using a block strategy:
    # - 2x2 blocks multiplied via Strassen (rank 7)
    # - 2x1 and 1x2 blocks via direct (rank 4 each)
    # - 1x2 x 2x1 dot product (rank 1)
    # - 1x1 cross terms (rank 4)
    # - 1x1 diagonal (rank 1)
    # Total: 7 + 4 + 4 + 1 + 4 + 1 = 21... wait, let me recompute.
    #
    # Actually, let's use a cleaner known approach:
    # Decompose 3x3 = [2x2 | 2x1; 1x2 | 1x1]
    # Let A = [[A22, A21], [A12, A11]], B = [[B22, B21], [B12, B11]]
    # where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1.
    # C = A*B = [[A22*B22 + A21*B12, A22*B21 + A21*B11],
    #            [A12*B22 + A11*B12, A12*B21 + A11*B11]]
    #
    # Terms:
    # 1. A22*B22: 2x2*2x2, rank 7 (Strassen)
    # 2. A21*B12: (2x1)*(1x2) = 2x2 outer product, rank 4 (each element is a single product)
    #    Wait, (2x1)*(1x2) gives a 2x2 matrix where C[i,j] = A21[i]*B12[j]. That's 4 terms.
    # 3. A22*B21: (2x2)*(2x1) = 2x1, rank 4 (each of 2 rows does 2 products)
    #    Actually for 2x2 times 2x1: result[i] = sum_j A22[i,j]*B21[j]. That's 2*2=4 products.
    # 4. A21*B11: (2x1)*(1x1) = 2x1, rank 2 (A21[0]*B11, A21[1]*B11)
    # 5. A12*B22: (1x2)*(2x2) = 1x2, rank 4 (result[j] = sum_i A12[i]*B22[i,j], so 2*2=4)
    # 6. A11*B12: (1x1)*(1x2) = 1x2, rank 2 (A11*B12[0], A11*B11... wait B12 is 1x2)
    #    A11 * B12 where B12 is 1x2: result is 1x2, each element is A11*B12[j]. Rank 2.
    # 7. A12*B21: (1x2)*(2x1) = 1x1, rank 2 (dot product: A12[0]*B21[0] + A12[1]*B21[1])
    #    But as a tensor decomposition, this is a sum of 2 rank-1 terms.
    # 8. A11*B11: (1x1)*(1x1) = 1x1, rank 1.
    #
    # Total rank = 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    
    import numpy as np
    
    n, m, p = DIMENSIONS  # 3, 3, 3
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 9 x 26
    V = [[0] * rank for _ in range(m * p)]  # 9 x 26
    W = [[0] * rank for _ in range(p * n)]  # 9 x 26, W[k*n + i, r]
    
    col = 0
    
    def add_term(U_idx, V_idx, W_idx, val=1):
        nonlocal col
        U[U_idx][col] = val
        V[V_idx][col] = val
        W[W_idx][col] = val
        col += 1
    
    # Indexing: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[i,k] -> W[k*n+i]
    # For 3x3: i,j,k in {0,1,2}
    # A22 = A[0:2, 0:2], A21 = A[0:2, 2], A12 = A[2, 0:2], A11 = A[2,2]
    # B22 = B[0:2, 0:2], B21 = B[0:2, 2], B12 = B[2, 0:2], B11 = B[2,2]
    
    # 1. A22 * B22 using Strassen (rank 7)
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A22 indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B22 indices: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[3], B[1,1]=V[4]
    # C22 output: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4]
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0)   # A00*B00 -> C00
    add_term(0, 1, 1)   # A00*B11 -> C01... wait this isn't right for Strassen
    # Actually Strassen combines, so we need to be more careful.
    # Let me just use the direct 4-term for A22*B22 to keep it simple, then add other terms.
    # Actually, let's stick with the plan: use Strassen for the 2x2 block.
    
    # Reset and do this more carefully. Let me implement each term explicitly.
    
    # Clear U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 1. A22 * B22 via Strassen (7 terms)
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    # C00=C[0,0], C01=C[0,1], C10=C[1,0], C11=C[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    # This expands to A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But in tensor decomposition, we can't have sums in a single rank-1 term.
    # So Strassen's 7 multiplications each produce a scalar, and then we add/subtract.
    # The key insight: each of the 7 "multiplications" is actually a single rank-1 term
    # in the decomposition because the coefficients are ±1.
    # 
    # M1 contributes: +A00*B00 to C00, +A00*B11 to C01, +A11*B00 to C10, +A11*B11 to C11
    # But wait, in the Strassen algorithm, M1 is ONE multiplication that feeds into multiple
    # output entries. In tensor decomposition terms, this means we have 4 rank-1 terms
    # that all share the same "intermediate" value... 
    #
    # Actually, the standard way to view Strassen as a tensor decomposition:
    # Each of the 7 multiplications corresponds to 4 rank-1 terms (one for each output
    # entry that uses that multiplication), but with signs.
    # No wait, that's not right either.
    #
    # Let me think again. The matrix multiplication tensor for 2x2 has rank 7.
    # This means there exist 7 triples (u_r, v_r, w_r) such that:
    # sum_r u_r ⊗ v_r ⊗ w_r = matmul_2x2
    # where u_r is a 4-dim vector (flattened 2x2), etc.
    #
    # The 7 terms of Strassen give exactly this. Each term r has:
    # u_r[i,j] = coefficient of A[i,j] in the r-th Strassen multiplication
    # v_r[j,k] = coefficient of B[j,k] in the r-th Strassen multiplication
    # w_r[i,k] = contribution to C[i,k] from the r-th multiplication
    #
    # Let me look up the exact Strassen tensor decomposition for 2x2.
    # 
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   u1 = [1, 0, 0, 1] (A00+A11)
    #   v1 = [1, 0, 0, 1] (B00+B11)
    #   w1 = [1, 0, 0, 1] (contributes +1 to C00, C01, C10, C11)
    #   But wait, M1 contributes: C00 += M1, C01 += M1, C10 += M1, C11 += M1
    #   So w1 = [1, 1, 1, 1]? No...
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So the w vectors are:
    # w1 = [1, 0, 0, 1] (M1 appears in C00 and C11)
    # w2 = [0, 0, 1, -1] (M2 appears in C10 with +1 and C11 with -1)
    # w3 = [0, 1, 0, 1] (M3 appears in C01 with +1 and C11 with +1)
    # w4 = [1, 0, 0, 1] (M4 appears in C00 with +1 and C10 with +1)
    # w5 = [0, 1, 0, -1] (M5 appears in C01 with +1 and C00 with -1)
    # w6 = [0, 0, 0, 1] (M6 appears in C11 with +1)
    # w7 = [1, 0, 0, 0] (M7 appears in C00 with +1)
    #
    # And u, v vectors:
    # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1]
    # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0]
    # u3 = [1, 0, 0, 0], v3 = [0, 0, 1, -1]
    # u4 = [0, 0, 0, 1], v4 = [0, 1, -1, 0]
    # u5 = [1, 1, 0, 0], v5 = [0, 0, 0, 1]
    # u6 = [-1, 0, 1, 0], v6 = [1, 0, 1, 0]
    # u7 = [0, 1, 0, -1], v7 = [0, 1, 0, 1]
    #
    # Now I can embed these into the 3x3 case.
    
    # For the 3x3 case, A22 occupies rows 0,1 and cols 0,1 of A.
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B22: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[3], B[1,1]=V[4]
    # C22: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4]
    
    # Strassen term 1: u1=[1,0,0,1] on A22, v1=[1,0,0,1] on B22, w1=[1,0,0,1] on C22
    add_term(0, 0, 0)   # A00*B00 -> C00
    add_term(4, 0, 0)   # A11*B00 -> C00  ... wait, this is getting complicated.
    
    # Let me just directly build the rank-26 decomposition using the block method
    # with explicit rank-1 terms, avoiding the complexity of embedding Strassen.
    
    # Fresh start with direct block decomposition:
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 1. A22 * B22: 4 terms (direct, not Strassen, to keep it simple first)
    # Actually, let's use Strassen properly. I'll add 7 terms.
    
    # Strassen for A22*B22 (indices in 3x3 grid):
    # A22: (0,0),(0,1),(1,0),(1,1) -> U indices: 0,1,3,4
    # B22: (0,0),(0,1),(1,0),(1,1) -> V indices: 0,1,3,4
    # C22: (0,0),(0,1),(1,0),(1,1) -> W indices: 0,1,3,4
    
    # Term 1: (A00+A11)(B00+B11) -> contributes to C00, C11
    # u: A00 and A11, v: B00 and B11, w: C00 and C11
    # This is 4 rank-1 sub-terms, but they're all part of one Strassen multiplication.
    # In the tensor decomposition, each "multiplication" M is a scalar, and then
    # M is added to various C entries. So:
    # M1 * 1 * 1 contributes to C00: coefficient +1
    # M1 * 1 * 1 contributes to C11: coefficient +1
    # But M1 = (A00+A11)(B00+B11) is not a single product of one A entry and one B entry.
    #
    # The correct tensor decomposition view:
    # The 2x2 matmul tensor = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are 4-vectors.
    # When embedding in 3x3, u_r is a 9-vector (only 4 nonzero), etc.
    
    # u1 = [1,0,0,1] (positions 0,1,3,4 in 3x3: A[0,0], A[0,1], A[1,0], A[1,1])
    # So u1 has 1 at U[0] and U[4]
    # v1 has 1 at V[0] and V[4]
    # w1 has 1 at W[0] and W[4] (C[0,0] and C[1,1])
    
    # But wait, u1 ⊗ v1 ⊗ w1 would give terms like:
    # u1[0]*v1[0]*w1[0] -> A[0,0]*B[0,0]*C[0,0] coefficient 1
    # u1[0]*v1[4]*w1[0] -> A[0,0]*B[1,1]*C[0,0] coefficient 1
    # u1[4]*v1[0]*w1[0] -> A[1,1]*B[0,0]*C[0,0] coefficient 1
    # u1[4]*v1[4]*w1[0] -> A[1,1]*B[1,1]*C[0,0] coefficient 1
    # And similarly for w1[4] (C[1,1])
    # 
    # But in actual matmul, C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # So A[0,0]*B[1,1] should NOT contribute to C[0,0].
    # This means the outer product u1 ⊗ v1 ⊗ w1 is NOT a valid term.
    #
    # I was confusing myself. The Strassen algorithm's 7 multiplications are
    # intermediate values, not rank-1 tensor terms. The rank-1 tensor terms
    # are the individual products A[i,j]*B[j,k] that sum to C[i,k].
    #
    # For 2x2 matmul, the tensor rank is 7, meaning we need 7 terms of the form
    # sum_i u_r[i]*A[i] * sum_j v_r[j]*B[j] * sum_k w_r[k]*C[k]
    # where each term is a rank-1 tensor (outer product of three vectors).
    #
    # The 7 terms for Strassen are NOT simply the 7 multiplications. They are
    # specific rank-1 decompositions. Let me just use a known rank-7 decomposition
    # of the 2x2 matmul tensor.
    
    # Known rank-7 decomposition of 2x2 matmul (Strassen):
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # Term 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
    # Term 3: u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,1]
    # Term 4: u=[0,0,0,1], v=[0,1,-1,0], w=[1,0,0,1]
    # Term 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,-1]
    # Term 6: u=[-1,0,1,0], v=[1,0,1,0], w=[0,0,0,1]
    # Term 7: u=[0,1,0,-1], v=[0,1,0,1], w=[1,0,0,0]
    
    # Let me verify term 1: u1⊗v1⊗w1
    # Nonzero when u1[i]!=0, v1[j]!=0, w1[k]!=0
    # i in {0,3}, j in {0,3}, k in {0,3} (using 0,1,3,4 for 3x3 indices of A22)
    # For 2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # u1=[1,0,0,1] means A[0,0] and A[1,1]
    # v1=[1,0,0,1] means B[0,0] and B[1,1]
    # w1=[1,0,0,1] means C[0,0] and C[1,1]
    # Products: A[0,0]B[0,0]C[0,0], A[0,0]B[1,1]C[0,0], A[1,1]B[0,0]C[0,0], A[1,1]B[1,1]C[0,0]
    # and same for C[1,1]
    # But C[0,0] should only get A[0,0]B[0,0] + A[0,1]B[1,0]
    # So A[0,0]B[1,1] should NOT be in C[0,0].
    # This decomposition is WRONG.
    
    # I think I'm confusing the tensor rank decomposition with the Strassen algorithm.
    # Let me just use a verified rank-7 decomposition.
    
    # Actually, the simplest approach: just use rank 27 (schoolbook) and accept score 1.0,
    # OR find a known lower-rank decomposition.
    
    # For now, let me implement the schoolbook (rank 27) to ensure correctness,
    # and then try to improve.
    
    U = [[0] * 27 for _ in range(9)]
    V = [[0] * 27 for _ in range(9)]
    W = [[0] * 27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
