import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x6 matrix B into three 3x2 blocks.
    # Each block is multiplied by the 2x3 matrix A.
    # The problem reduces to 3 independent 2x3 by 3x2 multiplications.
    # We use a rank 11 decomposition for each 2x3 by 3x2 block.
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    # Strassen decomposition for 2x2 by 2x2 (rank 7)
    # A2 = [[a, b], [c, d]], B2 = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-d)(e+g)
    # C11 = M1 - M5 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M4 + M6
    
    # Trivial decomposition for 2x1 by 1x2 (rank 4)
    # A1 = [[a], [c]], B1 = [[e, f]]
    # C11 = a*e
    # C12 = a*f
    # C21 = c*e
    # C22 = c*f
    
    # For each block b_idx in 0..2:
    for b_idx in range(3):
        # Offset in V and W for this block
        # V index: j * p + k. j in 0..2, k in 0..1 (local) -> k_global = b_idx * 2 + k_local
        # W index: k * n + i. k in 0..5. k_global = b_idx * 2 + k_local
        
        base_col = b_idx * 11
        
        # Part 1: j=0 (2x1 by 1x2), rank 4
        # Terms r = 0..3
        # r=0: A[0,0] * B[0,0] -> C[0,0]
        U[0 * 3 + 0][base_col + 0] = 1
        V[0 * 6 + (b_idx * 2 + 0)][base_col + 0] = 1
        W[(b_idx * 2 + 0) * 2 + 0][base_col + 0] = 1
        
        # r=1: A[0,0] * B[0,1] -> C[0,1]
        U[0 * 3 + 0][base_col + 1] = 1
        V[0 * 6 + (b_idx * 2 + 1)][base_col + 1] = 1
        W[(b_idx * 2 + 1) * 2 + 0][base_col + 1] = 1
        
        # r=2: A[1,0] * B[0,0] -> C[1,0]
        U[1 * 3 + 0][base_col + 2] = 1
        V[0 * 6 + (b_idx * 2 + 0)][base_col + 2] = 1
        W[(b_idx * 2 + 0) * 2 + 1][base_col + 2] = 1
        
        # r=3: A[1,0] * B[0,1] -> C[1,1]
        U[1 * 3 + 0][base_col + 3] = 1
        V[0 * 6 + (b_idx * 2 + 1)][base_col + 3] = 1
        W[(b_idx * 2 + 1) * 2 + 1][base_col + 3] = 1
        
        # Part 2: j=1,2 (2x2 by 2x2), rank 7
        # A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
        # B_sub = [[B[1, k], B[2, k]]] for k in 0,1
        # Local indices for A_sub: i in 0..1, j_local in 0..1 (global j = 1 + j_local)
        # Local indices for B_sub: j_local in 0..1 (global j = 1 + j_local), k_local in 0..1
        
        # Strassen terms
        # M1 = (A[0,1] + A[1,2]) * (B[1,0] + B[2,1])  -- Wait, B is 2x2.
        # B_sub = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
        # M1 = (a+d)(e+h) = (A[0,1] + A[1,2]) * (B[1,0] + B[2,1])? No.
        # e = B[1,0], h = B[2,1].
        
        # Let's map Strassen variables:
        # a = A[0,1], b = A[0,2], c = A[1,1], d = A[1,2]
        # e = B[1,0], f = B[1,1], g = B[2,0], h = B[2,1]
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+h)
        # M7 = (b-d)(e+g)
        
        # C11 = M1 - M5 + M6 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M4 + M6
        
        # We need to add these to U, V, W.
        # Each M_r is a product of a linear combo of A and a linear combo of B.
        # This corresponds to 7 terms in the decomposition.
        
        # Term 4: M1
        # U: a+d -> U[0,1]+=1, U[1,2]+=1
        # V: e+h -> V[1,0]+=1, V[2,1]+=1
        # W: C11 += 1, C22 += 1
        U[0 * 3 + 1][base_col + 4] = 1
        U[1 * 3 + 2][base_col + 4] = 1
        V[1 * 6 + (b_idx * 2 + 0)][base_col + 4] = 1
        V[2 * 6 + (b_idx * 2 + 1)][base_col + 4] = 1
        W[(b_idx * 2 + 0) * 2 + 0][base_col + 4] = 1
        W[(b_idx * 2 + 1) * 2 + 1][base_col + 4] = 1
        
        # Term 5: M2
        # U: c+d -> U[1,1]+=1, U[1,2]+=1
        # V: e -> V[1,0]+=1
        # W: C21 += 1
        U[1 * 3 + 1][base_col + 5] = 1
        U[1 * 3 + 2][base_col + 5] = 1
        V[1 * 6 + (b_idx * 2 + 0)][base_col + 5] = 1
        W[(b_idx * 2 + 0) * 2 + 1][base_col + 5] = 1
        
        # Term 6: M3
        # U: a -> U[0,1]+=1
        # V: f-h -> V[1,1]+=1, V[2,1]+=-1
        # W: C12 += 1
        U[0 * 3 + 1][base_col + 6] = 1
        V[1 * 6 + (b_idx * 2 + 1)][base_col + 6] = 1
        V[2 * 6 + (b_idx * 2 + 1)][base_col + 6] = -1
        W[(b_idx * 2 + 1) * 2 + 0][base_col + 6] = 1
        
        # Term 7: M4
        # U: d -> U[1,2]+=1
        # V: g-e -> V[2,0]+=1, V[1,0]+=-1
        # W: C21 += 1
        U[1 * 3 + 2][base_col + 7] = 1
        V[2 * 6 + (b_idx * 2 + 0)][base_col + 7] = 1
        V[1 * 6 + (b_idx * 2 + 0)][base_col + 7] = -1
        W[(b_idx * 2 + 0) * 2 + 1][base_col + 7] = 1
        
        # Term 8: M5
        # U: a+b -> U[0,1]+=1, U[0,2]+=1
        # V: h -> V[2,1]+=1
        # W: C11 += -1, C12 += 1
        U[0 * 3 + 1][base_col + 8] = 1
        U[0 * 3 + 2][base_col + 8] = 1
        V[2 * 6 + (b_idx * 2 + 1)][base_col + 8] = 1
        W[(b_idx * 2 + 0) * 2 + 0][base_col + 8] = -1
        W[(b_idx * 2 + 1) * 2 + 0][base_col + 8] = 1
        
        # Term 9: M6
        # U: c-d -> U[1,1]+=1, U[1,2]+=-1
        # V: f+h -> V[1,1]+=1, V[2,1]+=1
        # W: C11 += 1, C22 += 1
        U[1 * 3 + 1][base_col + 9] = 1
        U[1 * 3 + 2][base_col + 9] = -1
        V[1 * 6 + (b_idx * 2 + 1)][base_col + 9] = 1
        V[2 * 6 + (b_idx * 2 + 1)][base_col + 9] = 1
        W[(b_idx * 2 + 0) * 2 + 0][base_col + 9] = 1
        W[(b_idx * 2 + 1) * 2 + 1][base_col + 9] = 1
        
        # Term 10: M7
        # U: b-d -> U[0,2]+=1, U[1,2]+=-1
        # V: e+g -> V[1,0]+=1, V[2,0]+=1
        # W: C11 += 1
        U[0 * 3 + 2][base_col + 10] = 1
        U[1 * 3 + 2][base_col + 10] = -1
        V[1 * 6 + (b_idx * 2 + 0)][base_col + 10] = 1
        V[2 * 6 + (b_idx * 2 + 0)][base_col + 10] = 1
        W[(b_idx * 2 + 0) * 2 + 0][base_col + 10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
