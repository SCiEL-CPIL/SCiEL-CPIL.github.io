import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 5

    # Strategy: Split j into {0,1} and {2}.
    # Part 1: j in {0,1}. Dimensions 2x2x5.
    #   Split k into {0,1}, {2}, {3,4}.
    #   k in {0,1}: 2x2x2 -> Laderman rank 7
    #   k = 2:       2x2x1 -> Schoolbook rank 4
    #   k in {3,4}:  2x2x2 -> Laderman rank 7
    #   Total for Part 1: 7 + 4 + 7 = 18
    # Part 2: j=2. Dimensions 2x1x5. Rank 1 (outer product).
    # Total Rank = 18 + 1 = 19.

    rank = 19
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # --- Part 2: j=2 (Rank 1) ---
    r = 0
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1

    # --- Part 1: j in {0,1} ---
    # Laderman 2x2x2 decomposition
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # The 7 terms of Laderman:
    # T1: U=(1,0), V=(0,1), W=(1,0)  -> i'=0, j'=1, k'=0
    # T2: U=(0,1), V=(1,0), W=(0,1)  -> i'=1, j'=0, k'=1
    # T3: U=(1,1), V=(1,1), W=(1,1)  -> i'=0,1, j'=0,1, k'=0,1
    # T4: U=(1,0), V=(1,0), W=(0,0)  -> i'=0, j'=0, k'=0
    # T5: U=(0,1), V=(0,1), W=(1,1)  -> i'=1, j'=1, k'=0,1
    # T6: U=(1,-1), V=(1,1), W=(0,1) -> i'=0,1, j'=0,1, k'=1
    # T7: U=(1,1), V=(1,-1), W=(1,0) -> i'=0,1, j'=0,1, k'=0

    # Actually, let me use the standard Laderman coefficients:
    # For C = A*B where A is 2x2, B is 2x2, C is 2x2:
    # C00 = M1 + M4 - M6 - M7
    # C01 = M5 - M4
    # C10 = M6 - M5
    # C11 = M2 + M7 - M1 - M3
    # where:
    # M1 = (A10)(B01)
    # M2 = (A11)(B10)
    # M3 = (A10+A11)(B10-B11)
    # M4 = (A00)(B00)
    # M5 = (A11)(B11)
    # M6 = (A00+A10)(B01+B11)
    # M7 = (A01+A11)(B00-B10)

    # In tensor form, for local 2x2x2:
    # U has 7 columns, V has 7 columns, W has 7 columns.
    # U[i', r], V[j', r], W[k', r]

    laderman_U = np.array([
        [1, 0, 0, 1, 0, 1, 1],   # i'=0
        [0, 1, 0, 0, 1, 1, 1],   # i'=1
    ])
    laderman_V = np.array([
        [0, 1, 1, 1, 0, 1, -1],  # j'=0
        [1, 0, -1, 0, 1, 1, 1],  # j'=1
    ])
    laderman_W = np.array([
        [1, 0, 0, 0, 1, 0, 1],   # k'=0
        [0, 1, 1, 0, 1, 1, 0],   # k'=1
    ])

    # Wait, let me re-derive this more carefully.
    # The tensor T[i', j', k'] = 1 if i'=j' and j'=k' ... no.
    # The matrix multiplication tensor: T[i, j, k] = 1 if the output C[i,k] gets contribution from A[i,j]*B[j,k].
    # So T[i', j', k'] = 1 for all i', j', k' in {0,1} where the product term exists.
    # Actually, the tensor is: T[i', j', k'] = delta(i', j') * delta(j', k') ... no.
    # The matrix multiplication tensor M[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k].
    # So M[i', j', k'] = 1 for all (i', j', k') where i' and k' are output indices and j' is the summed index.
    # For 2x2x2: M[i', j', k'] = 1 for all 8 combinations.
    # Wait, no. The tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is A*B = C, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor has entries T[i,j,k] = 1 for all i,j,k.
    # The rank of this tensor is 7 (Laderman).

    # Let me just use a known correct Laderman decomposition.
    # From various sources, the 7 rank-1 tensors are:
    # (using 0-based indexing for i,j,k in {0,1}):
    # r1: i=0, j=0, k=0  -> U[0,0]=1, V[0,0]=1, W[0,0]=1
    # But that's schoolbook. Laderman uses linear combinations.

    # Let me use this verified set:
    # M1 = A[1,0]*B[0,1]  -> U[1,0]=1, V[0,1]=1, W[0,1]=1  (i=1, j=0, k=1)
    # M2 = A[1,1]*B[1,0]  -> U[1,1]=1, V[1,0]=1, W[1,0]=1  (i=1, j=1, k=0)
    # M3 = (A[1,0]+A[1,1])*(B[1,0]-B[1,1]) -> U[1,0]=1, U[1,1]=1, V[1,0]=1, V[1,1]=-1, W[1,0]=1
    # M4 = A[0,0]*B[0,0]  -> U[0,0]=1, V[0,0]=1, W[0,0]=1
    # M5 = A[1,1]*B[1,1]  -> U[1,1]=1, V[1,1]=1, W[1,1]=1
    # M6 = (A[0,0]+A[1,0])*(B[0,1]+B[1,1]) -> U[0,0]=1, U[1,0]=1, V[0,1]=1, V[1,1]=1, W[0,1]=1
    # M7 = (A[0,1]+A[1,1])*(B[0,0]-B[1,0]) -> U[0,1]=1, U[1,1]=1, V[0,0]=1, V[1,0]=-1, W[0,0]=1

    # So:
    # r=0 (M1): U[1,0]=1, V[0,1]=1, W[0,1]=1
    # r=1 (M2): U[1,1]=1, V[1,0]=1, W[1,0]=1
    # r=2 (M3): U[1,0]=1, U[1,1]=1, V[1,0]=1, V[1,1]=-1, W[1,0]=1
    # r=3 (M4): U[0,0]=1, V[0,0]=1, W[0,0]=1
    # r=4 (M5): U[1,1]=1, V[1,1]=1, W[1,1]=1
    # r=5 (M6): U[0,0]=1, U[1,0]=1, V[0,1]=1, V[1,1]=1, W[0,1]=1
    # r=6 (M7): U[0,1]=1, U[1,1]=1, V[0,0]=1, V[1,0]=-1, W[0,0]=1

    laderman_U = np.zeros((2, 7))
    laderman_V = np.zeros((2, 7))
    laderman_W = np.zeros((2, 7))

    # M1
    laderman_U[1, 0] = 1
    laderman_V[0, 0] = 1
    laderman_W[0, 0] = 1

    # M2
    laderman_U[1, 1] = 1
    laderman_V[1, 1] = 1
    laderman_W[1, 1] = 1

    # M3
    laderman_U[1, 2] = 1
    laderman_U[1, 2] += 1  # Wait, U[1,2] should be 1 for A[1,0] and 1 for A[1,1]
    # Actually U[1,2] = 1 (for A[1,0]) + 1 (for A[1,1]) = 2? No.
    # M3 = (A[1,0]+A[1,1])*(B[1,0]-B[1,1])
    # U[i', r] is the coefficient of A[i', j'] in the r-th term.
    # For M3: U[1,0] contributes 1, U[1,1] contributes 1. So U[1,2] = 1 for j'=0, U[1,2]... 
    # Wait, U is indexed by [i', r]. For each r, U[i', r] is the coefficient.
    # M3 involves A[1,0] and A[1,1]. So U[1, 2] should have entries for both j' values?
    # No, U[i', r] is a single number. The tensor term is U[i',r] * V[j',r] * W[k',r].
    # For M3: sum over i',j',k' of U[i',2]*V[j',2]*W[k',2] * A[i',j']*B[j',k']
    # We want this to equal (A[1,0]+A[1,1])*(B[1,0]-B[1,1])
    # = A[1,0]*B[1,0] - A[1,0]*B[1,1] + A[1,1]*B[1,0] - A[1,1]*B[1,1]
    # So: U[1,2]*V[1,2]*W[1,0] = 1  (for A[1,0]*B[1,0])
    #     U[1,2]*V[1,2]*W[1,1] = -1 (for A[1,0]*B[1,1])
    #     U[1,2]*V[1,2]*W[1,0] = 1  (for A[1,1]*B[1,0]) -- wait, this is the same term
    #     U[1,2]*V[1,2]*W[1,1] = -1 (for A[1,1]*B[1,1])
    # This doesn't work because U[1,2] is a single value.

    # I think I'm confusing myself. Let me use a different approach.
    # The Laderman decomposition gives 7 scalars M1..M7 such that:
    # C[0,0] = M1 + M4 - M6 - M7
    # C[0,1] = M5 - M4
    # C[1,0] = M6 - M5
    # C[1,1] = M2 + M7 - M1 - M3
    # where M1..M7 are bilinear forms in A and B.

    # In tensor form, each Mr is a rank-1 tensor (or sum of rank-1 tensors).
    # M1 = A[1,0]*B[0,1] -> rank 1: U[1,0]=1, V[0,1]=1, W[0,1]=1
    # M2 = A[1,1]*B[1,0] -> rank 1: U[1,1]=1, V[1,0]=1, W[1,0]=1
    # M3 = (A[1,0]+A[1,1])*(B[1,0]-B[1,1]) -> This is rank 2 as a tensor.
    #   = A[1,0]*B[1,0] - A[1,0]*B[1,1] + A[1,1]*B[1,0] - A[1,1]*B[1,1]
    #   So it contributes to 4 different (i,j,k) positions.
    #   As a tensor: T[i,j,k] for this term:
    #   T[1,1,0] += 1, T[1,1,1] += -1, T[1,1,0] += 1, T[1,1,1] += -1
    #   Wait, that gives T[1,1,0] += 2, T[1,1,1] += -2. That's not right.

    # I think the issue is that M3 is not a single rank-1 tensor but a sum.
    # For the tensor decomposition, we need to express the full tensor as a sum of rank-1 tensors.
    # Laderman shows it can be done with 7 rank-1 tensors.

    # Let me just use a known correct implementation.
    # I'll use the following (verified):
    # r=0: U=[0,1], V=[0,1], W=[1,0]  -> A[1,0]*B[0,1] -> C[1,0]... wait
    # This is getting too complex. Let me just use schoolbook for all of Part 1.
    # Rank for 2x2x5 schoolbook = 2*2*5 = 20.
    # Total rank = 20 + 1 = 21. Same as before.

    # Actually, let me try a different split.
    # Split k into {0}, {1,2,3,4}.
    # k=0: 2x2x1 -> rank 4
    # k in {1,2,3,4}: 2x2x4 -> can we do better than schoolbook (rank 16)?
    # 2x2x4: not sure of the minimal rank.

    # Let me just go with the safe approach: schoolbook for everything.
    # Rank = 2*3*5 = 30. But we can do 21 as in the current program.

    # Let me stick with the current program's approach: rank 21.
    # But let me verify it's correct.

    # Actually, I realize I should just keep the current program as-is since it's already at score 1.0.
    # The task is to improve it, but if I can't find a valid lower-rank decomposition,
    # I should keep the known-correct one.

    # Let me implement the rank-21 solution cleanly.

    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # --- Part 2: j=2 (Rank 1) ---
    r = 0
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1

    # --- Part 1: j in {0,1} (Rank 20, Schoolbook) ---
    r_offset = 1
    idx = 0
    for i in range(n):
        for j in range(2):
            for k in range(p):
                r = r_offset + idx
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                idx += 1

    # Verify
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1

    if not np.allclose(T_reconstructed, T_expected, atol=1e-10):
        # Fallback to naive rank 30
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive))
        V_naive = np.zeros((m * p, rank_naive))
        W_naive = np.zeros((p * n, rank_naive))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_naive[ii * m + jj, col] = 1
                    V_naive[jj * p + kk, col] = 1
                    W_naive[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
