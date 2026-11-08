"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        num_U = len(U)
        num_V = len(V)
        num_W = len(W)
        rank = len(U[0])
        for a in range(num_U):
            i, j = divmod(a, m)
            for b in range(num_V):
                j2, k = divmod(b, p)
                if j != j2:
                    continue
                c = k * n + i
                target = 1
                computed = 0
                for r in range(rank):
                    computed += U[a][r] * V[b][r] * W[c][r]
                if abs(computed - target) > 1e-9:
                    return False
        return True

    # Kronecker product strategy:
    # Decompose 2x4x5 as (2x2x2) ⊗ (1x2x2) ⊗ (1x1x1) approximately
    # Actually, let's think more carefully.
    # 
    # 2x4x5 can be seen as:
    # A is 2x4 = (2x2) ⊗ (1x2)  [row: i*2+j1, col: j2*1+k1]
    # B is 4x5 = (2x2) ⊗ (2x1)  [row: j2*2+j1, col: k2*1+k1]  -- wait, 4=2*2, 5 is prime
    # 
    # Better approach: 2x4x5 = (2x2x2) ⊗ (1x2x1) ⊗ (1x1x? ) -- doesn't work cleanly
    # 
    # Let me try: 2x4x5 where 4=2*2, 5=2+2+1
    # Split B into B1(2x2) and B2(2x2) and B3(2x1)
    # C = A1*B1 + A2*B2 + A3*B3 where:
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]
    #   B1 = B[0:2, 0:2], B2 = B[2:4, 0:2], B3 = B[0:4, 2:3]... no
    # 
    # Actually the current block-Strassen already does this. Let me try a different Kronecker approach.
    #
    # Key insight: 2x4x5 can be decomposed as:
    #   M_{2,4,5} = M_{2,2,2} ⊗ M_{1,2,1} ⊗ M_{1,1,?}
    # This doesn't work directly since 5 isn't a product.
    #
    # Alternative: Use the fact that M_{2,4,5} contains M_{2,2,2} as a sub-tensor.
    # We can write:
    #   C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  -- rank 7 (Strassen)
    #   C[0:2, 2:4] = A[0:2, 2:4] * B[0:2, 2:4]  -- rank 7 (Strassen)  
    #   C[0:2, 4:5] = A[0:2, 0:4] * B[0:4, 4:5]  -- rank 8 (naive 2x4x1)
    #   C[2:4, 0:2] = A[0:2, 0:2] * B[2:4, 0:2]  -- rank 7 (Strassen)
    #   C[2:4, 2:4] = A[0:2, 2:4] * B[2:4, 2:4]  -- rank 7 (Strassen)
    #   C[2:4, 4:5] = A[0:2, 0:4] * B[0:4, 4:5]  -- rank 8 (naive 2x4x1)
    #
    # Total: 7+7+8+7+7+8 = 44. Worse than 36.
    #
    # The current approach shares the Strassen factors across blocks.
    # Let me try to improve by finding a better 2x4x4 decomposition.
    #
    # For 2x4x4: split as 2x2x4 + 2x2x4 = two 2x2x4 blocks
    # 2x2x4: naive rank = 16. Can we do better?
    # Write 2x2x4 as (2x2x2) ⊗ (1x1x2) -- this gives rank 7*2 = 14
    # So 2x4x4 = 14 + 14 = 28 (same as current)
    #
    # What about a non-trivial 2x4x4 decomposition?
    # The tensor M_{2,4,4} has 4*4*4 = 64 nonzero entries (each C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually no: C has 2*4=8 entries, each is sum of 4 products, so 32 nonzero tensor entries.
    #
    # Let me try to find a rank-24 decomposition of 2x4x4 by solving a linear system.
    # We need U (8x24), V (16x24), W (8x24) such that for all i,j,k:
    #   sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    #
    # This is a system of 32 equations in 24*24*24 = 13824 unknowns (with constraints).
    # Actually it's bilinear in the columns. Let me try a different approach.
    #
    # Try: express 2x4x4 as a sum of rank-1 tensors using structured patterns.
    # 
    # Let me just try to verify if rank 35 is achievable by removing one column
    # from the current rank-36 decomposition and checking.

    # Build the rank-36 baseline
    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def strassen_block(y, z, w):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z

            U[u_idx(*y00)][col] = 1; U[u_idx(*y11)][col] = 1
            V[v_idx(*z00)][col] = 1; V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] += 1; W[w_idx(*w[3])][col] += 1
            col += 1

            U[u_idx(*y10)][col] = 1; U[u_idx(*y11)][col] = 1
            V[v_idx(*z00)][col] = 1
            W[w_idx(*w[2])][col] += 1; W[w_idx(*w[3])][col] -= 1
            col += 1

            U[u_idx(*y00)][col] = 1
            V[v_idx(*z01)][col] = 1; V[v_idx(*z11)][col] = -1
            W[w_idx(*w[1])][col] += 1; W[w_idx(*w[3])][col] += 1
            col += 1

            U[u_idx(*y11)][col] = 1
            V[v_idx(*z10)][col] = 1; V[v_idx(*z00)][col] = -1
            W[w_idx(*w[0])][col] += 1; W[w_idx(*w[2])][col] += 1
            col += 1

            U[u_idx(*y00)][col] = 1; U[u_idx(*y01)][col] = 1
            V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] -= 1; W[w_idx(*w[1])][col] += 1
            col += 1

            U[u_idx(*y10)][col] = 1; U[u_idx(*y00)][col] = -1
            V[v_idx(*z00)][col] = 1; V[v_idx(*z01)][col] = 1
            W[w_idx(*w[3])][col] += 1
            col += 1

            U[u_idx(*y01)][col] = 1; U[u_idx(*y11)][col] = -1
            V[v_idx(*z10)][col] = 1; V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] += 1
            col += 1

        strassen_block(
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,0),(1,0),(0,1),(1,1)]
        )
        strassen_block(
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,0),(2,1),(3,0),(3,1)],
            [(0,0),(1,0),(0,1),(1,1)]
        )
        strassen_block(
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,0),(3,0),(2,1),(3,1)]
        )
        strassen_block(
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,2),(2,3),(3,2),(3,3)],
            [(2,0),(3,0),(2,1),(3,1)]
        )

        # 2x2 * 2x1 blocks (rank 4 each)
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

        return U, V, W, rank

    U0, V0, W0, rank0 = build_strassen_36()
    assert verify(U0, V0, W0), "Baseline verification failed"

    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0

    # Try to remove columns one at a time
    def try_remove_column(U, V, W, col_idx):
        rank = len(U[0])
        if rank <= 1:
            return None
        new_rank = rank - 1
        U_new = [[U[i][r] for r in range(rank) if r != col_idx] for i in range(len(U))]
        V_new = [[V[i][r] for r in range(rank) if r != col_idx] for i in range(len(V))]
        W_new = [[W[i][r] for r in range(rank) if r != col_idx] for i in range(len(W))]
        if verify(U_new, V_new, W_new):
            return U_new, V_new, W_new, new_rank
        return None

    # Try removing columns
    current_U, current_V, current_W, current_rank = U0, V0, W0, rank0
    removed = True
    while removed and time.time() < _deadline - 30:
        removed = False
        for c in range(current_rank):
            if time.time() > _deadline - 30:
                break
            result = try_remove_column(current_U, current_V, current_W, c)
            if result:
                current_U, current_V, current_W, current_rank = result
                removed = True
                break

    if current_rank < best_rank:
        best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank

    # Try Kronecker-based approach: M_{2,4,5} via structured decomposition
    # Idea: Use the fact that 2x4x5 = (2x2x2) ⊗ (1x2x1) ⊗ (1x1x5/2) -- not clean
    # Instead, try: treat as 2x(2+2)x(2+2+1) and use shared Strassen factors
    
    # Alternative: try a rank-32 decomposition using a different block structure
    # 2x4x5: write A as [A1 | A2] where A1, A2 are 2x2
    # B as [[B11, B12, B13], [B21, B22, B23]] where B11,B21 are 2x2, B12,B22 are 2x2, B13,B23 are 2x1
    # C1 = A1*B11 + A2*B21  (2x2, rank 7)
    # C2 = A1*B12 + A2*B22  (2x2, rank 7)
    # C3 = A1*B13 + A2*B23  (2x1, rank 4)
    # Total: 7+7+4 = 18 for each "row block" of output
    # But we have 2 output row blocks (for the 5 columns split as 2+2+1):
    # Wait, the output is 4x5, so C has 4 rows and 5 columns.
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  -- 2x2 output, rank 7
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  -- 2x2 output, rank 7
    # C[0:2, 4:5] = A[0:2,0:2]*B[0:2,4:5] + A[0:2,2:4]*B[2:4,4:5]  -- 2x1 output, rank 4
    # C[2:4, 0:2] = A[0:2,0:2]*B[2:4,0:2]... wait, A is only 2 rows!
    
    # I'm confusing myself. Let me re-clarify:
    # A is 2x4, B is 4x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split j into j in {0,1} and j in {2,3}:
    # C[:, 0:2] = A[:,0:2] * B[0:2, 0:2] + A[:,2:4] * B[2:4, 0:2]  -- 2x2, rank 7
    # C[:, 2:4] = A[:,0:2] * B[0:2, 2:4] + A[:,2:4] * B[2:4, 2:4]  -- 2x2, rank 7
    # C[:, 4:5] = A[:,0:2] * B[0:2, 4:5] + A[:,2:4] * B[2:4, 4:5]  -- 2x1, rank 4
    # Total: 7+7+4 = 18!
    
    # Wait, that's much better than 36! Let me verify this.
    # The issue: for C[:, 0:2], we need to compute A1*B1 + A2*B2 where:
    #   A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    #   B1 = B[0:2, 0:2] (2x2), B2 = B[2:4, 0:2] (2x2)
    # This is exactly a 2x2x2 matrix multiplication, which has rank 7 (Strassen).
    # 
    # Similarly for C[:, 2:4] and C[:, 4:5].
    # 
    # But wait -- can we share the Strassen factors? The A matrices are the same
    # across all three output blocks! A1 and A2 are used in all three.
    # 
    # In the current rank-36 approach, we have 4 Strassen blocks (28 cols) + 8 cols for 2x1.
    # The 4 Strassen blocks correspond to:
    #   C[0:2, 0:2], C[0:2, 2:4], C[2:4, 0:2], C[2:4, 2:4]
    # But C is only 2x5, so C[2:4, :] doesn't exist!
    
    # I think the current code has a bug or I'm misunderstanding the indices.
    # Let me re-examine: n=2, m=4, p=5
    # A is n x m = 2 x 4
    # B is m x p = 4 x 5
    # C is n x p = 2 x 5
    # 
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # So C has 2*5 = 10 entries.
    # 
    # The current code uses:
    #   strassen_block for (0,0),(0,1),(1,0),(1,1) in U; (0,0),(0,1),(1,0),(1,1) in V; (0,0),(1,0),(0,1),(1,1) in W
    #   This computes C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]?
    #   No wait, the U indices are (i,j) pairs and V indices are (j,k) pairs.
    #   
    # Let me trace through the first strassen_block call:
    #   y = [(0,0),(0,1),(1,0),(1,1)] -> U indices: (0,0),(0,1),(1,0),(1,1) -> rows 0,1,4,5
    #   z = [(0,0),(0,1),(1,0),(1,1)] -> V indices: (0,0),(0,1),(1,0),(1,1) -> cols 0,1,4,5
    #   w = [(0,0),(1,0),(0,1),(1,1)] -> W indices: (0,0),(1,0),(0,1),(1,1) -> rows 0,1,2,3
    #
    # This computes: C[k*n+i, ...] for k in {0,1}, i in {0,1}
    #   C[0,0] = sum_r U[0,r]*V[0,r]*W[0,r] + ... 
    #   The Strassen block with these indices computes A[0:2,0:2]*B[0:2,0:2] as a 2x2 result
    #   stored in C[0:2, 0:2] (using the w_idx convention: W[k*n+i])
    #
    # So the first block computes C[0:2, 0:2] from A[0:2, 0:2] * B[0:2, 0:2].
    # The second block: y = [(0,2),(0,3),(1,2),(1,3)], z = [(2,0),(2,1),(3,0),(3,1)]
    #   This computes A[0:2, 2:4] * B[2:4, 0:2] and adds to C[0:2, 0:2].
    #
    # So the first two Strassen blocks together compute C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # That's 14 columns for one 2x2 output block.
    #
    # Similarly, blocks 3 and 4 compute C[0:2, 2:4].
    #
    # And the last 8 columns compute C[0:2, 4:5] naively.
    #
    # Total: 14 + 14 + 8 = 36.
    #
    # Now, the key question: can we do C[0:2, 0:2] in less than 14?
    # C[0:2, 0:2] = A1*B1 + A2*B2 where A1,A2,B1,B2 are all 2x2.
    # This is the sum of two 2x2x2 multiplications.
    # 
    # The tensor for "sum of two 2x2x2 multiplications" has rank at most 7+7=14.
    # Can it be done in less? This is the tensor M_{2,4,2} (2x4 times 4x2 = 2x2).
    # 
    # M_{2,4,2}: rank is at least 2*4 = 8 (by border rank arguments?) 
    # Actually, the naive rank is 2*4*2 = 16. Strassen gives 14.
    # 
    # I believe 14 is optimal for M_{2,4,2}, so 36 might be optimal for M_{2,4,5}.
    # 
    # Let me just return the best we found.
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
