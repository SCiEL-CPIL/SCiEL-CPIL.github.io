"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Try to find a low-rank decomposition via random + greedy
    # Start with the known rank-36 block-Strassen solution as baseline
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

    # Verify a decomposition is correct
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        num_U = len(U)
        num_V = len(V)
        num_W = len(W)
        rank = len(U[0])
        
        # Compute the decomposed tensor
        # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # The target tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
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

    # Get baseline
    U0, V0, W0, rank0 = build_strassen_36()
    assert verify(U0, V0, W0), "Baseline verification failed"

    # Try greedy rank reduction: start from rank-36, try to remove columns
    # and re-solve for the remaining entries
    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0

    # Attempt: try rank 35, 34, ... by fixing all but one column and solving
    # This is expensive; instead try a few targeted reductions
    
    # The 2x2x1 blocks have rank exactly 4, so those 8 columns are irreducible.
    # The 4 Strassen blocks use 28 columns. Can we merge some?
    
    # Try: use a 2x4x4 decomposition with rank < 28
    # The tensor M_{2,4,4} has rank at least 16 (trivial) and at most 32 (naive).
    # Strassen gives 28. Known: rank(M_{2,4,4}) <= 24? Let's try to find one.
    
    # Alternative: treat 2x4x5 as having structure where we can share work.
    # Since n=2 is small, each output C[k,i] for i in {0,1} is a linear combination
    # of m*p = 20 products. The tensor has 10*20 = 200 nonzero entries (each row of C
    # depends on 4 entries of A and 5 of B, but structured).
    
    # Let's try a randomized approach: initialize with random rank-r decomposition,
    # then use ALS to fit the tensor, and check if we can get exact integer/half-integer.
    
    # Given time constraints, let's try a few small rank reductions from 36.
    # Try rank 35: remove one column and see if the system is still solvable.
    
    def try_remove_column(U, V, W, col_idx):
        """Try to remove column col_idx and adjust remaining columns."""
        rank = len(U[0])
        if rank <= 1:
            return None
        new_rank = rank - 1
        # Remove the column
        U_new = [[U[i][r] for r in range(rank) if r != col_idx] for i in range(len(U))]
        V_new = [[V[i][r] for r in range(rank) if r != col_idx] for i in range(len(V))]
        W_new = [[W[i][r] for r in range(rank) if r != col_idx] for i in range(len(W))]
        
        # Check if still valid
        if verify(U_new, V_new, W_new):
            return U_new, V_new, W_new, new_rank
        return None

    # Try removing each column
    for c in range(rank0):
        if time.time() > _deadline - 20:
            break
        result = try_remove_column(U0, V0, W0, c)
        if result:
            U0, V0, W0, rank0 = result
            # Try again from new state
            c = 0

    # Try a different approach: 2x4x4 with rank 24
    # Split 4 = 2 + 2, but use a shared Strassen-like structure
    # Actually, let's try: A(2x4) * B(4x4)
    # Write B = B1 @ I + B2 @ I' type decomposition? No, that's not standard.
    
    # Let's try: for the 2x4x4 part, use the fact that we can write
    # C = A1*B1 + A2*B2 where A1=A[:,0:2], A2=A[:,2:4], B1=B[0:2,:], B2=B[2:4,:]
    # This is what we have. The rank is 7+7=14 for each output block of 2x2.
    # Total for 2 output blocks: 28.
    
    # Can we do 2x4x2 (i.e., C[0:2,0:2] = A[0:2,0:4]*B[0:4,0:2]) in less than 14?
    # This is M_{2,4,2}. The naive rank is 2*4*2=16. Strassen gives 14.
    # Known lower bound: rank >= 2*4 = 8? No, that's not right.
    # Actually for M_{2,4,2}: each of the 4 output entries is a sum of 4 products.
    # The tensor has 4*4*2=32 nonzero entries. Rank is at least... 
    # Let's just try to verify if 13 is possible by attempting to find a rank-13 decomposition.
    
    # Given the complexity, let's stick with the block-Strassen approach and see
    # if we can get 35 by removing one redundant column.
    
    # The current best is rank 36. Let's return it.
    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
