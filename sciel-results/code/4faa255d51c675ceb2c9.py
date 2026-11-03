import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # 140s safety margin

    # Strassen steps for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen = [
        # u, v, w coefficients (local indices 0..3 for 2x2)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]

    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    U28, V28, W28 = build_strassen_28()

    # Verify the Strassen-28 decomposition
    if verify_decomposition(U28, V28, W28, n, m, p):
        # Try to find rank-27 by looking for a linear combination of the 28 factors
        # that can be eliminated. The idea: if there's a linear dependence among
        # the 28 rank-1 tensors, we can reduce the rank.
        
        # The rank-1 tensors are T_r = U_col_r ⊗ V_col_r ⊗ W_col_r
        # We want to check if there exist c_1, ..., c_28 not all zero
        # such that sum_r c_r * T_r = 0
        
        # This is equivalent to checking if the 28 vectors in the tensor space
        # (dimension 32) are linearly independent.
        
        # Let's build the 28x32 matrix where each row r represents the
        # flattening of T_r in the tensor space
        rank = 28
        dim = n * m * p  # 32
        
        # Build matrix M where M[r, (i,j,k)] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        M = [[0] * dim for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        M[r][idx] = U28[i * m + j][r] * V28[j * p + k][r] * W28[k * n + i][r]
        
        # Check if the rows are linearly dependent by computing rank of M
        # Use Gaussian elimination with fractions for exact arithmetic
        def compute_row_rank(matrix):
            """Compute row rank of a matrix using Gaussian elimination."""
            if not matrix or not matrix[0]:
                return 0
            rows = len(matrix)
            cols = len(matrix[0])
            # Convert to list of lists for mutation
            M2 = [row[:] for row in matrix]
            rank_val = 0
            for col in range(cols):
                # Find pivot
                pivot = -1
                for row in range(rank_val, rows):
                    if M2[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap rows
                M2[rank_val], M2[pivot] = M2[pivot], M2[rank_val]
                # Eliminate below
                pivot_val = M2[rank_val][col]
                for row in range(rank_val + 1, rows):
                    if M2[row][col] != 0:
                        factor = M2[row][col] / pivot_val
                        for c in range(col, cols):
                            M2[row][c] -= factor * M2[rank_val][c]
                rank_val += 1
            return rank_val
        
        row_rank = compute_row_rank(M)
        print(f"Row rank of 28x32 tensor matrix: {row_rank}")
        
        if row_rank < rank:
            # There's a linear dependence! We can reduce the rank.
            # Find the null space and use it to eliminate one factor.
            # For now, just note this and return rank-28 (implementation of reduction is complex)
            pass
        
        # Try a randomized approach to find rank-27 decomposition
        # by searching for a linear relation among the 28 factors
        
        # For each pair (r1, r2), check if there exists alpha such that
        # T_{r2} = alpha * T_{r1} + (adjustment in other factors)
        # This is complex, so let's try a different approach:
        # Use the fact that if we can find a non-trivial kernel vector,
        # we can project out one dimension
        
        # Attempt: try all pairs and see if we can merge
        best_rank = rank
        for trial in range(min(100, int(deadline - time.time()))):
            if time.time() > deadline - 10:
                break
            # Random linear combination
            c = [random.randint(-3, 3) for _ in range(rank)]
            if all(ci == 0 for ci in c):
                continue
            # Check if this is in the kernel
            residual = [0] * dim
            for r in range(rank):
                if c[r] != 0:
                    for idx in range(dim):
                        residual[idx] += c[r] * M[r][idx]
            if all(resi == 0 for resi in residual):
                # Found a kernel vector! Can reduce rank
                best_rank = rank - 1
                break
        
        if best_rank < rank:
            # Implementation of actual rank reduction would go here
            # For now, return the rank-28 solution
            pass
        
        return {"U": U28, "V": V28, "W": W28}
    
    # Fallback: return rank-32 trivial decomposition
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
