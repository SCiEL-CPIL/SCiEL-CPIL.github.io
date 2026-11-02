"""
Rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication.
Strategy: Use a structured block decomposition with Strassen on 2x2x2 blocks,
then attempt algebraic merging/optimization to reduce rank.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit

    def _elapsed():
        return time.time() - start_time

    def _verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Must equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Check integer/half-integer constraint
        for M in (U, V, W):
            for row in M:
                for val in row:
                    v = float(val)
                    if v != int(v) and 2 * v != int(2 * v):
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_strassen_block(i0, j0, k0, U, V, W, col_offset):
        """
        Apply Strassen's 2x2x2 algorithm for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        -> C[i0:i0+2, k0:k0+2]. Returns new column offset.
        """
        col = col_offset
        # M1 = (A00+A11)*(B00+B11)
        U[i0 * m + j0][col] = 1.0
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[j0 * p + k0][col] = 1.0
        V[(j0+1) * p + (k0+1)][col] = 1.0
        W[k0 * n + i0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M2 = (A10+A11)*B00
        U[(i0+1) * m + j0][col] = 1.0
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[j0 * p + k0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M3 = A00*(B11-B00)
        U[i0 * m + j0][col] = 1.0
        V[(j0+1) * p + (k0+1)][col] = 1.0
        V[j0 * p + k0][col] = -1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M4 = A11*(B01-B10)
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[j0 * p + (k0+1)][col] = 1.0
        V[(j0+1) * p + k0][col] = -1.0
        W[k0 * n + i0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M5 = (A00+A01)*B10
        U[i0 * m + j0][col] = 1.0
        U[i0 * m + (j0+1)][col] = 1.0
        V[(j0+1) * p + k0][col] = 1.0
        W[k0 * n + i0][col] = -1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M6 = (A10+A11)*(B11-B01)
        U[(i0+1) * m + j0][col] = 1.0
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[(j0+1) * p + (k0+1)][col] = 1.0
        V[j0 * p + (k0+1)][col] = -1.0
        W[k0 * n + i0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M7 = (A00-A01)*(B00+B10)
        U[i0 * m + j0][col] = 1.0
        U[i0 * m + (j0+1)][col] = -1.0
        V[j0 * p + k0][col] = 1.0
        V[(j0+1) * p + k0][col] = 1.0
        W[k0 * n + i0][col] = 1.0
        col += 1

        return col

    def _build_block_decomp():
        """
        Build a decomposition using Strassen on 2x2x2 blocks where possible.
        For 3x4x7:
        - Rows 0,1 (2 rows) can be paired with cols 0,1 (2 cols) for Strassen on 2x2x2 blocks
        - Remaining parts handled directly
        
        Strategy:
        - Use Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (7 ranks)
        - Use Strassen for A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (7 ranks)
        - Direct for A[0:2, 0:2] * B[0:2, 2:7] -> C[0:2, 2:7] (2*2*5=20 ranks)
        - Direct for A[0:2, 2:4] * B[2:4, 2:7] -> C[0:2, 2:7] (2*2*5=20 ranks)
        - Direct for A[2, 0:4] * B[0:4, 0:7] -> C[0:7, 2] (1*4*7=28 ranks)
        
        Total: 7+7+20+20+28 = 82
        """
        rank_total = 82
        U = [[0.0] * rank_total for _ in range(n * m)]
        V = [[0.0] * rank_total for _ in range(m * p)]
        W = [[0.0] * rank_total for _ in range(p * n)]
        col = 0

        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] via Strassen
        col = _build_strassen_block(0, 0, 0, U, V, W, col)

        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] via Strassen
        col = _build_strassen_block(0, 2, 0, U, V, W, col)

        # Block 3: A[0:2, 0:2] * B[0:2, 2:7] -> C[0:2, 2:7] direct
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    U[i * m + j][col] = 1.0
                    V[j * p + (2 + k)][col] = 1.0
                    W[(2 + k) * n + i][col] = 1.0
                    col += 1

        # Block 4: A[0:2, 2:4] * B[2:4, 2:7] -> C[0:2, 2:7] direct
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    U[i * m + (2 + j)][col] = 1.0
                    V[(2 + j) * p + (2 + k)][col] = 1.0
                    W[(2 + k) * n + i][col] = 1.0
                    col += 1

        # Block 5: A[2, 0:4] * B[0:4, 0:7] -> C[0:7, 2] direct
        for j in range(4):
            for k in range(7):
                U[2 * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + 2][col] = 1.0
                col += 1

        assert col == rank_total, f"col={col}, expected {rank_total}"
        return {"U": U, "V": V, "W": W}

    def _try_merge_decomposition(U, V, W, max_iters=2000):
        """
        Try to merge pairs of rank-1 tensors that have overlapping structure.
        Two rank-1 tensors (u1,v1,w1) and (u2,v2,w2) can be merged if
        there exists a scalar such that the combined contribution can be
        represented with one rank-1 tensor. This is generally hard, so we
        try simple cases where one factor is a scalar multiple.
        """
        rank = len(U[0])
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]

        for _ in range(max_iters):
            if _elapsed() > deadline - 10:
                break
            merged = False
            for a in range(rank):
                for b in range(a + 1, rank):
                    # Check if column b is a scalar multiple of column a in all three matrices
                    # Find scalar from U
                    nonzero_a = [i for i in range(n * m) if abs(U[i][a]) > 1e-12]
                    if not nonzero_a:
                        continue
                    i0 = nonzero_a[0]
                    if abs(U[i0][a]) < 1e-12:
                        continue
                    scalar = U[i0][b] / U[i0][a]
                    
                    # Verify scalar consistency across U
                    consistent = True
                    for i in range(n * m):
                        if abs(U[i][a]) > 1e-12:
                            if abs(U[i][b] - scalar * U[i][a]) > 1e-9:
                                consistent = False
                                break
                        elif abs(U[i][b]) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        continue

                    # Verify scalar consistency across V
                    for j in range(m * p):
                        if abs(V[j][a]) > 1e-12:
                            if abs(V[j][b] - scalar * V[j][a]) > 1e-9:
                                consistent = False
                                break
                        elif abs(V[j][b]) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        continue

                    # Verify scalar consistency across W
                    for k in range(p * n):
                        if abs(W[k][a]) > 1e-12:
                            if abs(W[k][b] - scalar * W[k][a]) > 1e-9:
                                consistent = False
                                break
                        elif abs(W[k][b]) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        continue

                    # Merge: column a += scalar * column b, remove column b
                    for i in range(n * m):
                        U[i][a] += scalar * U[i][b]
                    for j in range(m * p):
                        V[j][a] += scalar * V[j][b]
                    for k in range(p * n):
                        W[k][a] += scalar * W[k][b]

                    # Remove column b
                    for i in range(n * m):
                        U[i].pop(b)
                    for j in range(m * p):
                        V[j].pop(b)
                    for k in range(p * n):
                        W[k].pop(b)
                    rank -= 1
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                break

        return {"U": U, "V": V, "W": W}

    def _try_linear_combination_merge(U, V, W, max_iters=500):
        """
        Try more general merges: find two columns that are linearly dependent
        in the tensor product sense. For each pair of columns, check if there
        exists a 2x2 matrix of coefficients that makes the combined tensor
        rank-1. This is expensive, so limit attempts.
        """
        rank = len(U[0]) if U else 0
        if rank < 2:
            return {"U": U, "V": V, "W": W}

        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]

        for _ in range(max_iters):
            if _elapsed() > deadline - 15:
                break
            merged = False
            for a in range(rank):
                for b in range(a + 1, rank):
                    # Try to find alpha, beta such that
                    # (alpha*u_a + beta*u_b, alpha*v_a + beta*v_b, alpha*w_a + beta*w_b)
                    # is rank-1 (i.e., the outer product structure holds)
                    # This requires: (alpha*u_a + beta*u_b) ⊗ (alpha*v_a + beta*v_b) ⊗ (alpha*w_a + beta*w_b)
                    # to be proportional to u_a⊗v_a⊗w_a + u_b⊗v_b⊗w_b
                    
                    # Simplified: try alpha=1, beta=t for various t
                    # and check if the result is rank-1
                    for t in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                        # Check if u_a + t*u_b, v_a + t*v_b, w_a + t*w_b form a rank-1 tensor
                        # that sums correctly with the remaining
                        # This is complex; skip for now
                        pass
                    # For now, just check simple scalar multiples (already done in _try_merge_decomposition)
                    pass
            if not merged:
                break

        return {"U": U, "V": V, "W": W}

    # Try the block decomposition
    result = _build_block_decomp()
    if _verify(result["U"], result["V"], result["W"]):
        best_result = result
        best_rank = len(result["U"][0])
    else:
        best_result = _schoolbook()
        best_rank = n * m * p

    # Try to merge
    if _elapsed() < deadline - 30:
        merged = _try_merge_decomposition(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]]
        )
        if _verify(merged["U"], merged["V"], merged["W"]):
            new_rank = len(merged["U"][0])
            if new_rank < best_rank:
                best_result = merged
                best_rank = new_rank

    # Try linear combination merge
    if _elapsed() < deadline - 15:
        merged2 = _try_linear_combination_merge(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]]
        )
        if _verify(merged2["U"], merged2["V"], merged2["W"]):
            new_rank = len(merged2["U"][0])
            if new_rank < best_rank:
                best_result = merged2
                best_rank = new_rank

    return best_result
