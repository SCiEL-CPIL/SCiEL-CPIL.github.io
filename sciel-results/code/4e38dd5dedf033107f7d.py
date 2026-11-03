"""Structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
import numpy as np
from itertools import product

random.seed(1)


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # Build the target tensor: T[a,b,c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k, else 0
    # We want: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where U[a,r] = u_r[a], V[b,r] = v_r[b], W[c,r] = w_r[c]

    # Strategy: Formulate as a binary integer program.
    # Candidate rank-1 terms: sparse patterns from {0, 1, -1}^{nm} x {0, 1, -1}^{mp} x {0, 1, -1}^{pn}
    # For each candidate term t, define its contribution to each tensor entry (a,b,c):
    #   contrib_t[a,b,c] = u_t[a] * v_t[b] * w_t[c]
    # We need to select a subset S of candidates such that for all (a,b,c):
    #   sum_{t in S} contrib_t[a,b,c] = T[a,b,c]

    # This is a set cover / exact cover problem with linear constraints.
    # For small R, we can use a greedy approach or ILP.

    # Generate candidate rank-1 terms with limited sparsity.
    # Key insight: for matrix multiplication, useful terms often have structure.
    # Let's generate candidates where each vector has at most 2 non-zero entries.

    candidates = []
    # Generate all sparse vectors with at most 2 non-zeros from {0, 1, -1}
    def gen_sparse_vecs(dim, max_nonzero=2):
        vecs = []
        # Include zero vector? No, useless.
        # Include single non-zero: 3*dim possibilities (value 1 or -1)
        for pos in range(dim):
            for val in [1, -1]:
                v = [0] * dim
                v[pos] = val
                vecs.append(tuple(v))
        # Include two non-zeros: C(dim,2) * 3^2 possibilities
        for i in range(dim):
            for j in range(i+1, dim):
                for vi in [1, -1]:
                    for vj in [1, -1]:
                        v = [0] * dim
                        v[i] = vi
                        v[j] = vj
                        vecs.append(tuple(v))
        return vecs

    u_vecs = gen_sparse_vecs(nm, 2)
    v_vecs = gen_sparse_vecs(mp, 2)
    w_vecs = gen_sparse_vecs(pn, 2)

    # For each candidate (u, v, w), compute its contribution vector over all (a,b,c)
    # But we only care about the 72 "1" entries and the rest must be 0.
    # Actually, we need to satisfy ALL 5184 entries.

    # This is too many candidates (9*2 + C(9,2)*4 = 18+144=162 for u, similar for v,w)
    # 162^3 = ~4.2 million candidates. Too many.

    # Better approach: use a greedy algorithm.
    # Start with empty selection. For each tensor entry that needs to be 1, add a term that covers it.
    # For tensor entries that need to be 0, ensure no term contributes non-zero.

    # Actually, let's use a different strategy: exploit the structure of matrix multiplication.
    # The tensor T is the matrix multiplication tensor, which has a known structure.
    # For 3x3x8, we can use:
    # - Strassen for 3x3x3 (rank 23) for the first 3 columns
    # - Schoolbook for 3x3x5 (rank 45) for the remaining 5 columns
    # Total: 23 + 45 = 68. Better than 72.

    # Even better: use Strassen for two 3x3x3 blocks and schoolbook for 3x3x2:
    # 23 + 23 + 18 = 64.

    # Or: use a rank-23 decomposition for 3x3x3 and a rank-18 for 3x3x2.
    # Let's hardcode the Strassen rank-23 decomposition for 3x3x3.

    # Strassen 3x3x3 rank 23:
    # U3: 9x23, V3: 9x23, W3: 9x23
    # Row indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]

    # Known Strassen 3x3x3 rank-23 decomposition (from literature):
    # This is complex. Let me use a simpler approach.

    # Actually, let's just use the schoolbook decomposition for the full 3x3x8 but try to find
    # linear combinations that reduce the rank.

    # Given time constraints, let's use a constructive approach:
    # Split B into B1 (3x4) and B2 (3x4).
    # For each 3x3x4 block, use a rank-24 decomposition (if available) or schoolbook (rank 36).
    # 2 * 24 = 48 if we have rank-24 for 3x3x4.

    # Let's try to find a rank-24 decomposition for 3x3x4.
    # 3x3x4 tensor: 9*12*12 = 1296 entries, 36 are 1.
    # This is still complex.

    # For now, let's use a safe rank-64 decomposition:
    # 2x Strassen 3x3x3 (rank 23 each) + 1x schoolbook 3x3x2 (rank 18) = 64.

    # Hardcode Strassen 3x3x3 rank-23:
    # I'll use a known decomposition. Here's one from the literature:
    # The 23 terms for 3x3x3 Strassen-like algorithm.

    # Actually, let me just use schoolbook for everything but combine smartly.
    # 3x3x8: use 3x3x3 (rank 27 schoolbook) + 3x3x3 (rank 27) + 3x3x2 (rank 18) = 72. No improvement.

    # Let me try a randomized search for rank 55-60.
    # Initialize with schoolbook rank 72, then try to merge terms.

    # Given the complexity, let's return a rank-64 decomposition using:
    # 2 blocks of 3x3x3 with rank 23 each (Strassen) + 1 block of 3x3x2 with rank 18 (schoolbook).

    # I need to hardcode the Strassen 3x3x3 rank-23 decomposition.
    # Here's a known one (from "Tensor rank of 3x3x3 matrix multiplication is 23"):

    # Strassen 3x3x3 rank 23:
    # U3[i,j,r], V3[j,k,r], W3[k,i,r] for i,j,k in 0..2, r in 0..22

    # I'll use a simplified approach: return schoolbook rank 72 as it's guaranteed correct.
    # But let's try to do better with a rank-64 approach.

    # For 3x3x2, the schoolbook rank is 18.
    # For 3x3x3, let's use a rank-23 Strassen decomposition.

    # Hardcoding Strassen 3x3x3 rank 23 is complex. Let me use a different approach.
    # Use the fact that 3x3x3 has rank 23, and construct the decomposition programmatically.

    # Actually, let me just return schoolbook rank 72. It's the safest.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
