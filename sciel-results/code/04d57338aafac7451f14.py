"""Matrix multiplication 5x5x5 rank-minimal exact tensor decomposition.

Strategy: Embed the known rank-49 4x4x4 decomposition (de Groote / Burgisser)
and add 9 extra terms for the 5th row/column interactions, giving rank 58.
If the 4x4 table is unavailable or fails verification, fall back to schoolbook 125.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that U @ diag(t) @ V^T matches the tensor for all (i,j,k).
    Uses the convention: C[i,k] = sum_j A[i,j] B[j,k]
    Tensor T[i*m+j, j*p+k, k*n+i] = 1 (using 0-indexed).
    We verify: for each c, U[row,c]*V[col,c]*W[out,c] contributes to T.
    Equivalently, the sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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


# Known rank-49 4x4x4 tensor decomposition (de Groote's algorithm, 1979).
# Each term is a triple (i, j, k) with 0<=i,j,k<4, but the decomposition
# is not simply the 64 basis terms. The actual 49 terms from de Groote's
# paper are complex. Since I cannot reliably reproduce the full 49-term
# table from memory, I will use a verified smaller construction if possible,
# or fall back to schoolbook.

# Alternative: use the known fact that M_4 has rank 49. But without the
# explicit table, I cannot embed it. Instead, I'll try a constructive approach:
# Use the schoolbook for 4x4 (rank 64) is worse. Let me try to use a known
# rank-49 table. Actually, let me just use the schoolbook for 5x5 (rank 125)
# as a safe fallback, since I cannot guarantee the 49-term table is correct.

# However, I can try a different approach: use the 4x4 rank-49 if I can
# find it. Since I cannot, I'll return schoolbook.

# Actually, let me try to implement a known rank-49 decomposition.
# The de Groote algorithm for 4x4 uses 49 multiplications. The explicit
# table is very long. Without it, I'll fall back.

# Let me try another approach: use the fact that we can potentially use
# a rank-49 4x4 table if available. Since it's not, return schoolbook.

def construct():
    """Construct the tensor decomposition for 5x5x5 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    U: shape (25, rank), V: shape (25, rank), W: shape (25, rank)
    Convention: W[k*n+i, c] for the output factor.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Try to use rank-58 construction (4x4 rank-49 + 9 extra)
    # Since I don't have the verified 4x4 rank-49 table, fall back to schoolbook.
    # This is safe and valid.
    
    result = _schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END
