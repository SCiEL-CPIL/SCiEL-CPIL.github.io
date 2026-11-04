"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Exploits the n=2 structure: C[0,k] = sum_j A[0,j] B[j,k] and C[1,k] = sum_j A[1,j] B[j,k].
The schoolbook rank is 56. We aim to reduce it by sharing work between the two output rows.

Key insight: For each (j,k), the product A[0,j]*B[j,k] contributes to C[0,k], and
A[1,j]*B[j,k] contributes to C[1,k]. We can use rank-1 terms that have structure
allowing us to combine contributions.

Approach: Use the fact that we can write:
C[0,k] = sum_j A[0,j] B[j,k]
C[1,k] = sum_j A[1,j] B[j,k]

Define S[j,k] = A[0,j]*B[j,k] and D[j,k] = A[1,j]*B[j,k].
Then C[0,k] = sum_j S[j,k] and C[1,k] = sum_j D[j,k].

But S and D are not separable in a useful way for direct sharing.

Alternative: Use the identity that for n=2, we can use a "Strassen-like" approach
where we compute intermediate products that combine both rows.

Let me try a concrete construction with rank < 56.

For each j (0..3) and k (0..6), we need A[0,j]*B[j,k] in C[0,k] and A[1,j]*B[j,k] in C[1,k].

If I use a term where U has 1 at (0,j) and (1,j), V has 1 at (j,k), and W has 1 at (k,0) and (k,1),
then the contribution to C[0,k] is (A[0,j]+A[1,j]) * B[j,k] * 1, and to C[1,k] is also (A[0,j]+A[1,j]) * B[j,k] * 1.
That's not what we want.

Better: Use two types of terms:
Type 1: U has 1 at (0,j), V has 1 at (j,k), W has 1 at (k,0). Contribution: A[0,j]*B[j,k] to C[0,k].
Type 2: U has 1 at (1,j), V has 1 at (j,k), W has 1 at (k,1). Contribution: A[1,j]*B[j,k] to C[1,k].

This is just schoolbook with rank 56.

To reduce rank, I need to find linear combinations. Consider:
For a fixed j, define X = A[0,j] and Y = A[1,j]. Then we need X*B[j,k] in C[0,k] and Y*B[j,k] in C[1,k] for all k.

If I use terms with U having (1,1) at positions (0,j) and (1,j), that gives (X+Y) which isn't directly useful.

Let me try: For each j, use 7 terms for the "sum" and 7 terms for the "difference":
S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j].
Then A[0,j] = (S_j + D_j)/2, A[1,j] = (S_j - D_j)/2.

C[0,k] = sum_j (S_j + D_j)/2 * B[j,k]
C[1,k] = sum_j (S_j - D_j)/2 * B[j,k]

So C[0,k] = (1/2) sum_j S_j B[j,k] + (1/2) sum_j D_j B[j,k]
C[1,k] = (1/2) sum_j S_j B[j,k] - (1/2) sum_j D_j B[j,k]

Let T1[k] = sum_j S_j B[j,k] and T2[k] = sum_j D_j B[j,k].
Then C[0,k] = (T1[k] + T2[k])/2 and C[1,k] = (T1[k] - T2[k])/2.

Computing T1 requires a [1,4,7] multiplication (rank 28 schoolbook), and T2 similarly (rank 28).
Total rank would be 56, same as before. No gain.

Let me just verify the schoolbook works and return it, ensuring correctness.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
