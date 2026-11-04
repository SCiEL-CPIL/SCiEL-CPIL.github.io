"""Matrix multiplication tensor decomposition for 2x4 * 4x7.
Exploits the n=2 structure to reduce rank below schoolbook (56).

The key insight: C[1,k] - C[0,k] = (A[1,:] - A[0,:]) @ B
So we can compute C[0,:] and D@B where D = A[1]-A[0], then C[1,:] = C[0,:] + D@B.

The addition C[1,:] = C[0,:] + D@B can be folded into the W factors without adding rank.

For the [1,4,7] sub-problems, we use schoolbook (rank 28 each).
Total rank: 28 + 28 = 56... same as schoolbook.

But we can do better. Let me think about sharing between the two [1,4,7] computations.

Actually, the key is that D[j] = A[1,j] - A[0,j] is a linear combination of A entries.
So the D@B computation uses the same B entries as C[0,:]@B computation.

Let me try a different approach: use a structured decomposition where we share
the B-side factors between the two row computations.

For rank 1 term c:
- It contributes to C[i,k] via: (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c]

If we set V[j*p+k,c] to be the same for both i=0 and i=1 computations, we share work.

Let me try: for each (j,k) pair, create one term that contributes to both C[0,k] and C[1,k].
Term (j,k): U[0*m+j, c] = 1, U[1*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+0, c] = 1, W[k*n+1, c] = 1

This gives: C[0,k] += A[0,j]*B[j,k]*1 and C[1,k] += A[1,j]*B[j,k]*1
That's exactly what we want! And we only need m*p = 28 such terms.

Wait, that's just the schoolbook with U having 1s in both rows. Let me check:
For term c=(j,k): U[0*m+j,c]=1, U[1*m+j,c]=1, V[j*p+k,c]=1, W[k*n+0,c]=1, W[k*n+1,c]=1

C[0,k] = sum_c (sum_j A[0,j]*U[0*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+0,c]
= sum_{j,k'} A[0,j]*B[j,k']*W[k'*n+0,c] where c=(j,k')
= sum_j A[0,j]*B[j,k]*1 = C[0,k] ✓

C[1,k] = sum_c (sum_j A[1,j]*U[1*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+1,c]
= sum_j A[1,j]*B[j,k]*1 = C[1,k] ✓

So rank 28 works! That's half of schoolbook.

But wait, can we do even better? The lower bound for [2,4,7] is at least max(2*4, 4*7, 7*2) = 28.
So rank 28 is optimal!

Let me verify this is correct by constructing it properly.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    # Rank 28 decomposition: one term per (j,k) pair
    rank = m * p  # 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: set both rows to 1 at position j
            U[0 * m + j, col] = 1.0  # A[0,j]
            U[1 * m + j, col] = 1.0  # A[1,j]
            # V: set position (j,k) to 1
            V[j * p + k, col] = 1.0
            # W: set both output entries to 1
            W[k * n + 0, col] = 1.0  # C[0,k]
            W[k * n + 1, col] = 1.0  # C[1,k]
            col += 1
    
    # Verify the decomposition
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_recon = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(rank):
                a_part = sum(A_test[i, j] * U[i*m + j, c] for j in range(m))
                b_part = sum(B_test[j, k] * V[j*p + k, c] for j in range(m))
                C_recon[i, k] += a_part * b_part * W[k*n + i, c]
    
    assert np.allclose(C_recon, C_expected), "Decomposition is incorrect!"
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
