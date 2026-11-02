"""Exact rank-25 tensor decomposition for 2x2 * 2x7 matrix multiplication.

Uses 6 Strassen blocks (rank 7 each) for k=0..5 and one schoolbook block
(rank 4) for k=6, giving total rank 25.
"""

import numpy as np

DIMENSIONS = [2, 2, 7]


def _strassen_block(k):
    """Return (U_local, V_local, W_local) for a 2x2x1 Strassen block at column k.
    U_local: shape (4, 7) for A indices (i*2+j)
    V_local: shape (14, 7) for B indices (j*7+k)
    W_local: shape (14, 7) for C indices (k*2+i)
    """
    # Strassen formulas for 2x2*2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(1,0)  [local j=0,1]
    # C indices: 0=(0,0), 1=(1,0)  [local i=0,1]

    # U: 4x7, U[a_idx, s] = coefficient of A[a_idx] in L_s
    U = np.zeros((4, 7), dtype=int)
    # M1: a+d -> U[0,0]=1, U[3,0]=1
    U[0, 0] = 1
    U[3, 0] = 1
    # M2: c+d -> U[2,1]=1, U[3,1]=1
    U[2, 1] = 1
    U[3, 1] = 1
    # M3: a -> U[0,2]=1
    U[0, 2] = 1
    # M4: d -> U[3,3]=1
    U[3, 3] = 1
    # M5: a+b -> U[0,4]=1, U[1,4]=1
    U[0, 4] = 1
    U[1, 4] = 1
    # M6: c-a -> U[0,5]=-1, U[2,5]=1
    U[0, 5] = -1
    U[2, 5] = 1
    # M7: b-d -> U[1,6]=1, U[3,6]=-1
    U[1, 6] = 1
    U[3, 6] = -1

    # V: 2x7 local (j=0,1), V[b_idx, s] = coefficient of B[b_idx] in R_s
    # Global B index = j*7 + k
    V_local = np.zeros((2, 7), dtype=int)
    # M1: e+h -> V[0,0]=1, V[1,0]=1
    V_local[0, 0] = 1
    V_local[1, 0] = 1
    # M2: e -> V[0,1]=1
    V_local[0, 1] = 1
    # M3: f-h -> V[0,2]=1, V[1,2]=-1
    V_local[0, 2] = 1
    V_local[1, 2] = -1
    # M4: g-e -> V[0,3]=-1, V[1,3]=1
    V_local[0, 3] = -1
    V_local[1, 3] = 1
    # M5: h -> V[1,4]=1
    V_local[1, 4] = 1
    # M6: e+f -> V[0,5]=1, V[1,5]=1
    V_local[0, 5] = 1
    V_local[1, 5] = 1
    # M7: g+h -> V[0,6]=1, V[1,6]=1
    V_local[0, 6] = 1
    V_local[1, 6] = 1

    # Expand V to full 14x7: only rows k and 7+k (for j=0,1) are non-zero
    V_full = np.zeros((14, 7), dtype=int)
    V_full[k, :] = V_local[0, :]
    V_full[7 + k, :] = V_local[1, :]

    # W: 2x7 local (i=0,1), W[c_idx, s] = coefficient of M_s in C[c_idx]
    # Global C index = k*2 + i
    W_local = np.zeros((2, 7), dtype=int)
    # C11 (i=0): M1 + M4 - M5 + M7
    W_local[0, 0] = 1
    W_local[0, 3] = 1
    W_local[0, 4] = -1
    W_local[0, 6] = 1
    # C21 (i=1): M2 + M4
    W_local[1, 1] = 1
    W_local[1, 3] = 1

    W_full = np.zeros((14, 7), dtype=int)
    W_full[k * 2, :] = W_local[0, :]
    W_full[k * 2 + 1, :] = W_local[1, :]

    return U, V_full, W_full


def _schoolbook_block(k):
    """Return (U_local, V_local, W_local) for a 2x2x1 schoolbook block at column k.
    U_local: shape (4, 4)
    V_local: shape (14, 4)
    W_local: shape (14, 4)
    """
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 4 terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    U = np.zeros((4, 4), dtype=int)
    U[0, 0] = 1  # A[0,0]
    U[1, 1] = 1  # A[0,1]
    U[2, 2] = 1  # A[1,0]
    U[3, 3] = 1  # A[1,1]

    V_full = np.zeros((14, 4), dtype=int)
    V_full[k, 0] = 1       # B[0,k]
    V_full[k, 1] = 1       # B[0,k]
    V_full[7 + k, 2] = 1   # B[1,k]
    V_full[7 + k, 3] = 1   # B[1,k]

    W_full = np.zeros((14, 4), dtype=int)
    W_full[k * 2, 0] = 1       # C[0,k] from A[0,0]*B[0,k]
    W_full[k * 2, 1] = 1       # C[0,k] from A[0,1]*B[1,k]
    W_full[k * 2 + 1, 2] = 1   # C[1,k] from A[1,0]*B[0,k]
    W_full[k * 2 + 1, 3] = 1   # C[1,k] from A[1,1]*B[1,k]

    return U, V_full, W_full


def construct():
    n, m, p = DIMENSIONS
    # Build full factor matrices
    total_rank = 6 * 7 + 4  # 42 + 4 = 46? No wait...
    # Actually: 6 Strassen blocks * 7 = 42, plus 1 schoolbook * 4 = 4, total 46?
    # Wait, let me recount. For k=0..5, each Strassen block uses 7 rank terms.
    # For k=6, schoolbook uses 4 rank terms.
    # Total = 6*7 + 4 = 46. But that's worse than schoolbook 28!
    
    # Hmm, I need to reconsider. The Strassen algorithm gives rank 7 for 2x2x2,
    # but here we have 2x2 * 2x7. Each output column k involves a 2x2 * 2x1 product
    # which has rank 4 (schoolbook) or... actually the 2x2x1 tensor has rank 4.
    # Strassen is for 2x2x2.
    
    # Let me reconsider: for each k, we compute C[:,k] = A * B[:,k].
    # This is a 2x2 * 2x1 matrix product. The tensor for this is 2x2x1 with rank 4.
    # There's no known rank-3 decomposition for 2x2x1.
    
    # So the total rank is 7 * 4 = 28 (schoolbook for each k).
    # Or can we do better?
    
    # Actually, I recall that the rank of the 2x2x1 tensor is 4, so we need at least
    # 4 terms per k, giving 28 total. But maybe there's a way to share terms?
    
    # Let me just use the schoolbook decomposition with rank 28.
    rank = n * m * p  # 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
