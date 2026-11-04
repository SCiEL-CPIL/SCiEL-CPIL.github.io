"""
Matrix multiplication 3x4x8 tensor decomposition.

Strategy: Use the Kronecker product structure to reduce rank.
- 3x4x8 = (3x2x2) ⊗ (1x2x4) or similar decompositions
- Use Strassen (rank 7) for 2x2x2 sub-problems where applicable
- Combine with schoolbook for remaining dimensions

The key insight: 4 = 2*2 and 8 = 2*4, so we can decompose the inner
dimension j into (j1, j2) and output k into (k1, k2).

For the 3x2x2 part (3 rows, 2 inner, 2 outer): rank 12 (schoolbook)
For the 1x2x4 part (1 row, 2 inner, 4 outer): rank 8 (schoolbook)

But we can do better by using Strassen on the 2x2x2 core.

Actually, let's think more carefully:
- C[i, k] = sum_j A[i,j] * B[j,k]
- j ranges 0..3, k ranges 0..7, i ranges 0..2

Decompose j = 2*j1 + j2, k = 4*k1 + k2 where j1,j2 in {0,1}, k1 in {0,1}, k2 in {0..3}

Then C[i, 4*k1+k2] = sum_{j1,j2} A[i, 2*j1+j2] * B[2*j1+j2, 4*k1+k2]

For fixed i, k2: C[i, 4*k1+k2] = sum_{j1,j2} A[i, 2*j1+j2] * B[2*j1+j2, 4*k1+k2]

This is a sum over 4 terms (j1,j2) for each (i, k1, k2).

We can group by j2:
C[i, 4*k1+k2] = sum_{j2} sum_{j1} A[i, 2*j1+j2] * B[2*j1+j2, 4*k1+k2]

For each j2, this is a 3x2 (i x j1) times 2x2 (j1 x k1) matrix multiplication
with an extra k2 index.

Actually, let me try a different approach. Use the fact that:
- The tensor T[(i,j), (j',k)] = delta_{j,j'}
- We can write this as a sum of rank-1 tensors

For the Kronecker approach:
T_{3,4,8} = T_{3,2,2} ⊗ T_{1,2,4} (approximately, with index rearrangement)

T_{3,2,2} has rank 12 (schoolbook)
T_{1,2,4} has rank 8 (schoolbook)

But T_{2,2,2} has rank 7 (Strassen), so:
T_{3,4,8} can be seen as combining T_{3,2,2} and T_{2,2,4}

Let me try: decompose as (3x2x2) * (2x2x4) using tensor product structure.

For 3x2x2: schoolbook rank 12
For 2x2x4: schoolbook rank 16, but we can use Strassen-like: 2x2x2 (rank 7) times 1x2x2 (rank 4) = 28? No...

Actually 2x2x4: C[i,k] = sum_j A[i,j] B[j,k] where i in {0,1}, j in {0,1}, k in {0..3}
This is two independent 2x2 multiplications (one for k=0,1 and one for k=2,3)
Each 2x2x2 has rank 7 (Strassen), so total rank = 14.

So 3x4x8 = 3x2x2 (rank 12) combined with 2x2x4 (rank 14)?
But how to combine? The tensor product would give rank 12*14 = 168, which is worse.

Let me try a different decomposition. 

3x4x8: We can think of this as computing 3 outputs of size 8 from 4 inputs.
For each output row i (0..2), we compute a vector of length 8.
C[i, :] = A[i, :] * B where A[i,:] is 1x4 and B is 4x8.

This is a 1x4 times 4x8 multiplication for each i, but they share the same B.

For 1x4 x 4x8: rank 32 (schoolbook), but we can do better.
1x4 x 4x8 = (1x2 x 2x4) via Kronecker? 
1x2 x 2x4: rank 8 (schoolbook)
But 2x2 x 2x2 has rank 7 (Strassen)

Hmm, let me just try to use gradient descent more aggressively with better initialization.

Actually, let me try a structured approach based on Strassen:

The 2x2x2 Strassen algorithm uses 7 multiplications:
m1 = (a11+a22)(b11+b22)
m2 = (a21+a22)b11
m3 = a11(b12-b22)
m4 = a22(b21-b11)
m5 = (a11+a12)b22
m6 = (a21-a11)(b11+b12)
m7 = (a12-a22)(b21+b22)

c11 = m1 + m4 - m5 + m7
c12 = m3 + m5
c21 = m2 + m4
c22 = m1 - m2 + m3 + m6

For 3x4x8, we can try to use this structure on the 2x2 sub-blocks.

Let me try: partition the 4 inner dimensions into two groups of 2, and the 8 output dimensions into two groups of 4.

A is 3x4, B is 4x8.
Split A into A1 (3x2) and A2 (3x2)
Split B into B1 (2x8) and B2 (2x8)

C = A1*B1 + A2*B2

Each A*B is 3x2 times 2x8, which has rank 24 (schoolbook).
Total: 48. Still better than 96!

But can we do even better? 3x2 times 2x8:
For each output (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
This is a sum of 2 rank-1 terms per (i,k) pair, total 3*8*2 = 48.

Can we use Strassen here? 3x2x2 has rank 12 (no known improvement).
But 3x2x8 = 3x2x2 x 1x1x4? Not quite.

Actually 3x2x8: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
This is just 2 independent 3x1x8 vector multiplications added together.
Rank = 2 * 3 * 8 / ... no, rank = 2*3*8 = 48 for schoolbook.

Wait, that's not right. For each (i,k), we have 2 terms. Total terms = 3*8*2 = 48.
Each term is rank 1, so rank <= 48.

But can we do better? The tensor is T[(i,j),(j',k)] = delta_{j,j'} for i in 0..2, j in 0..1, k in 0..7.
This has rank exactly 2*3*8 = 48? Let me check the lower bound.

The slice rank of T_{n,m,p} is at least n*m + m*p + p*n - 2*m*p... no, that's not right.
The standard lower bound is max(n*m, m*p, p*n) = max(6, 16, 24) = 24.
But the actual minimum rank for 3x2x8 might be higher.

Let me just try to find a rank-48 solution (which is easy: split into two 3x2x8 problems)
and see if we can do better.

Actually, 3x4x8 = 3x2x8 + 3x2x8 where the two 3x2x8 problems share the same B structure.
If we use schoolbook for each 3x2x8, we get rank 48.

But wait, can we use Strassen on the 2x2 part? 
3x4x8: split j into (j1,j2) where j1 in {0,1}, j2 in {0,1}
Split k into (k1,k2) where k1 in {0,1}, k2 in {0..3}

C[i, 2*k1+k2] = sum_{j1,j2} A[i, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]

Hmm, this doesn't decompose cleanly into Strassen.

Let me try the simplest improvement: rank 48 by splitting the inner dimension.

Then try to use gradient descent to find something better.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _build_true_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_to_half_inters(U, V, W):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10):
    n, m, p = T_true.shape
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_U, best_V, best_W, best_err

def _build_split_decomposition(n, m, p, split_point):
    """Split inner dimension at split_point, creating two sub-problems."""
    m1 = split_point
    m2 = m - split_point
    
    rank = n * m1 * p + n * m2 * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # First part: rows 0..m1-1
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Second part: rows m1..m-1
    for i in range(n):
        for j in range(m1, m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    T_true = _build_true_tensor(n, m, p)
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    # Try split decomposition (rank 48 for 3x4x8 with split at 2)
    try:
        U_split, V_split, W_split = _build_split_decomposition(n, m, p, 2)
        if _verify(U_split, V_split, W_split, T_true, atol=1e-6):
            snapped = _snap_to_half_inters(np.array(U_split), np.array(V_split), np.array(W_split))
            if _verify(snapped[0], snapped[1], snapped[2], T_true, atol=1e-6):
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
            return {"U": U_split, "V": V_split, "W": W_split}
    except Exception:
        pass
    
    # Try to find better solution via gradient descent
    best_rank = 48  # from split decomposition
    best_result = {"U": U_split, "V": V_split, "W": W_split}
    
    # Try lower ranks with gradient descent
    for target_rank in [40, 36, 32, 28]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
            
            rng = np.random.RandomState(restart * 1000 + target_rank)
            
            U = rng.randn(nm, target_rank) * 0.3
            V = rng.randn(mp, target_rank) * 0.3
            W = rng.randn(pn, target_rank) * 0.3
            
            eta = 0.005
            max_iters = 3000
            
            for phase in range(4):
                U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
                
                if err < 1e-8:
                    snapped = _snap_to_half_inters(U, V, W)
                    if _verify(snapped[0], snapped[1], snapped[2], T_true, atol=1e-6):
                        return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
                    if _verify(U.tolist(), V.tolist(), W.tolist(), T_true, atol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                eta *= 0.5
                max_iters = max(500, max_iters // 2)
    
    return best_result
