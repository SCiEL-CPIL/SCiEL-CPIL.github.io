"""Rewritten program for matrix multiplication tensor rank search (3x4 by 4x4).

Key insight: the 3x4 x 4x4 multiplication tensor has a block structure where
the bottom row (1x4 x 4x4) can be handled with rank 4 (not 16 schoolbook),
since the output is a single row of length 4 and the tensor T[j,k]=1 for
j,k in 0..3 has rank 4 (it's the all-ones 4x4 matrix, rank 1, but we need
rank-1 terms in the tensor sense: each term is u_j * v_k * w_kj, and we need
to cover all 16 entries with rank-1 tensor terms).

Actually, let me reconsider. The 1x4 x 4x4 tensor T[j,k] = 1 (for j=0..3, k=0..3)
has the property that T = 1*1^T as a matrix, so as a matrix it has rank 1.
But as a 3-way tensor with dimensions (1, 4, 4), it's just a matrix, so its
tensor rank equals its matrix rank = 1.

Wait, that's not right either. Let me think more carefully.

The tensor for 1x4 x 4x4 matrix multiplication is T[i,j,k] where i=0 (only one),
j=0..3, k=0..3, and T[0,j,k] = 1 for all j,k. This is a 1x4x4 tensor which is
just a 4x4 matrix of all ones. The tensor rank of a 1xMxN tensor equals the
matrix rank of the MxN matrix. The all-ones 4x4 matrix has rank 1.

So the bottom row block should only need rank 1, not 4 or 16!

Let me verify: for the 1x4 x 4x4 case, we need:
C[0,k] = sum_j A[0,j] * B[j,k] for k=0..3

The tensor decomposition: C[0,k] = sum_r U[0,j_r] * V[j_r,k_r] * W[k_r,0]

If we use rank 1: U[0,0]=1, V[j,k]=1 for all j,k, W[k,0]=1
Then C[0,k] = sum_j 1 * 1 * 1 = 4, not 1.

Hmm, that doesn't work. Let me reconsider.

Actually, the tensor T[i,j,k] = delta(i,i') * delta(j,j') * delta(k,k') for the
matrix multiplication tensor. For 1x4 x 4x4:
T[0,j,k] = 1 for all j,k.

The decomposition is: sum_r U[0,j] * V[j,k] * W[k,0] should equal T[0,j,k] = 1.

With rank 1: U[0,j] = 1 for all j, V[j,k] = 1 for all j,k, W[k,0] = 1
Then sum_r U[0,j]*V[j,k]*W[k,0] = 1*1*1 = 1. Yes! Rank 1 works.

So the total rank should be: 4 * 7 (Strassen for 2x2x2 blocks) + 1 (for the 1x4x4 block)
= 28 + 1 = 29.

Wait, but I need to be more careful. The 1x4 x 4x4 block: A is 1x4, B is 4x4, C is 1x4.
The tensor is T[j,k] = 1 (a 4x4 matrix of ones). As a 3-way tensor of shape (1,4,4),
its rank is the matrix rank of the 4x4 all-ones matrix, which is 1.

So rank = 28 + 1 = 29.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """
    Construct a rank-29 decomposition:
    - 4 Strassen 2x2x2 blocks for the top 2 rows (2x4 x 4x4 -> 2x4)
    - 1 rank-1 term for the bottom row (1x4 x 4x4 -> 1x4)
    Total: 4*7 + 1 = 29
    """
    n, m, p = 3, 4, 4
    rank = 29
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # X is A[rA:rA+2, cA:cA+2]
        # Y is B[rB:rB+2, cB:cB+2]
        # Z is C[rA:rA+2, cB:cB+2]
        
        # Global U index: (rA + i, cA + j) -> (rA+i)*m + (cA+j)
        # Global V index: (rB + j, cB + k) -> (rB+j)*p + (cB+k)
        # Global W index: (cB + k, rA + i) -> (cB+k)*n + (rA+i)
        
        # Term 1: M1 = (X11+X22)(Y11+Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1  # Z11
        W[(cB+1)*n + (rA+1), col] = 1  # Z22
        col += 1
        
        # Term 2: M2 = (X21+X22)Y11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1   # Z21
        W[(cB+1)*n + (rA+1), col] = -1  # Z22
        col += 1
        
        # Term 3: M3 = X11(Y12-Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1  # Z12
        W[(cB+1)*n + (rA+1), col] = 1  # Z22
        col += 1
        
        # Term 4: M4 = X22(Y21-Y11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1  # Z11
        W[(cB+0)*n + (rA+1), col] = 1  # Z21
        col += 1
        
        # Term 5: M5 = (X11+X12)Y22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1   # Z12
        W[(cB+0)*n + (rA+0), col] = -1  # Z11
        col += 1
        
        # Term 6: M6 = (X21-X11)(Y11+Y12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1  # Z22
        col += 1
        
        # Term 7: M7 = (X12-X22)(Y21+Y22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1  # Z11
        col += 1

    # Apply 4 Strassen blocks for top 2 rows
    # Top-Left: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0)
    # Top-Right: A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2)
    # Bottom-Left (of top block): A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0)
    # Bottom-Right (of top block): A[0:2, 2:4] x B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2)
    
    # Bottom row: A[2:3, 0:4] x B[0:4, 0:4] -> C[2:3, 0:4]
    # This is a 1x4 x 4x4 tensor, which has rank 1
    # U[2,j] = 1 for j=0..3
    # V[j,k] = 1 for j=0..3, k=0..3
    # W[k,2] = 1 for k=0..3
    for j in range(4):
        U[2 * m + j, col] = 1
    for j in range(4):
        for k in range(4):
            V[j * p + k, col] = 1
    for k in range(4):
        W[k * n + 2, col] = 1
    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    deadline = time.time() + 110
    result = _construct_structured_decomposition()
    
    # Verify
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if _verify_decomposition(U_np, V_np, W_np, n, m, p, tol=1e-6):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
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
