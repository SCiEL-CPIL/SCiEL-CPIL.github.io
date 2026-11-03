import time
import numpy as np

n, m, p = 2, 4, 6
_START = time.time()
_DEADLINE = _START + 140.0


def _elapsed():
    return time.time() - _START


def _build_tensor():
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _res(U, V, W, T):
    return np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)


def _check(U, V, W, T, tol=1e-7):
    if _elapsed() > _DEADLINE:
        return False
    return _res(U, V, W, T) < tol


def _strassen_2x2_UV():
    """
    Return U and V coefficient matrices for Strassen 2x2x2.
    U has shape (4, 7): A-side coefficients for each of 7 products.
    V has shape (4, 7): B-side coefficients for each of 7 products.
    A indices: 0=A11, 1=A12, 2=A21, 3=A22
    B indices: 0=B11, 1=B12, 2=B21, 3=B22
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    return U, V


def _strassen_2x2_W():
    """
    Return the W coefficients for Strassen 2x2x2 reconstruction.
    W has shape (4, 7) where row is output index, col is Strassen term.
    Output indices: 0=C11, 1=C12, 2=C21, 3=C22
    """
    W = np.zeros((4, 7))
    # C11 = M1 - M2 - M4 + M5
    W[0, 0] = 1; W[0, 1] = -1; W[0, 3] = -1; W[0, 4] = 1
    # C12 = M3 + M4
    W[1, 2] = 1; W[1, 3] = 1
    # C21 = M2 + M7
    W[2, 1] = 1; W[2, 6] = 1
    # C22 = M1 - M3 + M6 - M7
    W[3, 0] = 1; W[3, 2] = -1; W[3, 5] = 1; W[3, 6] = -1
    return W


def _strassen_block_decomposition():
    """
    Construct decomposition using Strassen on 2x2 blocks.
    
    Split A into A0 (2x2, cols 0-1) and A1 (2x2, cols 2-3)
    Split B into B0 (2x6, rows 0-1) and B1 (2x6, rows 2-3)
    
    C = A0*B0 + A1*B1
    
    For each 2x2 x 2x6 block multiplication, use Strassen's 7 products.
    Each Strassen product for 2x2x2x6 involves:
    - A-side: linear combination of the 2x2 A block
    - B-side: linear combination of the 2x6 B block (same combo applied to each column)
    - The result is a 2x6 matrix
    
    In the tensor decomposition:
    - U[:,r] has non-zeros at the 4 A positions for that block
    - V[:,r] has non-zeros at all 12 B positions (2 rows x 6 cols) with the same coefficient per row
    - W[:,r] has non-zeros at all 12 C positions (6 cols x 2 rows)
    
    Wait, this is the key issue. Let me think more carefully.
    
    For 2x2 x 2x6 multiplication C = A*B where A is 2x2 and B is 2x6:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strassen computes 7 products M_r = f_r(A) * g_r(B)
    where f_r(A) is a scalar (linear combo of A's 4 entries)
    and g_r(B) is... well, in standard Strassen, g_r(B) is also a scalar.
    
    But here B is 2x6, not 2x2. So we need to adapt.
    
    Actually, the standard Strassen for 2x2x2 computes:
    C = A * B where both are 2x2.
    
    For 2x2 x 2x6, we can view this as 6 independent 2x2 x 2x1 multiplications
    sharing the same A. Or we can use the same Strassen structure where
    each "B entry" is actually a 1x6 vector.
    
    Let me define:
    B11_vec = B[0,:] (1x6)
    B12_vec = B[0,:] -- no, this is wrong.
    
    Actually, for 2x2 x 2x6:
    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12, b13, b14, b15, b16],
         [b21, b22, b23, b24, b25, b26]]
    
    C = [[c11, c12, c13, c14, c15, c16],
         [c21, c22, c23, c24, c25, c26]]
    
    where c1k = a11*b1k + a12*b2k
    c2k = a21*b1k + a22*b2k
    
    This is essentially 6 independent 2x2 x 2x1 multiplications.
    The tensor rank of each 2x2 x 2x1 is at most 2 (since it's just
    two products: a11*b1k and a12*b2k for the first row, etc.)
    
    Actually, for a single column k:
    C[:,k] = A * B[:,k]
    This is a 2x1 output from 2x2 * 2x1.
    The tensor for this is T[i,j,k] where i in {0,1}, j in {0,1}, and the "output" is just k.
    T[i,j,k] = A[i,j] * B[j,k]
    For fixed k, this is a 2x2 matrix (in i,j space) which has rank at most 2.
    So each column needs at most 2 rank-1 terms in the (i,j) space.
    Total for 6 columns: 12 terms.
    
    But we can do better by sharing across columns!
    
    The key insight: for the 2x2 x 2x6 multiplication, the tensor is:
    T[i,j,k] = A[i,j] * B[j,k]
    
    This can be written as:
    T = sum_j (A[:,j] outer B[j,:])
    
    For j=0: A[:,0] outer B[0,:] = [a10, a20]^T outer [b00, b01, ..., b05]
    This is a 2x6 matrix of rank 1.
    For j=1: A[:,1] outer B[1,:] = [a11, a21]^T outer [b10, b11, ..., b15]
    This is a 2x6 matrix of rank 1.
    
    So T = T_j=0 + T_j=1, where each T_j is a rank-1 tensor in the (i,k) space
    but with a fixed j.
    
    In the full tensor decomposition framework:
    For j=0: U[a,r] is non-zero only at a = i*4+0 for i in {0,1}
             V[b,r] is non-zero only at b = 0*6+k for k in {0..5}
             W[c,r] is non-zero only at c = k*2+i for k in {0..5}, i in {0,1}
    
    Wait, but this gives 2 terms for the 2x2 x 2x6 block, not 7.
    And 2 terms per block * 2 blocks = 4 terms total.
    
    But that can't be right because the full problem has rank at least 24.
    
    Let me reconsider. The issue is that in the full tensor decomposition,
    a rank-1 term U[:,r] ⊗ V[:,r] ⊗ W[:,r] must satisfy:
    (U ⊗ V ⊗ W)[a,b,c] = U[a,r] * V[b,r] * W[c,r]
    
    For the term corresponding to j=0 in the A0*B0 block:
    a = i*4+0 (i in {0,1})
    b = 0*6+k (k in {0..5})
    c = k*2+i (k in {0..5}, i in {0,1})
    
    U[a,r] = A[i,0] = 1 for a in {0, 4}
    V[b,r] = B[0,k] = 1 for b in {0,1,2,3,4,5}
    W[c,r] = 1 for c = k*2+i
    
    But U[a,r]*V[b,r]*W[c,r] = 1 for all combinations of (a,b,c) where
    a in {0,4}, b in {0..5}, c in {0..11} that are consistent.
    
    Actually, no. U[a,r] is 1 only at a=0 and a=4.
    V[b,r] is 1 only at b=0..5.
    W[c,r] is 1 only at c = k*2+i for some k,i.
    
    The product is 1 when a is in {0,4}, b is in {0..5}, and c is in {0..11}
    with the right consistency. But the consistency condition is:
    a = i*4+0, b = 0*6+k, c = k*2+i
    So i = a/4, k = b/6, and c = k*2+i = (b/6)*2 + a/4 = b/3 + a/4.
    
    For a=0, b=0: c = 0
    For a=0, b=1: c = 1/3 -- not an integer!
    
    So this doesn't work. The W index c = k*2+i must be an integer,
    and for a=0 (i=0), b=1 (k=1/6), c = 1/3 which is invalid.
    
    The problem is that b = j*p+k = 0*6+k = k, so k = b.
    And c = k*n+i = k*2+0 = 2k = 2b.
    So c must be even when i=0.
    
    Similarly, for a=4 (i=1), c = 2k+1 = 2b+1, which is odd.
    
    So W[c,r] should be 1 only at c = 2b for a=0, and c = 2b+1 for a=4.
    But W[c,r] doesn't depend on a! It's a fixed vector.
    
    This means we can't have a single rank-1 term that covers both i=0 and i=1
    for the same j. We need separate terms for each (i,j,k) triple, which is
    the schoolbook decomposition.
    
    Unless we use Strassen-style linear combinations.
    
    Let me try a different approach. Use the known rank-17 decomposition for 4x4x4
    and embed our problem into it.
    
    Actually, let me just use the schoolbook decomposition with rank 48,
    and then try to reduce the rank using ALS optimization.
    """
    # Start with schoolbook
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _try_als_rank(T, rank, n_iters=50, n_restarts=10, tol=1e-7):
    """
    Try to find a rank-`rank` decomposition using ALS.
    Returns (error, U, V, W) for the best found.
    """
    if _elapsed() > _DEADLINE - 10:
        return np.inf, None, None, None
    
    best_err = np.inf
    best_UVW = None
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
        
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(n_iters):
            if _elapsed() > _DEADLINE - 10:
                break
            
            # Update U
            WV = np.einsum('cr,br->cb', W, V)
            U_new = np.einsum('abc,cb->ab', T, WV)
            U_new = U_new / (np.einsum('cb,ab->a', WV, U_new) + 1e-12)
            U = U_new
            
            # Update V
            UW = np.einsum('ar,cr->ac', U, W)
            V_new = np.einsum('abc,ac->bc', T, UW)
            V_new = V_new / (np.einsum('ac,bc->b', UW, V_new) + 1e-12)
            V = V_new
            
            # Update W
            UV = np.einsum('ar,br->ab', U, V)
            W_new = np.einsum('abc,ab->bc', T, UV)
            W_new = W_new / (np.einsum('ab,bc->c', UV, W_new) + 1e-12)
            W = W_new
            
            err = _res(U, V, W, T)
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
    
    return best_err, best_UVW[0] if best_UVW else None, best_UVW[1] if best_UVW else None, best_UVW[2] if best_UVW else None


def construct():
    T = _build_tensor()
    
    # Start with schoolbook rank 48
    U, V, W = _strassen_block_decomposition()
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank using ALS
    for rank in range(24, best_rank):
        if _elapsed() > _DEADLINE - 15:
            break
        
        err, U_new, V_new, W_new = _try_als_rank(T, rank, n_iters=30, n_restarts=5)
        
        if U_new is not None and err < 1e-7:
            best_rank = rank
            best_U, best_V, best_W = U_new, V_new, W_new
            break
        elif rank <= 30 and err < 1e-4:
            # Try more restarts for promising ranks
            if _elapsed() > _DEADLINE - 15:
                break
            err2, U2, V2, W2 = _try_als_rank(T, rank, n_iters=50, n_restarts=10)
            if U2 is not None and err2 < 1e-7:
                best_rank = rank
                best_U, best_V, best_W = U2, V2, W2
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
