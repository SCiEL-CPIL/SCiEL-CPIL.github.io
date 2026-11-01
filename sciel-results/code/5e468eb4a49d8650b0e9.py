"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: a = i*2+j, b = j*2+k, c = k*2+i
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _schoolbook_2x2x1():
    """
    Return schoolbook decomposition for 2x2 * 2x1 -> 2x1.
    U: shape (4, 4), V: shape (2, 4), W: shape (2, 4)
    Indexing: a = i*2+j, b = j, c = i
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    T_dec = np.zeros((nm, mp, pn), dtype=int)
    for r in range(rank):
        T_dec += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(nm, mp, pn)
    
    return np.array_equal(T, T_dec)


def _build_block_decomposition():
    """Build the block Strassen decomposition with rank 36."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    rank = 36
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    
    # Block 1: A_L * B_T (i in 0..1, j in 0..1 for A; j in 0..1, k in 0..4 for B)
    
    # Sub-block 1a: A_L * B_TL (j in 0..1, k in 0..1)
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + j
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = j * 5 + k
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = k * 2 + i
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Sub-block 1b: A_L * B_TM (j in 0..1, k in 2)
    sU2, sV2, sW2 = _schoolbook_2x2x1()
    for r in range(4):
        for i in range(2):
            for j in range(2):
                a = i * 4 + j
                U[a, col] += sU2[i*2+j, r]
        for j in range(2):
            b = j * 5 + 2
            V[b, col] += sV2[j, r]
        for i in range(2):
            c = 2 * 2 + i
            W[c, col] += sW2[i, r]
        col += 1
    
    # Sub-block 1c: A_L * B_TR (j in 0..1, k in 3..4)
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + j
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = j * 5 + (3 + k)
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = (3 + k) * 2 + i
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Block 2: A_R * B_B (i in 0..1, j in 2..3 for A; j in 2..3, k in 0..4 for B)
    
    # Sub-block 2a: A_R * B_BL (j in 2..3, k in 0..1)
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + (2 + j)
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = (2 + j) * 5 + k
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = k * 2 + i
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Sub-block 2b: A_R * B_BM (j in 2..3, k in 2)
    sU2, sV2, sW2 = _schoolbook_2x2x1()
    for r in range(4):
        for i in range(2):
            for j in range(2):
                a = i * 4 + (2 + j)
                U[a, col] += sU2[i*2+j, r]
        for j in range(2):
            b = (2 + j) * 5 + 2
            V[b, col] += sV2[j, r]
        for i in range(2):
            c = 2 * 2 + i
            W[c, col] += sW2[i, r]
        col += 1
    
    # Sub-block 2c: A_R * B_BR (j in 2..3, k in 3..4)
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + (2 + j)
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = (2 + j) * 5 + (3 + k)
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = (3 + k) * 2 + i
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_iter=5000):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    If the sum of two rank-1 terms can be replaced by a single rank-1 term
    (i.e., they are linearly dependent in the tensor space), we can merge them.
    
    Strategy: For each pair of columns (r1, r2), check if the tensor formed by
    their sum is rank-1. If so, we can replace both with a single term.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Build the full tensor from current decomposition
    T = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(nm, mp, pn)
    
    # Check if we can find any pair that can be merged
    # A pair (r1, r2) is mergeable if T[:, :, :] restricted to those two terms
    # forms a rank-1 tensor when considered as a 3D array.
    # Actually, two rank-1 tensors sum to rank 1 iff they are scalar multiples.
    # So we need: u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2 = c * (u' ⊗ v' ⊗ w')
    # This happens iff u1/v1/w1 and u2/v2/w2 are proportional.
    
    # Check for proportional columns
    for r1 in range(rank):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        for r2 in range(r1 + 1, rank):
            # Check if column r1 is proportional to column r2 in U, V, W
            u1, u2 = U[:, r1], U[:, r2]
            v1, v2 = V[:, r1], V[:, r2]
            w1, w2 = W[:, r1], W[:, r2]
            
            # Find scale factor from U
            nonzero1 = u1 != 0
            nonzero2 = u2 != 0
            common = nonzero1 & nonzero2
            if not np.any(common):
                continue
            
            # Check if u1/common is proportional to u2/common
            ratio_u = u1[common] / u2[common]
            if not np.allclose(ratio_u, ratio_u[0], atol=1e-10):
                continue
            
            scale = ratio_u[0]
            
            # Check V
            if not np.allclose(v1, scale * v2, atol=1e-10):
                continue
            
            # Check W
            if not np.allclose(w1, scale * w2, atol=1e-10):
                continue
            
            # Found a mergeable pair! Remove r2 and add to r1
            # Actually, we can just remove one of them since they're proportional
            # The sum is (1+scale) * (u2 ⊗ v2 ⊗ w2)
            U[:, r1] += scale * U[:, r2]
            V[:, r1] += scale * V[:, r2]
            W[:, r1] += scale * W[:, r2]
            U[:, r2] = 0
            V[:, r2] = 0
            W[:, r2] = 0
            
            # Verify
            if _verify(U, V, W, n, m, p):
                # Remove the zero column
                U = np.delete(U, r2, axis=1)
                V = np.delete(V, r2, axis=1)
                W = np.delete(W, r2, axis=1)
                rank -= 1
                # Restart search with new rank
                return U, V, W
    
    return U, V, W


def _schoolbook_full(n, m, p):
    """Full schoolbook decomposition with rank n*m*p."""
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with block Strassen decomposition (rank 36)
    U, V, W = _build_block_decomposition()
    
    # Verify initial decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_full(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    if time.time() - _START_TIME < _TIME_LIMIT - 10:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_full(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
