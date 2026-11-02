import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U @ diag? No - verify tensor decomposition.
    The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    should equal 1 if j is the middle index (i.e., T[i,j,k] = delta_{j,j} = 1 for all valid).
    Actually for matrix multiplication: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    So T[i,j,k] should be 1 for all (i,j,k) in the valid range.
    """
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-tensor. Check all entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_remove_columns(U, V, W, n, m, p, max_cols_to_try=3, deadline=None):
    """Try to remove columns from the decomposition while maintaining validity.
    A column can be removed if it's a linear combination of other columns in the
    tensor sense. We check: for each column r, see if removing it still gives a valid
    decomposition (i.e., the tensor is still exactly decomposed by remaining columns).
    
    Actually, we need: the tensor T = sum_r u_r ⊗ v_r ⊗ w_r.
    If we remove column r, we need T = sum_{r'≠r} u_r' ⊗ v_r' ⊗ w_r' + adjustment.
    But we can't adjust - we need exact decomposition with fewer terms.
    
    Better approach: check if column r is "redundant" - i.e., the tensor decomposed
    without column r still equals T. This is equivalent to checking if the tensor
    T_r = u_r ⊗ v_r ⊗ w_r can be expressed as a sum of tensors from other columns.
    But that's not how it works - we need the SUM to remain the same.
    
    So: sum_{r'≠r} u_r'⊗v_r'⊗w_r' should equal T. But T = sum_all. 
    So we need: u_r⊗v_r⊗w_r = 0 as a tensor, which is only true if one factor is zero.
    
    Wait - that's not right either. The decomposition is:
    T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    If we remove column r, we get T' = T - u_r⊗v_r⊗w_r (as a tensor).
    For T' to also equal T (the multiplication tensor), we'd need u_r⊗v_r⊗w_r = 0.
    
    Hmm, that means we can only remove columns where one of the factors is identically zero.
    That's not useful.
    
    The correct approach: we need to find a DIFFERENT set of rank-1 tensors that sum to T.
    The column-removal approach works when we have a non-minimal decomposition and some
    columns are linearly dependent in a specific tensor sense.
    
    Actually, the right way: if we have T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r, and there
    exists a non-trivial linear relation among the "outer products" in some sense...
    
    Let me think differently. The standard approach for rank reduction:
    1. Start with a valid decomposition of rank R.
    2. Try removing one column at a time. For each column r, check if the remaining
       R-1 columns still decompose T. This is generally false unless the decomposition
       was not minimal.
    
    But there's a smarter approach: we can try to "absorb" a column into others by
    modifying the remaining columns. However, that changes the decomposition.
    
    The simplest valid approach: just check if any column has all-zero entries in one
    of its three factor vectors (making it contribute nothing), and remove it.
    
    For a more powerful search: try all pairs of columns and see if their difference
    or some linear combination can be zeroed out.
    
    Let me implement a practical search:
    1. Check for trivially removable columns (zero factor).
    2. For each pair (r,s), check if we can find scalars a,b such that
       a*u_r⊗v_r⊗w_r + b*u_s⊗v_s⊗w_s contributes zero to T when we adjust...
       This is getting complex.
       
    Simpler: just verify the current decomposition and return it. The rank-36 is already
    a valid answer. Try to see if we can find a rank-35 by checking specific structural
    symmetries.
    """
    if deadline is not None and time.time() > deadline:
        return U, V, W
    
    rank = U.shape[1]
    
    # Check for trivially zero columns
    for r in range(rank):
        if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
            # This column contributes nothing - remove it
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            # Restart search with new rank
            return _try_remove_columns(U, V, W, n, m, p, max_cols_to_try, deadline)
    
    # Try removing each column and checking validity
    for r in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 36 decomposition
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1
        W[w_idx(*w11)][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        W[w_idx(*w01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w10)][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] -= 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    def schoolbook_2x2x1(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z0, z1 = z_indices
        w0, w1 = w_indices
        
        # M1 = y00 * z0
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z0)][col] += 1
        W[w_idx(*w0)][col] += 1
        col += 1
        
        # M2 = y01 * z1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z1)][col] += 1
        W[w_idx(*w0)][col] += 1
        col += 1
        
        # M3 = y10 * z0
        U[u_idx(*y10)][col] += 1
        V[v_idx(*z0)][col] += 1
        W[w_idx(*w1)][col] += 1
        col += 1
        
        # M4 = y11 * z1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z1)][col] += 1
        W[w_idx(*w1)][col] += 1
        col += 1

    # Block 1: A_L * B_1 (rows 0,1 of B, cols 0,1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R * B_2 (rows 2,3 of B, cols 0,1)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L * B_3 (rows 0,1 of B, cols 2,3)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R * B_4 (rows 2,3 of B, cols 2,3)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L * B_5 (rows 0,1 of B, col 4)
    y5 = [(0,0), (0,1), (1,0), (1,1)]
    z5 = [(0,4), (1,4)]
    w5 = [(4,0), (4,1)]
    schoolbook_2x2x1(y5, z5, w5)
    
    # Block 6: A_R * B_6 (rows 2,3 of B, col 4)
    y6 = [(0,2), (0,3), (1,2), (1,3)]
    z6 = [(2,4), (3,4)]
    w6 = [(4,0), (4,1)]
    schoolbook_2x2x1(y6, z6, w6)
    
    # Verify the base decomposition
    assert _verify_decomposition(U, V, W, n, m, p), "Base decomposition failed verification"
    
    # Try to reduce rank by removing redundant columns
    if time.time() < deadline:
        U, V, W = _try_remove_columns(U, V, W, n, m, p, deadline=deadline)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return the original rank-36 decomposition
        U = np.zeros((n * m, 36), dtype=np.float64)
        V = np.zeros((m * p, 36), dtype=np.float64)
        W = np.zeros((p * n, 36), dtype=np.float64)
        col = 0
        strassen_block(y1, z1, w1)
        strassen_block(y2, z2, w2)
        strassen_block(y3, z3, w3)
        strassen_block(y4, z4, w4)
        schoolbook_2x2x1(y5, z5, w5)
        schoolbook_2x2x1(y6, z6, w6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
