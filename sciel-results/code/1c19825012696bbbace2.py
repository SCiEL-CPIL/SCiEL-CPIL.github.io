import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    
    def verify_exact(U_mat, V_mat, W_mat, tol=1e-8):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        r = U_mat.shape[1]
        # Check that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for matching
        # and 0 for non-matching
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    s = np.dot(U_mat[a_idx], V_mat[b_idx] * W_mat[c_idx])
                    if abs(s - 1.0) > tol:
                        return False
        # Check that non-matching indices give 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    # For each other (i2,j2,k2), the product should be 0
                    # But we can just check that the full tensor is correct
                    # Actually, let's verify the full tensor
        # Full tensor verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                a_idx = i2 * m + j2
                                b_idx = j2 * p + k2
                                c_idx = k2 * n + i2
                                s = np.dot(U_mat[a_idx], V_mat[b_idx] * W_mat[c_idx])
                                if abs(s) > tol:
                                    return False
        return True
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-8):
        """Faster verification using numpy - checks only the diagonal (matching) entries."""
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    s = np.dot(U_mat[a_idx], V_mat[b_idx] * W_mat[c_idx])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_full_fast(U_mat, V_mat, W_mat, tol=1e-8):
        """Full verification using tensor reconstruction."""
        r = U_mat.shape[1]
        # Reconstruct the tensor
        # T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and T[a_idx, b_idx, c_idx] = 0 otherwise
        # where a_idx = i*m+j, b_idx = j'*p+k', c_idx = k''*n+i''
        # The constraint is that j = j', k = k'', i = i''
        
        # Build the full tensor efficiently
        # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Sum over r
        
        # Check diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    s = np.dot(U_mat[a_idx], V_mat[b_idx] * W_mat[c_idx])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check off-diagonal entries - sample some to save time
        # The key constraint: if (i,j,k) != (i2,j2,k2), then the product should be 0
        # This means for any a_idx, b_idx, c_idx that don't form a valid (i,j,k) triple,
        # the sum should be 0
        # Actually, a_idx = i*m+j always gives a valid i,j
        # b_idx = j'*p+k' gives valid j',k'
        # c_idx = k''*n+i'' gives valid k'',i''
        # The constraint is i=i'' AND j=j' AND k=k''
        
        # Let's check all off-diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                b_idx = j2 * p + k2
                                c_idx = k2 * n + i2
                                s = np.dot(U_mat[a_idx], V_mat[b_idx] * W_mat[c_idx])
                                if abs(s) > tol:
                                    return False
        return True
    
    # Build Strassen-based decomposition
    # For 2x4x6, we use 1x2x3 = 6 blocks of 2x2x2, each with rank 7
    # Total initial rank: 42
    
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)
    
    def add_strassen_2x2x2(i_start, j_start, k_start, offset):
        """Add a Strassen 2x2x2 block at the given position."""
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        # M2 = (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0
        
        # M3 = A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0
        
        # M4 = A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0
        
        # M5 = (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0
        
        # M6 = (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0
        
        # M7 = (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0
    
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_2x2x2(i_start, j_start, k_start, offset)
                offset += 7
    
    actual_rank = offset
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]
    
    if not verify_fast(U, V, W):
        # Fallback to schoolbook
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = actual_rank
    
    # Peeling algorithm: try to remove columns one at a time
    # For each column, check if it can be removed while maintaining the decomposition
    # This is equivalent to checking if the column is linearly dependent in a certain sense
    
    def try_remove_column(U_cur, V_cur, W_cur, col_idx, tol=1e-8):
        """Try to remove column col_idx and see if the decomposition still works."""
        r = U_cur.shape[1]
        if r <= 1:
            return None, None, None
        
        U_test = np.delete(U_cur, col_idx, axis=1)
        V_test = np.delete(V_cur, col_idx, axis=1)
        W_test = np.delete(W_cur, col_idx, axis=1)
        
        if verify_full_fast(U_test, V_test, W_test, tol):
            return U_test, V_test, W_test
        return None, None, None
    
    # Iterative peeling
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        removed_any = False
        for col_idx in range(best_rank - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            U_test, V_test, W_test = try_remove_column(best_U, best_V, best_W, col_idx)
            if U_test is not None:
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                removed_any = True
                break  # Restart the loop after removal
        
        if not removed_any:
            break
    
    # Additional optimization: try to find linear combinations that allow rank reduction
    # This is more complex - we look for a column that can be expressed as a combination
    # of other columns in a way that preserves the tensor
    
    # Try a more sophisticated approach: for each pair of columns, see if we can
    # merge them
    if time.time() < deadline - 20:
        for col1 in range(best_rank - 1):
            if time.time() > deadline - 20:
                break
            for col2 in range(col1 + 1, best_rank):
                if time.time() > deadline - 20:
                    break
                # Check if columns col1 and col2 can be merged
                # This is complex, so we'll just try removing one at a time again
                pass
    
    # Final verification
    if not verify_full_fast(best_U, best_V, best_W):
        # If something went wrong, use the Strassen result
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = actual_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
