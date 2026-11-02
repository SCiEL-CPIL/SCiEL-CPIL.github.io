"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _build_base_decomposition(n, m, p):
    """Build the base rank-36 decomposition and return U, V, W as numpy arrays."""
    rank = 36
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1
        U[u_idx_00, col] += 1
        U[u_idx_11, col] += 1
        V[v_idx_00, col] += 1
        V[v_idx_11, col] += 1
        W[w_idx_00, col] += 1
        W[w_idx_11, col] += 1
        col += 1
        
        # M2
        U[u_idx_10, col] += 1
        U[u_idx_11, col] += 1
        V[v_idx_00, col] += 1
        W[w_idx_10, col] += 1
        W[w_idx_11, col] -= 1
        col += 1
        
        # M3
        U[u_idx_00, col] += 1
        V[v_idx_01, col] += 1
        V[v_idx_11, col] -= 1
        W[w_idx_01, col] += 1
        W[w_idx_11, col] += 1
        col += 1
        
        # M4
        U[u_idx_11, col] += 1
        V[v_idx_10, col] += 1
        V[v_idx_00, col] -= 1
        W[w_idx_00, col] += 1
        W[w_idx_10, col] += 1
        col += 1
        
        # M5
        U[u_idx_00, col] += 1
        U[u_idx_01, col] += 1
        V[v_idx_11, col] += 1
        W[w_idx_00, col] -= 1
        W[w_idx_01, col] += 1
        col += 1
        
        # M6
        U[u_idx_10, col] += 1
        U[u_idx_00, col] -= 1
        V[v_idx_00, col] += 1
        V[v_idx_01, col] += 1
        W[w_idx_11, col] += 1
        col += 1
        
        # M7
        U[u_idx_01, col] += 1
        U[u_idx_11, col] -= 1
        V[v_idx_10, col] += 1
        V[v_idx_11, col] += 1
        W[w_idx_00, col] += 1
        col += 1

    # Part 1: C1 = A B1 (2x2)
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Part 2: C21 = A B21 (2x1) schoolbook
    for i in range(2):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Part 3: C22 = A B22 (2x2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # The tensor T[i,j,k] = delta[i,j] * delta[j,k] should be reconstructed as sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # U shape: (n*m, rank) indexed by (i*m + j)
    # V shape: (m*p, rank) indexed by (j*p + k)
    # W shape: (p*n, rank) indexed by (k*n + i)
    
    rank = U.shape[1]
    
    # Build the full tensor from the decomposition
    # T_decomposed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i == j and j == k) else 0.0
                # Wait, that's not right. The matrix multiplication tensor is:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
                # Actually no. The tensor is defined such that:
                # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
                # where T[i,j,k] = 1 for all valid i,j,k.
                # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Let me re-read the convention: W[k*n+i, r]
                # So W is indexed by (k*n + i) where k is the column of B / row of C, i is the row of A / column of C
                
                # The reconstruction is:
                # For each output C[i,k]:
                #   C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This should equal sum_j A[i,j] * B[j,k]
                
                # So the tensor entry T[i,j,k] should be 1 for all i,j,k
                # And T_decomposed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                decomp_val = 0.0
                for r in range(rank):
                    decomp_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                
                if abs(decomp_val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using tensor contraction."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U3 = U.reshape(n, m, rank)  # U[i,j,r]
    V3 = V.reshape(m, p, rank)  # V[j,k,r]
    W3 = W.reshape(p, n, rank)  # W[k,i,r]
    
    # T_decomposed[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T_decomposed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Expected: T[i,j,k] = 1 for all i,j,k
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_decomposed, T_expected, atol=tol):
        return False
    return True

def _try_remove_columns(U, V, W, n, m, p, max_remove=3, deadline=None):
    """Try to remove up to max_remove columns to reduce rank while maintaining validity."""
    rank = U.shape[1]
    
    if deadline is not None and time.time() > deadline - 5:
        return U, V, W
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing single columns
    for r in range(rank):
        if deadline is not None and time.time() > deadline - 10:
            break
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
            best_rank = rank - 1
            break
    
    # If we found one removable column, try to find more
    if best_rank < rank and max_remove >= 2:
        if deadline is not None and time.time() > deadline - 15:
            return best_U, best_V, best_W
        
        current_U, current_V, current_W = best_U, best_V, best_W
        current_rank = best_rank
        for _ in range(max_remove - 1):
            if deadline is not None and time.time() > deadline - 15:
                break
            found = False
            for r in range(current_rank):
                U_test = np.delete(current_U, r, axis=1)
                V_test = np.delete(current_V, r, axis=1)
                W_test = np.delete(current_W, r, axis=1)
                
                if _verify_fast(U_test, V_test, W_test, n, m, p):
                    current_U, current_V, current_W = U_test, V_test, W_test
                    current_rank -= 1
                    found = True
                    break
            if not found:
                break
        
        if current_rank < best_rank:
            best_U, best_V, best_W = current_U, current_V, current_W
            best_rank = current_rank
    
    return best_U, best_V, best_W

def _try_remove_pairs(U, V, W, n, m, p, deadline=None):
    """Try to remove pairs of columns simultaneously."""
    rank = U.shape[1]
    
    if deadline is not None and time.time() > deadline - 20:
        return U, V, W
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try all pairs
    for r1 in range(rank):
        if deadline is not None and time.time() > deadline - 30:
            break
        for r2 in range(r1 + 1, rank):
            if deadline is not None and time.time() > deadline - 30:
                break
            
            # Remove both columns
            cols_to_keep = [c for c in range(rank) if c != r1 and c != r2]
            U_test = U[:, cols_to_keep]
            V_test = V[:, cols_to_keep]
            W_test = W[:, cols_to_keep]
            
            if _verify_fast(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                best_rank = rank - 2
                return best_U, best_V, best_W
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Build base decomposition
    U, V, W = _build_base_decomposition(n, m, p)
    
    # Verify base decomposition
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback: return schoolbook if base is broken (shouldn't happen)
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to remove columns
    U, V, W = _try_remove_columns(U, V, W, n, m, p, max_remove=3, deadline=deadline)
    
    # If we have time, try pair removals on the reduced result
    if time.time() < deadline - 30:
        U, V, W = _try_remove_pairs(U, V, W, n, m, p, deadline=deadline)
    
    # Final verification
    if not _verify_fast(U, V, W, n, m, p):
        # Rebuild and return base if something went wrong
        U, V, W = _build_base_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
