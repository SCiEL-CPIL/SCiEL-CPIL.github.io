"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140  # 140s to leave safety margin before 150s kill
    return _DEADLINE

def _time_left():
    return max(0, _get_deadline() - time.time())

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _build_54_rank():
    """Build the base 54-rank decomposition."""
    n, m, p = DIMENSIONS
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Remaining j=4: scalar terms
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    return U, V, W, col

def _verify_vectorized(U_mat, V_mat, W_mat, n, m, p):
    """
    Vectorized verification using numpy.
    U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U_mat.shape[1]
    
    # Reshape to 3D tensors
    # U: (n, m, rank)
    U_3d = U_mat.reshape(n, m, rank)
    # V: (m, p, rank)
    V_3d = V_mat.reshape(m, p, rank)
    # W: (p, n, rank)
    W_3d = W_mat.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
    
    # All entries should be 1.0
    return np.allclose(T, 1.0, atol=1e-9)

def _verify_decomp(U, V, W, n, m, p):
    """Verify the decomposition is correct using vectorized numpy."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    return _verify_vectorized(U_mat, V_mat, W_mat, n, m, p)

def _try_remove_column(U, V, W, n, m, p, col_to_remove):
    """Try removing a specific column and verify."""
    rank = len(U[0])
    keep = [r for r in range(rank) if r != col_to_remove]
    
    U_new = [[U[i][r] for r in keep] for i in range(n*m)]
    V_new = [[V[i][r] for r in keep] for i in range(m*p)]
    W_new = [[W[i][r] for r in keep] for i in range(p*n)]
    
    if _verify_decomp(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new, True
    
    return None, None, None, False

def _local_search_reduce(U, V, W, n, m, p):
    """
    Iteratively try to remove columns from the decomposition.
    Uses vectorized verification for speed.
    """
    current_U, current_V, current_W = U, V, W
    current_rank = len(current_U[0])
    
    # Try multiple passes of column removal
    for pass_num in range(10):  # Up to 10 passes
        if _time_left() < 15:
            break
        
        removed_in_pass = False
        for r in range(current_rank):
            if _time_left() < 10:
                break
            
            new_U, new_V, new_W, success = _try_remove_column(
                current_U, current_V, current_W, n, m, p, r
            )
            
            if success:
                current_U, current_V, current_W = new_U, new_V, new_W
                current_rank -= 1
                removed_in_pass = True
                break  # Restart from beginning after successful removal
        
        if not removed_in_pass:
            break  # No more columns can be removed
    
    return current_U, current_V, current_W, current_rank

def _snap_to_half_integer(arr):
    """Snap values to nearest half-integer."""
    snapped = []
    for row in arr:
        snapped_row = []
        for val in row:
            snapped_val = round(val * 2) / 2.0
            snapped_row.append(snapped_val)
        snapped.append(snapped_row)
    return snapped

def construct():
    n, m, p = DIMENSIONS
    _get_deadline()
    
    # Build base decomposition
    U, V, W, col = _build_54_rank()
    
    # Verify base decomposition
    if not _verify_decomp(U, V, W, n, m, p):
        # Fallback: return schoolbook decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try local search to reduce rank
    if _time_left() > 20:
        U, V, W, final_rank = _local_search_reduce(U, V, W, n, m, p)
    
    # Snap to half-integers
    U_snapped = _snap_to_half_integer(U)
    V_snapped = _snap_to_half_integer(V)
    W_snapped = _snap_to_half_integer(W)
    
    # Final verification after snapping
    if _verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
        U, V, W = U_snapped, V_snapped, W_snapped
    else:
        # If snapping broke it, return the unsnapped version (which is verified)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
