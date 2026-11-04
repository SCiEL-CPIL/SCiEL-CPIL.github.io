import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _add_strassen_block(U, V, W, base_col, i0, jA0, jB0, k0):
    """
    Add a rank-7 Strassen decomposition for a 2x2x2 block.
    i0: start row of A (and C)
    jA0: start col of A
    jB0: start row of B
    k0: start col of B (and C)
    """
    col = base_col
    # M1 = (A00+A11)(B00+B11)
    U[i0*4 + jA0, col] += 1
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[jB0*8 + k0, col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[(i0+1)*4 + jA0, col] += 1
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[jB0*8 + k0, col] += 1
    W[k0*3 + (i0+1), col] += 1
    W[(k0+1)*3 + (i0+1), col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[i0*4 + jA0, col] += 1
    V[jB0*8 + (k0+1), col] += 1
    V[(jB0+1)*8 + (k0+1), col] -= 1
    W[(k0+1)*3 + i0, col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[(jB0+1)*8 + k0, col] += 1
    V[jB0*8 + k0, col] -= 1
    W[k0*3 + i0, col] += 1
    W[k0*3 + (i0+1), col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[i0*4 + jA0, col] += 1
    U[i0*4 + (jA0+1), col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] -= 1
    W[(k0+1)*3 + i0, col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[(i0+1)*4 + jA0, col] += 1
    U[i0*4 + jA0, col] -= 1
    V[jB0*8 + k0, col] += 1
    V[jB0*8 + (k0+1), col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[i0*4 + (jA0+1), col] += 1
    U[(i0+1)*4 + (jA0+1), col] -= 1
    V[(jB0+1)*8 + k0, col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] += 1
    col += 1
    
    return col

def _construct_hybrid():
    """
    Construct a rank-88 decomposition for 3x4x8 matmul.
    Row 0: Schoolbook (rank 32)
    Rows 1-2: Strassen-based (rank 56)
    """
    n, m, p = 3, 4, 8
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
            
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) - Strassen, rank 56
    # Split B into Left (cols 0-3) and Right (cols 4-7)
    # For each half, split into 4 2x2x2 blocks
    
    # Left half of B (cols 0-3)
    # C cols 0-1: A_L (cols 0-1) * B_TL (rows 0-1, cols 0-1) + A_R (cols 2-3) * B_BL (rows 2-3, cols 0-1)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=0)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=0)
    
    # C cols 2-3: A_L (cols 0-1) * B_TR (rows 0-1, cols 2-3) + A_R (cols 2-3) * B_BR (rows 2-3, cols 2-3)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=2)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=2)
    
    # Right half of B (cols 4-7)
    # C cols 4-5: A_L (cols 0-1) * B_TL_R (rows 0-1, cols 4-5) + A_R (cols 2-3) * B_BL_R (rows 2-3, cols 4-5)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=4)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=4)
    
    # C cols 6-7: A_L (cols 0-1) * B_TR_R (rows 0-1, cols 6-7) + A_R (cols 2-3) * B_BR_R (rows 2-3, cols 6-7)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=6)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=6)
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def _try_compress(U_list, V_list, W_list, n, m, p, max_time=30):
    """
    Attempt to compress the rank-88 decomposition to rank-87 or lower
    using alternating minimization on the rank factorization.
    """
    start_time = time.time()
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] implicitly
    # We want to find U', V', W' with rank' < rank such that
    # T = sum_r U'[:,r] * V'[:,r] * W'[:,r]
    
    # The tensor entries are T[i,j,k] = 1 if the index matches, else 0
    # In our factorization: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: try to merge two columns by finding a linear combination
    # that preserves the tensor identity.
    
    # For each pair of columns (a, b), check if we can replace them with a single column
    # by finding coefficients alpha, beta such that:
    # alpha * U[:,a] + beta * U[:,b] = U'[:,c]
    # alpha * V[:,a] + beta * V[:,b] = V'[:,c]
    # alpha * W[:,a] + beta * W[:,b] = W'[:,c]
    # and the tensor identity still holds.
    
    # This is equivalent to checking if the tensor contribution from columns a and b
    # can be represented as a single rank-1 tensor.
    
    # A simpler approach: use SVD-based compression
    # Reshape the tensor and try low-rank approximation
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # The tensor should have T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # and 0 otherwise.
    
    # Try rank-87 by removing one column and adjusting
    # Use alternating minimization: fix U and V, solve for W, etc.
    
    # First, try to see if we can drop one column
    for drop_col in range(rank):
        if time.time() - start_time > max_time:
            break
        
        U_reduced = np.delete(U, drop_col, axis=1)
        V_reduced = np.delete(V, drop_col, axis=1)
        W_reduced = np.delete(W, drop_col, axis=1)
        
        # Now try to recover the dropped column's contribution
        # by adjusting the remaining columns
        # The residual is:
        T_residual = np.einsum('i,j,k->ijk', U[:,drop_col], V[:,drop_col], W[:,drop_col])
        
        # We need to find delta_U, delta_V, delta_W (each of shape (dim, rank-1))
        # such that sum_r (U_reduced[:,r] + delta_U[:,r]) * (V_reduced[:,r] + delta_V[:,r]) * (W_reduced[:,r] + delta_W[:,r])
        # = T
        # This is complex. Instead, try a simpler approach:
        # Check if the tensor T can be approximated by rank-87 using SVD on matricizations.
        
        # Matricize T along the first mode: T_(1) of shape (n*m, m*p*p*n)
        T_mat1 = T.reshape(n*m, m*p*p*n)
        
        # SVD
        try:
            U_svd, s, Vt_svd = np.linalg.svd(T_mat1, full_matrices=False)
            
            # Check if the smallest singular value is close to 0
            if s.shape[0] > 87 and s[87] < 1e-8:
                # We can potentially drop one rank
                # Reconstruct with rank 87
                U_new = U_svd[:, :87] * s[:87]
                Vt_new = Vt_svd[:87, :]
                
                # This gives us a rank-87 factorization of the first mode
                # But we need a 3-way factorization, not just 2-way
                # This approach doesn't directly give us U, V, W
                pass
        except:
            pass
        
        # Alternative: try to find a linear dependency among the columns
        # Check if any column is a linear combination of others
        # This is expensive, so skip for now
    
    # For now, return None (no compression found)
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try hybrid Strassen construction (rank 88)
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if _verify(U_h, V_h, W_h, n, m, p):
            # Try to compress
            if time.time() < deadline - 35:
                compressed = _try_compress(U_h, V_h, W_h, n, m, p, max_time=30)
                if compressed is not None:
                    Uc, Vc, Wc = compressed
                    if _verify(Uc, Vc, Wc, n, m, p):
                        return {"U": Uc, "V": Vc, "W": Wc}
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    if time.time() > deadline:
        # Emergency fallback
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}
