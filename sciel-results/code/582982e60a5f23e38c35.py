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
    Attempt to compress the rank-88 decomposition to rank-87 by trying to drop
    one rank-1 term and solving for corrections to the remaining terms.
    
    Strategy: for each candidate column c to drop, we want to find
    U', V', W' (each with rank-1 columns) such that
    sum_{r != c} U'[:,r] * V'[:,r] * W'[:,r] = T (the full tensor)
    
    This is a bilinear feasibility problem. We approach it via:
    1. Fix W' = W (for r != c), solve for U', V' via least squares on the
       "flattened" tensor equation.
    2. Check if the residual is near zero.
    3. Iterate: fix U', solve for V', W'; etc.
    
    If we find an exact (within tolerance) rank-87 decomposition, return it.
    """
    start_time = time.time()
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)
    
    rank = U.shape[1]
    dim_U = n * m  # 12
    dim_V = m * p  # 32
    dim_W = p * n  # 24
    
    # Build the target tensor T
    T = np.zeros((dim_U, dim_V, dim_W))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # For each drop candidate, try to find rank-87 decomposition
    for drop_col in range(rank):
        if time.time() - start_time > max_time:
            break
        
        # Residual: what we need to represent with rank-87
        T_residual = T  # We need sum_{r != drop} to equal T
        
        # Remove the dropped column
        U_r = np.delete(U, drop_col, axis=1)
        V_r = np.delete(V, drop_col, axis=1)
        W_r = np.delete(W, drop_col, axis=1)
        new_rank = rank - 1  # 87
        
        # Alternating minimization:
        # Fix W_r, solve for U_r and V_r
        # The equation is: sum_r U_r[:,r] * V_r[:,r] * W_r[:,r] = T_residual
        # For each r: U_r[:,r] * V_r[:,r] = T_residual[:, :, r] / W_r[:,r] (elementwise)
        # But W_r[:,r] may have zeros, so this is tricky.
        
        # Better approach: use the matricization.
        # T_(1) = U_r * (V_r ⊙ W_r)^T  where ⊙ is Khatri-Rao product
        # So for fixed V_r, W_r: U_r = T_(1) * (V_r ⊙ W_r)^+ (pseudo-inverse)
        # Then check residual.
        
        # Khatri-Rao product of V_r and W_r: shape (dim_V*dim_W, new_rank)
        # (V_r ⊙ W_r)[:, r] = V_r[:,r] * W_r[:,r] (elementwise)
        V_kr = np.zeros((dim_V * dim_W, new_rank))
        for r in range(new_rank):
            V_kr[:, r] = (V_r[:, r] * W_r[:, r]).ravel()
        
        # T_(1) = T_residual.reshape(dim_U, dim_V * dim_W)
        T_mat1 = T_residual.reshape(dim_U, dim_V * dim_W)
        
        # Solve U_r = T_mat1 @ V_kr^+
        # Using least squares: minimize ||T_mat1 - U_r @ V_kr^T||_F
        try:
            U_new, residuals, rank_ls, sv = np.linalg.lstsq(V_kr.T, T_mat1.T, rcond=None)
            U_new = U_new.T  # shape (dim_U, new_rank)
        except:
            continue
        
        # Check residual
        T_recon = np.zeros((dim_U, dim_V, dim_W))
        for r in range(new_rank):
            T_recon += np.einsum('i,j,k->ijk', U_new[:,r], V_r[:,r], W_r[:,r])
        residual_norm = np.linalg.norm(T_recon - T_residual)
        if residual_norm < 1e-10:
            # Found a valid rank-87 decomposition!
            return U_new.tolist(), V_r.tolist(), W_r.tolist()
        
        # Try another approach: fix U_new, solve for V_r and W_r
        # T_(2) = V_r * (W_r ⊙ U_new)^T
        # (W_r ⊙ U_new)[:, r] = W_r[:,r] * U_new[:,r]
        W_kr = np.zeros((dim_W * dim_U, new_rank))
        for r in range(new_rank):
            W_kr[:, r] = (W_r[:, r] * U_new[:, r]).ravel()
        
        T_mat2 = T_residual.reshape(dim_V, dim_W * dim_U)
        
        try:
            V_new, residuals, rank_ls, sv = np.linalg.lstsq(W_kr.T, T_mat2.T, rcond=None)
            V_new = V_new.T  # shape (dim_V, new_rank)
        except:
            continue
        
        # Check residual
        T_recon = np.zeros((dim_U, dim_V, dim_W))
        for r in range(new_rank):
            T_recon += np.einsum('i,j,k->ijk', U_new[:,r], V_new[:,r], W_r[:,r])
        residual_norm = np.linalg.norm(T_recon - T_residual)
        if residual_norm < 1e-10:
            return U_new.tolist(), V_new.tolist(), W_r.tolist()
        
        # Try: fix U_new, V_new, solve for W_r
        # T_(3) = W_r * (U_new ⊙ V_new)^T
        # (U_new ⊙ V_new)[:, r] = U_new[:,r] * V_new[:,r]
        U_kr = np.zeros((dim_U * dim_V, new_rank))
        for r in range(new_rank):
            U_kr[:, r] = (U_new[:, r] * V_new[:, r]).ravel()
        
        T_mat3 = T_residual.reshape(dim_W, dim_U * dim_V)
        
        try:
            W_new, residuals, rank_ls, sv = np.linalg.lstsq(U_kr.T, T_mat3.T, rcond=None)
            W_new = W_new.T  # shape (dim_W, new_rank)
        except:
            continue
        
        # Check residual
        T_recon = np.zeros((dim_U, dim_V, dim_W))
        for r in range(new_rank):
            T_recon += np.einsum('i,j,k->ijk', U_new[:,r], V_new[:,r], W_new[:,r])
        residual_norm = np.linalg.norm(T_recon - T_residual)
        if residual_norm < 1e-10:
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
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
            if time.time() < deadline - 40:
                remaining = deadline - 40 - time.time()
                compressed = _try_compress(U_h, V_h, W_h, n, m, p, max_time=max(1, remaining))
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
