"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140  # 140s safety margin

def _build_full_tensor(n, m, p):
    """Build the full matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1
    return T

def _verify_decomposition(U, V, W, T_exp):
    """Verify if U, V, W exactly decompose T_exp."""
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    # Convert to numpy if lists
    if isinstance(U, list):
        U = np.array(U)
    if isinstance(V, list):
        V = np.array(V)
    if isinstance(W, list):
        W = np.array(W)
    
    T_check = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.array_equal(T_check, T_exp)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses a rank-44 construction based on 4 blocks of (Strassen 7 + Schoolbook 4).
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the expected tensor for verification
    T_exp = _build_full_tensor(n, m, p)
    
    # Fallback: Schoolbook rank
    rank_school = n * m * p  # 48
    U_fb = np.zeros((n*m, rank_school), dtype=np.int64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.int64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    # Primary: Rank 44 Construction
    # Structure: 4 blocks of p=2.
    # Each block:
    #   - Strassen (rank 7) for A[:, 0:2] * B[0:2, :]
    #   - Schoolbook (rank 4) for A[:, 2] * B[2, :]
    
    rank_44 = 44
    U = np.zeros((n*m, rank_44), dtype=np.int64)
    V = np.zeros((m*p, rank_44), dtype=np.int64)
    W = np.zeros((p*n, rank_44), dtype=np.int64)
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part for A[:, 0:2] (2x2) * B[0:2, :] (2x2) ---
        # A_sub indices: 
        # A00 = U[0], A01 = U[1]
        # A10 = U[3], A11 = U[4]
        # B_sub indices:
        # B0k0 = V[k0], B0k1 = V[k1]
        # B1k0 = V[p+k0], B1k1 = V[p+k1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        V[p+k1, col] = 1
        W[k0*n+0, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        W[k0*n+1, col] = 1
        W[k1*n+1, col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0, col] = 1
        V[k1, col] = 1
        V[p+k1, col] = -1
        W[k1*n+0, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4, col] = 1
        V[p+k0, col] = 1
        V[k0, col] = -1
        W[k0*n+0, col] = 1
        W[k0*n+1, col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0, col] = 1
        U[1, col] = 1
        V[p+k1, col] = 1
        W[k0*n+0, col] = -1
        W[k1*n+0, col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3, col] = 1
        U[0, col] = -1
        V[k0, col] = 1
        V[k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1, col] = 1
        U[4, col] = -1
        V[p+k0, col] = 1
        V[p+k1, col] = 1
        W[k0*n+0, col] = 1
        col += 1
        
        # --- Schoolbook part for A[:, 2] (2x1) * B[2, :] (1x2) ---
        # A_tail indices:
        # A02 = U[2], A12 = U[5]
        # B_tail indices:
        # B2k0 = V[2p+k0], B2k1 = V[2p+k1]
        
        # P1 = A02 * B2k0
        U[2, col] = 1
        V[2*p+k0, col] = 1
        W[k0*n+0, col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[2, col] = 1
        V[2*p+k1, col] = 1
        W[k1*n+0, col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[5, col] = 1
        V[2*p+k0, col] = 1
        W[k0*n+1, col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[5, col] = 1
        V[2*p+k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1

    # Verify the rank-44 solution
    if _verify_decomposition(U, V, W, T_exp):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
# EVOLVE-BLOCK-END
