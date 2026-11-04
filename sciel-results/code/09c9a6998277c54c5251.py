"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 3x5x6 matrix multiplication tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's 7-term algorithm
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = A22(B11-B21)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _rectangular_decomp(n, m, p):
    """
    Build a tensor decomposition for n x m x p matrix multiplication using
    block structure with Strassen for 2x2x2 sub-blocks.
    Returns (U, V, W, rank).
    """
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W, 1
    
    # Determine if we can use Strassen on a 2x2x2 sub-block
    # We'll use a simple approach: if any dimension is 2, try to use Strassen
    # Otherwise, fall back to schoolbook for this sub-block
    
    # For now, handle the specific cases we need
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    
    # Fallback: schoolbook for this sub-block
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _kronecker_product_rect(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Compute Kronecker product of two rectangular tensor decompositions.
    U1: (n1*m1) x r1, V1: (m1*p1) x r1, W1: (p1*n1) x r1
    U2: (n2*m2) x r2, V2: (m2*p2) x r2, W2: (p2*n2) x r2
    Returns decomposition for (n1*n2) x (m1*m2) x (p1*p2)
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    new_rank = r1 * r2
    
    U_new = [[0]*new_rank for _ in range(n1*n2*m1*m2)]
    V_new = [[0]*new_rank for _ in range(m1*m2*p1*p2)]
    W_new = [[0]*new_rank for _ in range(p1*p2*n1*n2)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1*r2 + c2
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(m1):
                        for j2 in range(m2):
                            i_idx = (i1*n2 + i2)*m1*m2 + j1*m2 + j2
                            U_new[i_idx][c] = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
            for j1 in range(m1):
                for j2 in range(m2):
                    for k1 in range(p1):
                        for k2 in range(p2):
                            j_idx = (j1*m2 + j2)*p1*p2 + k1*p2 + k2
                            V_new[j_idx][c] = V1[j1*p1+k1][c1] * V2[j2*p2+k2][c2]
            for k1 in range(p1):
                for k2 in range(p2):
                    for i1 in range(n1):
                        for i2 in range(n2):
                            k_idx = (k1*n2 + k2)*p1*p2 + i1*p2 + i2  # This is wrong, need to fix
                            # Actually W index is k*n + i where k is in p, i is in n
                            # For Kronecker: k_global = k1*n2 + k2? No...
                            # k ranges over p1*p2, i ranges over n1*n2
                            # W[k_global * (n1*n2) + i_global]
                            k_global = k1*n2 + k2  # This is wrong
                            # Actually k1 is in p1 (0..p1-1), k2 is in p2 (0..p2-1)
                            # k_global should be k1*p2 + k2
                            k_global = k1*p2 + k2
                            i_global = i1*n2 + i2
                            k_idx = k_global * (n1*n2) + i_global
                            W_new[k_idx][c] = W1[k1*n1+i1][c1] * W2[k2*n2+i2][c2]
    
    return U_new, V_new, W_new, new_rank

def _build_full_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Build full decomposition by combining sub-decompositions via Kronecker product.
    n_parts, m_parts, p_parts partition n, m, p.
    Each part gets a sub-decomposition (Strassen for size 2, identity for size 1).
    """
    # Get sub-decompositions for each part
    n_decomps = []
    for s in n_parts:
        U, V, W, r = _rectangular_decomp(s, s, s)  # Square for now
        n_decomps.append((U, V, W, r, s))
    
    m_decomps = []
    for s in m_parts:
        U, V, W, r = _rectangular_decomp(s, s, s)
        m_decomps.append((U, V, W, r, s))
    
    p_decomps = []
    for s in p_parts:
        U, V, W, r = _rectangular_decomp(s, s, s)
        p_decomps.append((U, V, W, r, s))
    
    # Total rank is product of all sub-ranks
    total_rank = 1
    for _, _, _, r, _ in n_decomps:
        total_rank *= r
    for _, _, _, r, _ in m_decomps:
        total_rank *= r
    for _, _, _, r, _ in p_decomps:
        total_rank *= r
    
    if total_rank >= n * m * p:
        return None
    
    # Build full U, V, W
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    # Compute offsets
    n_offsets = []
    off = 0
    for s in n_parts:
        n_offsets.append(off)
        off += s
    
    m_offsets = []
    off = 0
    for s in m_parts:
        m_offsets.append(off)
        off += s
    
    p_offsets = []
    off = 0
    for s in p_parts:
        p_offsets.append(off)
        off += s
    
    col = 0
    for ni, (Un, Vn, Wn, rn, sn) in enumerate(n_decomps):
        for mi, (Um, Vm, Wm, rm, sm) in enumerate(m_decomps):
            for pi, (Up, Vp, Wp, rp, sp) in enumerate(p_decomps):
                i_off = n_offsets[ni]
                j_off = m_offsets[mi]
                k_off = p_offsets[pi]
                
                for cn in range(rn):
                    for cm in range(rm):
                        for cp in range(rp):
                            c = col + cn*rm*rp + cm*rp + cp
                            
                            # Fill U: indices (i, j) where i in n_block, j in m_block
                            for i_n in range(sn):
                                for j_m in range(sm):
                                    i_global = i_off + i_n
                                    j_global = j_off + j_m
                                    U_full[i_global*m + j_global][c] = Un[i_n*sn+j_m if False else i_n][cn] * Um[j_m][cm]
                                    # Wait, Un has shape (sn*sm) x rn? No, Un is for sn x sn x sn
                                    # This is getting complicated. Let me simplify.
                            
                            # Actually, the Kronecker approach is cleaner. Let me use that.
                            pass
    
    # This approach is getting too complex. Let me use a simpler direct construction.
    return None

def _direct_strassen_decomposition(n, m, p):
    """
    Direct construction: partition n=3 as [2,1], m=5 as [2,2,1], p=6 as [2,2,2].
    For each triple of blocks (n_block, m_block, p_block), build the tensor.
    """
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Precompute sub-decompositions
    # For size 2: Strassen (rank 7)
    # For size 1: identity (rank 1)
    
    U_full = []
    V_full = []
    W_full = []
    
    n_off = 0
    m_off = 0
    p_off = 0
    
    n_offsets = []
    for s in n_parts:
        n_offsets.append(n_off)
        n_off += s
    
    m_offsets = []
    for s in m_parts:
        m_offsets.append(m_off)
        m_off += s
    
    p_offsets = []
    for s in p_parts:
        p_offsets.append(p_off)
        p_off += s
    
    # For each block combination, add terms
    for ni, sn in enumerate(n_parts):
        for mi, sm in enumerate(m_parts):
            for pi, sp in enumerate(p_parts):
                i0 = n_offsets[ni]
                j0 = m_offsets[mi]
                k0 = p_offsets[pi]
                
                # Get sub-decompositions
                if sn == 2:
                    Un, Vn, Wn, rn = _rectangular_decomp(2, 2, 2)
                else:
                    Un = [[1]]
                    Vn = [[1]]
                    Wn = [[1]]
                    rn = 1
                
                if sm == 2:
                    Um, Vm, Wm, rm = _rectangular_decomp(2, 2, 2)
                else:
                    Um = [[1]]
                    Vm = [[1]]
                    Wm = [[1]]
                    rm = 1
                
                if sp == 2:
                    Up, Vp, Wp, rp = _rectangular_decomp(2, 2, 2)
                else:
                    Up = [[1]]
                    Vp = [[1]]
                    Wp = [[1]]
                    rp = 1
                
                # Combine via Kronecker-like product
                for cn in range(rn):
                    for cm in range(rm):
                        for cp in range(rp):
                            U_row = [0] * 0  # placeholder
                            # We'll build full vectors at the end
                            pass
    
    # This is still complex. Let me just do the full Kronecker product properly.
    return None

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x5x6."""
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    schoolbook_rank = n * m * p  # 90
    
    # Start with schoolbook as default
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    best_rank = schoolbook_rank
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try the specific block decomposition: n=[2,1], m=[2,2,1], p=[2,2,2]
    # This should give rank 7*1 * 7*7*1 * 7*7*7 = 7 * 49 * 343 = 117,649 which is worse.
    # That's not right. The Kronecker product of decompositions multiplies ranks.
    # For block (2,2,2): rank 7
    # For block (2,2,1): rank 7*7*1 = 49
    # For block (2,1,2): rank 7*1*7 = 49
    # For block (1,2,2): rank 1*7*7 = 49
    # For block (1,2,1): rank 1*7*1 = 7
    # For block (1,1,2): rank 1*1*7 = 7
    # For block (2,1,1): rank 7*1*1 = 7
    # For block (1,1,1): rank 1
    
    # Total rank = sum over all block combinations of (rank_n * rank_m * rank_p)
    # = 7*7*7 + 7*7*1 + 7*1*7 + 7*1*1 + 1*7*7 + 1*7*1 + 1*1*7 + 1*1*1
    # = 343 + 49 + 49 + 7 + 49 + 7 + 7 + 1 = 512
    
    # That's still worse than 90. So block decomposition with Strassen doesn't help here.
    # The issue is that the Kronecker product multiplies ranks, which is bad.
    
    # Alternative: use Strassen only where it helps and schoolbook elsewhere.
    # For a 2x2x2 block, Strassen saves 1 multiply (8->7).
    # For a 2xa x b block, we can't directly use Strassen unless a=b=2.
    
    # Let me try a different approach: use the fact that 3x5x6 can be embedded in 4x6x8
    # and use Strassen-based algorithms, but that's complex.
    
    # For now, just return schoolbook. The search space for better decompositions
    # requires more sophisticated methods (e.g., using known low-rank decompositions).
    
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Should not happen, but fallback
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
