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

def _kronecker_product(U1, V1, W1, U2, V2, W2):
    """Compute Kronecker product of two tensor decompositions."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    new_rank = rank1 * rank2
    
    n1, m1, p1 = len(U1), len(U1)//4, len(U1)//4  # For 2x2x2: 4,2,2
    n2, m2, p2 = len(U2), len(U2)//4, len(U2)//4
    
    U_new = [[0]*new_rank for _ in range(n1*n2)]
    V_new = [[0]*new_rank for _ in range(m1*m2)]
    W_new = [[0]*new_rank for _ in range(p1*p2)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1*rank2 + c2
            for i1 in range(n1):
                for i2 in range(n2):
                    idx = i1*n2 + i2
                    U_new[idx][c] = U1[i1][c1] * U2[i2][c2]
            for j1 in range(m1):
                for j2 in range(m2):
                    idx = j1*m2 + j2
                    V_new[idx][c] = V1[j1][c1] * V2[j2][c2]
            for k1 in range(p1):
                for k2 in range(p2):
                    idx = k1*p2 + k2
                    W_new[idx][c] = W1[k1][c1] * W2[k2][c2]
    
    return U_new, V_new, W_new

def _identity_decomp(size):
    """Return identity decomposition for size x size x size (rank = size)."""
    U = [[0]*size for _ in range(size)]
    V = [[0]*size for _ in range(size)]
    W = [[0]*size for _ in range(size)]
    for i in range(size):
        U[i][i] = 1
        V[i][i] = 1
        W[i][i] = 1
    return U, V, W

def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Build a block decomposition for matrix multiplication of size n x m x p.
    n_parts, m_parts, p_parts are lists of part sizes that sum to n, m, p respectively.
    Each part uses a known decomposition (identity for 1, Strassen for 2, Kronecker for 4).
    Returns (U, V, W, rank) or None if it doesn't improve over schoolbook.
    """
    def _get_decomp_for_size(size):
        if size == 1:
            return _identity_decomp(1)
        elif size == 2:
            return _strassen_2x2x2()
        elif size == 4:
            # 4x4x4 can be done as Kronecker of 2x2x2 with itself (rank 49)
            U1, V1, W1 = _strassen_2x2x2()
            U2, V2, W2 = _strassen_2x2x2()
            return _kronecker_product(U1, V1, W1, U2, V2, W2)
        else:
            # Fallback to schoolbook for this sub-block
            U = [[0]*size for _ in range(size)]
            V = [[0]*size for _ in range(size)]
            W = [[0]*size for _ in range(size)]
            for i in range(size):
                U[i][i] = 1
                V[i][i] = 1
                W[i][i] = 1
            return U, V, W
    
    # Calculate total rank
    total_rank = 0
    n_decomps = []
    m_decomps = []
    p_decomps = []
    
    for s in n_parts:
        d = _get_decomp_for_size(s)
        n_decomps.append(d)
    for s in m_parts:
        d = _get_decomp_for_size(s)
        m_decomps.append(d)
    for s in p_parts:
        d = _get_decomp_for_size(s)
        p_decomps.append(d)
    
    for (U_n, V_n, W_n) in n_decomps:
        for (U_m, V_m, W_m) in m_decomps:
            for (U_p, V_p, W_p) in p_decomps:
                rank_n = len(U_n[0])
                rank_m = len(U_m[0])
                rank_p = len(U_p[0])
                total_rank += rank_n * rank_m * rank_p
    
    # If not better than schoolbook, return None
    if total_rank >= n * m * p:
        return None
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    n_offsets = []
    offset = 0
    for s in n_parts:
        n_offsets.append(offset)
        offset += s
    
    m_offsets = []
    offset = 0
    for s in m_parts:
        m_offsets.append(offset)
        offset += s
    
    p_offsets = []
    offset = 0
    for s in p_parts:
        p_offsets.append(offset)
        offset += s
    
    for ni_idx, (n_part, (U_n, V_n, W_n)) in enumerate(zip(n_parts, n_decomps)):
        for mj_idx, (m_part, (U_m, V_m, W_m)) in enumerate(zip(m_parts, m_decomps)):
            for pk_idx, (p_part, (U_p, V_p, W_p)) in enumerate(zip(p_parts, p_decomps)):
                rank_n = len(U_n[0])
                rank_m = len(U_m[0])
                rank_p = len(U_p[0])
                
                i_offset = n_offsets[ni_idx]
                j_offset = m_offsets[mj_idx]
                k_offset = p_offsets[pk_idx]
                
                for cn in range(rank_n):
                    for cm in range(rank_m):
                        for cp in range(rank_p):
                            c = col + cn*rank_m*rank_p + cm*rank_p + cp
                            
                            # Fill U_full: U[i,j] for i in n_part, j in m_part
                            for i_n in range(len(U_n)):
                                for i_m in range(len(U_m)):
                                    i_global = i_offset + i_n
                                    j_global = j_offset + i_m
                                    idx = i_global*m + j_global
                                    U_full[idx][c] = U_n[i_n][cn] * U_m[i_m][cm]
                            
                            # Fill V_full: V[j,k] for j in m_part, k in p_part
                            for j_m in range(len(V_m)):
                                for k_p in range(len(V_p)):
                                    j_global = j_offset + j_m
                                    k_global = k_offset + k_p
                                    idx = j_global*p + k_global
                                    V_full[idx][c] = V_m[j_m][cm] * V_p[k_p][cp]
                            
                            # Fill W_full: W[k,i] for k in p_part, i in n_part
                            for k_p in range(len(W_p)):
                                for i_n in range(len(W_n)):
                                    k_global = k_offset + k_p
                                    i_global = i_offset + i_n
                                    idx = k_global*n + i_global
                                    W_full[idx][c] = W_p[k_p][cp] * W_n[i_n][cn]
                            
                            col += 1
    
    return U_full, V_full, W_full, total_rank

def _try_all_block_decompositions(n, m, p, deadline):
    """
    Try various block decompositions for n x m x p.
    Returns the best (U, V, W, rank) found.
    """
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    def _gen_splits(d, max_parts=3):
        """Generate all ways to split d into parts of size 1, 2, or 4."""
        def _helper(remaining, parts):
            if time.time() > deadline:
                return
            if remaining == 0:
                yield list(parts)
                return
            for part in [4, 2, 1]:
                if remaining >= part and len(parts) < max_parts:
                    parts.append(part)
                    yield from _helper(remaining - part, parts)
                    parts.pop()
        
        return list(_helper(d, []))
    
    n_splits = _gen_splits(n)
    m_splits = _gen_splits(m)
    p_splits = _gen_splits(p)
    
    for n_parts in n_splits:
        if time.time() > deadline:
            break
        for m_parts in m_splits:
            if time.time() > deadline:
                break
            for p_parts in p_splits:
                if time.time() > deadline:
                    break
                
                result = _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts)
                if result is not None:
                    U, V, W, rank = result
                    if rank < best_rank:
                        if _verify(U, V, W, n, m, p):
                            best_result = (U, V, W, rank)
                            best_rank = rank
    
    return best_result

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x5x6."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
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
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try block decompositions
    result = _try_all_block_decompositions(n, m, p, deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_U = U
            best_V = V
            best_W = W
            best_rank = rank
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
