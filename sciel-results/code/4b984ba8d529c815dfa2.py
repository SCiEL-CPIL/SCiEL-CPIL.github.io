"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is (U_indices, V_indices, W_indices) where indices are (row, col) pairs.
    U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries).
    """
    terms = []
    # Term 0: M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # Term 1: M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    # Term 2: M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # Term 3: M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # Term 4: M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # Term 5: M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # Term 6: M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    return terms

def _naive_2x2x1():
    """Return the 2 terms of naive 2x2x1 matrix multiplication.
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    """
    terms = []
    # Term 0: A[i,0]*B[0,0]
    terms.append({
        'U': {(0,0): 1, (1,0): 1},
        'V': {(0,0): 1},
        'W': {(0,0): 1}
    })
    # Term 1: A[i,1]*B[1,0]
    terms.append({
        'U': {(0,1): 1, (1,1): 1},
        'V': {(1,0): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_block_decomposition(n, m, p):
    """Build decomposition by splitting into 2x2 blocks.
    
    For each block of size 2 in each dimension:
    - 2x2x2 blocks use Strassen (rank 7)
    - 2x2x1 blocks use naive (rank 2)
    - 2x1x2 blocks use naive (rank 2)
    - 2x1x1 blocks use naive (rank 1)
    - 1x2x2 blocks use naive (rank 2)
    - 1x2x1 blocks use naive (rank 1)
    - 1x1x2 blocks use naive (rank 1)
    - 1x1x1 blocks use naive (rank 1)
    """
    # Determine block structure
    # n=4: split into two blocks of 2
    # m=4: split into two blocks of 2
    # p=5: split into blocks of 2,2,1
    
    n_blocks = [2, 2]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 1]
    
    # Compute offsets
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U = {}  # (i, j) -> list of (coeff, rank_idx)
    V = {}  # (j, k) -> list of (coeff, rank_idx)
    W = {}  # (k, i) -> list of (coeff, rank_idx)
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                no = n_offsets[ni]
                mo = m_offsets[mi]
                po = p_offsets[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen
                    for term in _strassen_2x2():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                elif ns == 2 and ms == 2 and ps == 1:
                    # Use naive 2x2x1
                    for term in _naive_2x2x1():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                else:
                    # Fallback to schoolbook for this block
                    for li in range(ns):
                        for lj in range(ms):
                            for lk in range(ps):
                                i, j, k = no + li, mo + lj, po + lk
                                U.setdefault((i, j), []).append((1, rank))
                                V.setdefault((j, k), []).append((1, rank))
                                W.setdefault((k, i), []).append((1, rank))
                                rank += 1
    
    # Convert to matrix form
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, r in entries:
            U_mat[i * m + j][r] += coeff
    for (j, k), entries in V.items():
        for coeff, r in entries:
            V_mat[j * p + k][r] += coeff
    for (k, i), entries in W.items():
        for coeff, r in entries:
            W_mat[k * n + i][r] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition approach
    try:
        result = _build_block_decomposition(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
