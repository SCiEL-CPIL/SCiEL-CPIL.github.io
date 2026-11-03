"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1
    
    return U, V, W


def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_block_decomposition(i_blocks, j_blocks, k_blocks, n, m, p):
    """
    Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
    Returns (U, V, W, rank) or None if verification fails.
    """
    su, sv, sw = _strassen_222()
    
    # Calculate total rank
    total_rank = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_set) * len(j_set) * len(k_set)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    # Use Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    if col != total_rank:
        return None
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return None
    
    return U, V, W, total_rank


def _try_partition(i_blocks, j_blocks, k_blocks, n, m, p):
    """Try a specific partition and return (U, V, W, rank) if valid."""
    try:
        result = _build_block_decomposition(i_blocks, j_blocks, k_blocks, n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Try various partitions, ordered by expected rank
    partitions = [
        # Partition 1: i={0},{1,2}; j={0,1},{2,3},{4}; k={0,1},{2,3},{4}
        # 4 Strassen blocks + schoolbook = 28 + 43 = 71
        ([(0,), (1, 2)], [(0, 1), (2, 3), (4,)], [(0, 1), (2, 3), (4,)]),
        
        # Partition 2: i={0},{1},{2}; j={0,1},{2,3},{4}; k={0,1},{2,3},{4}
        # No Strassen (i blocks are size 1), all schoolbook = 75
        # (This is just schoolbook, skip)
        
        # Partition 3: i={0},{1,2}; j={0},{1,2},{3,4}; k={0},{1,2},{3,4}
        # 1 Strassen block + schoolbook = 7 + 67 = 74 (worse)
        
        # Try more creative partitions
        # Partition 4: i={0},{1,2}; j={0,1},{2,3},{4}; k={0,1},{2,3},{4}
        # Same as partition 1
        
        # Partition 5: i={0,1},{2}; j={0,1},{2,3},{4}; k={0,1},{2,3},{4}
        # 4 Strassen blocks (i={0,1} with j={0,1} or {2,3}, k={0,1} or {2,3})
        # = 4*7 + schoolbook
        # Schoolbook: i={0,1},j={4}: 2*1*5=10; i={2}: 1*5*5=25; i={0,1},j={0,1},k={4}: 2*2*1=4; i={0,1},j={2,3},k={4}: 2*2*1=4
        # Total: 28 + 10+25+4+4 = 71
        ([(0, 1), (2,)], [(0, 1), (2, 3), (4,)], [(0, 1), (2, 3), (4,)]),
        
        # Partition 6: i={0},{1,2}; j={0},{1,2},{3,4}; k={0,1},{2,3},{4}
        # Strassen: i={1,2}, j={1,2}, k={0,1} and i={1,2}, j={1,2}, k={2,3} = 2 blocks
        # = 2*7 + schoolbook
        # Schoolbook: i={0}: 1*5*5=25; i={1,2},j={0}: 2*1*5=10; i={1,2},j={3,4}: 2*2*5=20; i={1,2},j={1,2},k={4}: 2*2*1=4
        # Total: 14 + 25+10+20+4 = 73
        ([(0,), (1, 2)], [(0,), (1, 2), (3, 4)], [(0, 1), (2, 3), (4,)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for i_blocks, j_blocks, k_blocks in partitions:
        if time.time() > deadline:
            break
        
        result = _try_partition(i_blocks, j_blocks, k_blocks, n, m, p)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
