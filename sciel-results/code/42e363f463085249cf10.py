"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # Safety margin before 150s kill

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # P1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # P3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # P4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # P7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """Build schoolbook rank-4 decomposition for 2x2 x 2x1."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1
            V[j * p + 0, r] = 1
            W[0 * n + i, r] = 1
    
    return U, V, W

def _build_decomposition(config):
    """
    Build decomposition based on configuration.
    config: list of tuples (type, start_col, num_cols)
      type: 'strassen' for 2-col blocks, 'school' for 1-col blocks
    Returns (U, V, W) or None if invalid.
    """
    n, m, p = 2, 2, 15
    
    # Calculate total rank
    rank = 0
    for typ, start, num in config:
        if typ == 'strassen':
            rank += 7 * (num // 2)
        elif typ == 'school':
            rank += 4 * num
    
    if rank == 0:
        return None
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sch, V_sch, W_sch = _build_schoolbook_2x2x1()
    
    col_offset = 0
    for typ, start, num in config:
        if typ == 'strassen':
            for block in range(num // 2):
                k0 = start + 2 * block
                k1 = k0 + 1
                if k1 >= p:
                    return None
                for r in range(7):
                    c = col_offset + r
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, c] = U_str[i * m + j, r]
                    for j in range(m):
                        V[j * p + k0, c] = V_str[j * p + 0, r]
                        V[j * p + k1, c] = V_str[j * p + 1, r]
                    for k in range(2):
                        kk = k0 + k
                        for i in range(n):
                            W[kk * n + i, c] = W_str[k * n + i, r]
                col_offset += 7
        elif typ == 'school':
            for block in range(num):
                k = start + block
                if k >= p:
                    return None
                for r in range(4):
                    c = col_offset + r
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, c] = U_sch[i * m + j, r]
                    for j in range(m):
                        V[j * p + k, c] = V_sch[j, r]
                    for i in range(n):
                        W[k * n + i, c] = W_sch[i, r]
                col_offset += 4
    
    if col_offset != rank:
        return None
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=2, p=15):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _find_best_config():
    """
    Search for the best block configuration.
    We need to cover 15 columns with strassen (2 cols, rank 7) and school (1 col, rank 4) blocks.
    """
    p = 15
    
    # Generate all possible configurations
    # We can have any number of strassen blocks (covering 2 cols each) and school blocks (1 col each)
    # Let s = number of strassen blocks, t = number of school blocks
    # 2s + t = 15, s >= 0, t >= 0
    # s can be 0 to 7, t = 15 - 2s
    
    best_rank = 53  # current best
    best_config = None
    
    for s in range(0, 8):  # 0 to 7 strassen blocks
        t = 15 - 2 * s
        if t < 0:
            continue
        
        # All strassen blocks must be contiguous pairs, schools fill the rest
        # The arrangement matters for the tensor structure
        # For simplicity, try: all strassen first, then schools
        # Or: interleave them
        
        rank = 7 * s + 4 * t
        
        if rank >= best_rank:
            continue
        
        # Try different arrangements
        # Arrangement 1: strassen blocks first, then schools
        config = [('strassen', 0, 2 * s)] + [('school', 2 * s, t)]
        result = _build_decomposition(config)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W):
                best_rank = rank
                best_config = config
    
    return best_rank, best_config

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Try to find a better configuration
    best_rank, best_config = _find_best_config()
    
    if best_config is not None and best_rank < 53:
        result = _build_decomposition(best_config)
        if result is not None:
            U, V, W = result
            # Convert to lists for the interface
            return {
                "U": [list(map(float, row)) for row in U],
                "V": [list(map(float, row)) for row in V],
                "W": [list(map(float, row)) for row in W]
            }
    
    # Fallback to the known rank-53 solution
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sch, V_sch, W_sch = _build_schoolbook_2x2x1()
    
    col_offset = 0
    # 7 Strassen blocks for columns 0-13
    for block in range(7):
        k0 = 2 * block
        k1 = k0 + 1
        for r in range(7):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = U_str[i * m + j, r]
            for j in range(m):
                V[j * p + k0, c] = V_str[j * p + 0, r]
                V[j * p + k1, c] = V_str[j * p + 1, r]
            for k in range(2):
                kk = k0 + k
                for i in range(n):
                    W[kk * n + i, c] = W_str[k * n + i, r]
        col_offset += 7
    
    # Schoolbook for column 14
    k = 14
    for r in range(4):
        c = col_offset + r
        for i in range(n):
            for j in range(m):
                U[i * m + j, c] = U_sch[i * m + j, r]
        for j in range(m):
            V[j * p + k, c] = V_sch[j, r]
        for i in range(n):
            W[k * n + i, c] = W_sch[i, r]
        col_offset += 4
    
    return {
        "U": [list(map(float, row)) for row in U],
        "V": [list(map(float, row)) for row in V],
        "W": [list(map(float, row)) for row in W]
    }

# EVOLVE-BLOCK-END
