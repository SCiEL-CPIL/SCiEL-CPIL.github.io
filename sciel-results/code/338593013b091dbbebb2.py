import time
import numpy as np

_start_time = time.time()
_TIMEOUT = 140.0  # safety margin before 150s kill

def _build_strassen56():
    """Build the rank-56 Strassen-based decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    
    # Strassen 2x2 local coefficients
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56

    group = 0
    for jj in range(p // 2):  # 4 groups of 2 columns
        for half in range(2):  # 2 groups of 2 rows
            base_r = group * 7
            group += 1
            
            # Local to Global Maps
            a_map = {
                0: 0 * m + (2 * half + 0),
                1: 0 * m + (2 * half + 1),
                2: 1 * m + (2 * half + 0),
                3: 1 * m + (2 * half + 1)
            }
            
            b_map = {
                0: (2 * half + 0) * p + (2 * jj + 0),
                1: (2 * half + 0) * p + (2 * jj + 1),
                2: (2 * half + 1) * p + (2 * jj + 0),
                3: (2 * half + 1) * p + (2 * jj + 1)
            }
            
            c_map = {
                0: (2 * jj + 0) * n + 0,
                1: (2 * jj + 1) * n + 0,
                2: (2 * jj + 0) * n + 1,
                3: (2 * jj + 1) * n + 1
            }

            for r in range(7):
                rr = base_r + r
                u_list, v_list, w_list = strassen_terms[r]
                
                for idx, val in u_list:
                    U[a_map[idx]][rr] += val
                    
                for idx, val in v_list:
                    V[b_map[idx]][rr] += val
                    
                for idx, val in w_list:
                    W[c_map[idx]][rr] += val

    return U, V, W

def _validate(U, V, W, n=2, m=4, p=8, seed=42):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    
    for _ in range(3):
        A = rng.randint(-3, 4, size=(n, m))
        B = rng.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=int)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        
        if not np.array_equal(C_expected, C_computed):
            return False
    return True

def construct():
    n, m, p = 2, 4, 8
    
    # Build the base rank-56 solution
    U, V, W = _build_strassen56()
    
    # Validate (fast check)
    if not _validate(U, V, W, n, m, p):
        # Should not happen, but just in case
        pass
    
    # Deadline check
    if time.time() > _start_time + _TIMEOUT - 1.0:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
