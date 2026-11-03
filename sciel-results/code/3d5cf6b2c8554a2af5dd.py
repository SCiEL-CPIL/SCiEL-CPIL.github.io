import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 using Strassen's algorithm on 2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    r = 0
    # A_1 (cols 0,1) * B_1 (rows 0,1)
    # B_1 is 2x6, split into 3 2x2 blocks (cols 0-1, 2-3, 4-5)
    for b in range(3):
        for u, v, w in strassen_terms:
            # A_1 indices: i in 0,1; j in 0,1
            # B_1 indices: j in 0,1; k in 2b, 2b+1
            # C indices: k in 2b, 2b+1; i in 0,1
            
            for li in range(2):
                for lj in range(2):
                    # Local A index li, lj -> Global i=li, j=lj
                    u_idx = li * m + lj
                    U[u_idx][r] = u[li * 2 + lj]
                    
                    # Local B index lj, lk -> Global j=lj, k=2b+lk
                    for lk in range(2):
                        v_idx = lj * p + (2 * b + lk)
                        V[v_idx][r] = v[lj * 2 + lk]
                        
                    # Local C index li, lk -> Global k=2b+lk, i=li
                    for lk in range(2):
                        w_idx = (2 * b + lk) * n + li
                        W[w_idx][r] = w[li * 2 + lk]
            
            r += 1
            
    # A_2 (cols 2,3) * B_2 (rows 2,3)
    for b in range(3):
        for u, v, w in strassen_terms:
            # A_2 indices: i in 0,1; j in 2,3
            # B_2 indices: j in 2,3; k in 2b, 2b+1
            # C indices: k in 2b, 2b+1; i in 0,1
            
            for li in range(2):
                for lj in range(2):
                    # Local A index li, lj -> Global i=li, j=2+lj
                    u_idx = li * m + (2 + lj)
                    U[u_idx][r] = u[li * 2 + lj]
                    
                    # Local B index lj, lk -> Global j=2+lj, k=2b+lk
                    for lk in range(2):
                        v_idx = (2 + lj) * p + (2 * b + lk)
                        V[v_idx][r] = v[lj * 2 + lk]
                        
                    # Local C index li, lk -> Global k=2b+lk, i=li
                    for lk in range(2):
                        w_idx = (2 * b + lk) * n + li
                        W[w_idx][r] = w[li * 2 + lk]
            
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
