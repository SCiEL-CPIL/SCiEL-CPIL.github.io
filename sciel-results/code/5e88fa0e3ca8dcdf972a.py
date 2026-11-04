"""Combined strategy: Use Strassen decomposition for 2x2 blocks and schoolbook for the rest."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # Vectors for U (i,a), V (a,k), W (k,i)
    # Indices: U: 00,01,10,11 -> 0,1,2,3
    #          V: 00,01,10,11 -> 0,1,2,3
    #          W: 00,01,10,11 -> 0,1,2,3 (k*n+i, n=2)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Verify Strassen decomposition internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0,1. Cols 0..7.
    # Split B1 cols into 4 blocks of 2: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U indices: i in 0,1; a in 0,1 (global a = local a)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0: continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            # V indices: a in 0,1 (global a = local a); k in k0, k1 (global k = k0 + lk)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0: continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            # W indices: k in k0, k1 (global k = k0 + lk); i in 0,1 (global i = li)
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0: continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 uses col 2 of A (global a=2).
    # B2 uses row 2 of B (global a=2).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
