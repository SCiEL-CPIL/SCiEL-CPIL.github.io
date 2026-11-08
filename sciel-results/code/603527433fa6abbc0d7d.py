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
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Lathrop-Strassen 2x2x4 decomposition
    # U: 2x2 (4 entries), V: 2x4 (8 entries), W: 4x2 (8 entries)
    LS_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 0]
    ]
    LS_V = [
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 0]
    ]
    LS_W = [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ]

    # Verify Lathrop-Strassen decomposition internally
    def _verify_ls():
        for i in range(2):
            for a in range(2):
                for k in range(4):
                    acc = 0
                    for r in range(7):
                        u_val = LS_U[r][i*2 + a]
                        v_val = LS_V[r][a*4 + k]
                        w_val = LS_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_ls():
        return _schoolbook()

    # Build candidate
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x4) for k in 0..3
    for r in range(7):
        for li in range(2):
            for la in range(2):
                u_val = LS_U[r][li*2 + la]
                if u_val == 0: continue
                u_idx = li * m + la
                U[u_idx][col] = u_val
        
        for la in range(2):
            for lk in range(4):
                v_val = LS_V[r][la*4 + lk]
                if v_val == 0: continue
                v_idx = la * p + lk
                V[v_idx][col] = v_val
        
        for lk in range(4):
            for li in range(2):
                w_val = LS_W[r][lk*2 + li]
                if w_val == 0: continue
                w_idx = lk * n + li
                W[w_idx][col] = w_val
        
        col += 1
        if not _elapsed_ok():
            return _schoolbook()

    # Part 2: A1 (2x2) * B2 (2x4) for k in 4..7
    for r in range(7):
        for li in range(2):
            for la in range(2):
                u_val = LS_U[r][li*2 + la]
                if u_val == 0: continue
                u_idx = li * m + la
                U[u_idx][col] = u_val
        
        for la in range(2):
            for lk in range(4):
                v_val = LS_V[r][la*4 + lk]
                if v_val == 0: continue
                v_idx = la * p + (4 + lk)
                V[v_idx][col] = v_val
        
        for lk in range(4):
            for li in range(2):
                w_val = LS_W[r][lk*2 + li]
                if w_val == 0: continue
                w_idx = (4 + lk) * n + li
                W[w_idx][col] = w_val
        
        col += 1
        if not _elapsed_ok():
            return _schoolbook()

    # Part 3: A2 (2x1) * B3 (1x8)
    # A2 uses col 2 of A (global a=2).
    # B3 uses row 2 of B (global a=2).
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
