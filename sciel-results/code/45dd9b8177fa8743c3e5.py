"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

_t0 = time.time()
_DEADLINE = _t0 + 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Strategy: Use 5 Strassen blocks (2x2x2) for columns (0,1),(2,3),(4,5),(6,7),(8,9)
    # Then a 2x2x4 block for columns (10,11,12,13) with rank 14
    # Then a 2x2x1 block for column 14 with rank 4
    # Total rank = 5*7 + 14 + 4 = 49

    # First, let's construct a rank-14 decomposition for 2x2x4
    # 2x2x4 can be seen as two 2x2x2 blocks, which would give rank 14 (7+7)
    # But we need to verify if rank 14 is achievable or if we can do better
    
    # For now, use two Strassen blocks for the 2x2x4 part
    # This gives: 5*7 + 7 + 7 + 4 = 49
    
    rank = 5 * 7 + 7 + 7 + 4  # = 35 + 14 + 4 = 53... wait that's the same
    
    # Let me reconsider. The current approach is 7 pairs + 1 single = 7*7 + 4 = 53
    # Alternative: 6 pairs + 1 pair + 1 single = still 7 pairs total
    # Or: 5 pairs + 1 four-column block + 1 single
    # If the four-column block has rank 14 (two Strassen), total = 35 + 14 + 4 = 53 (same)
    
    # The key question: can 2x2x4 have rank less than 14?
    # The trivial rank is 8, Strassen gives 14. Is there something in between?
    # Actually, I recall that rank(2x2x4) = 14 is not proven minimal.
    # Let's just stick with the current 53 and see if we can find any improvement.
    
    # For now, return the same 53-rank decomposition
    rank = 53

    # Allocate flat factor matrices as 2-D lists of ints
    U = [[0] * rank for _ in range(n * m)]   # 4 x 53
    V = [[0] * rank for _ in range(m * p)]   # 30 x 53
    W = [[0] * rank for _ in range(p * n)]   # 30 x 53

    # U row indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V row indices: j*p + k  (j = 0 or 1, k = 0..14)
    # W row indices: k*n + i  (k = 0..14, i = 0 or 1)

    # Strassen 2x2x2 factor pattern
    u_entries = [
        [(0, 1), (3, 1)],   # P1
        [(2, 1), (3, 1)],   # P2
        [(0, 1)],           # P3
        [(3, 1)],           # P4
        [(0, 1), (1, 1)],   # P5
        [(2, 1), (0, -1)],  # P6
        [(1, 1), (3, -1)],  # P7
    ]
    v_entries = [
        [(('B0k0', 1)), (('B1k1', 1))],  # P1
        [(('B0k0', 1))],                  # P2
        [(('B0k1', 1)), (('B1k1', -1))],  # P3
        [(('B1k0', 1)), (('B0k0', -1))],  # P4
        [(('B1k1', 1))],                  # P5
        [(('B0k0', 1)), (('B0k1', 1))],   # P6
        [(('B1k0', 1)), (('B1k1', 1))],   # P7
    ]
    w_entries = [
        [(('C0k0', 1)), (('C1k1', 1))],  # P1
        [(('C1k0', 1)), (('C1k1', -1))], # P2
        [(('C0k1', 1)), (('C1k1', 1))],  # P3
        [(('C0k0', 1)), (('C1k0', 1))],  # P4
        [(('C0k0', -1)), (('C0k1', 1))], # P5
        [(('C1k1', 1))],                 # P6
        [(('C0k0', 1))],                 # P7
    ]

    def v_row(code, k0, k1):
        j = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return j * p + k

    def w_row(code, k0, k1):
        i = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return k * n + i

    # Place 7 Strassen blocks
    for blk in range(7):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        base = 7 * blk
        for t in range(7):
            r = base + t
            for (u_row_i, c) in u_entries[t]:
                U[u_row_i][r] += c
            for (code, c) in v_entries[t]:
                V[v_row(code, k0, k1)][r] += c
            for (code, c) in w_entries[t]:
                W[w_row(code, k0, k1)][r] += c

    # Place the residual 2x2x1 block for column 14 (rank 4, trivial)
    b14 = 49
    U[0][b14 + 0] = 1
    V[0 * p + 14][b14 + 0] = 1
    W[14 * 2 + 0][b14 + 0] = 1
    U[1][b14 + 1] = 1
    V[1 * p + 14][b14 + 1] = 1
    W[14 * 2 + 0][b14 + 1] = 1
    U[2][b14 + 2] = 1
    V[0 * p + 14][b14 + 2] = 1
    W[14 * 2 + 1][b14 + 2] = 1
    U[3][b14 + 3] = 1
    V[1 * p + 14][b14 + 3] = 1
    W[14 * 2 + 1][b14 + 3] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
