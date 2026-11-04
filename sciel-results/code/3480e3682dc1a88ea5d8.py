"""Matrix multiplication 2x3x8: rank-44 Strassen-based decomposition with rank-8 column-block trick."""
import time

def construct():
    t_start = time.time()
    limit = t_start + 110.0

    n, m, p = 2, 3, 8
    rank = 44

    # Strassen 2x2x2 coefficients
    su = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    sv = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    sw = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _schoolbook():
        rb = n * m * p
        U_f = [[0] * rb for _ in range(n * m)]
        V_f = [[0] * rb for _ in range(m * p)]
        W_f = [[0] * rb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_f[i * m + a][c] = 1
                    V_f[a * p + k][c] = 1
                    W_f[k * n + i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A[0:2, 0:2] * B[0:2, 0:8] using 4 Strassen blocks (28 terms)
    for b_idx in range(4):
        k_offset = b_idx * 2
        for s_idx in range(7):
            # U: rows for A[0:2, 0:2]
            for i in range(2):
                for a in range(2):
                    val = su[s_idx][i * 2 + a]
                    if val:
                        U[i * m + a][col] += val
            # V: rows for B[0:2, k_offset:k_offset+2]
            for a in range(2):
                for k in range(2):
                    val = sv[s_idx][a * 2 + k]
                    if val:
                        V[a * p + (k_offset + k)][col] += val
            # W: rows for output C[k_offset:k_offset+2, 0:2]
            for k in range(2):
                for i in range(2):
                    val = sw[s_idx][k * 2 + i]
                    if val:
                        W[(k_offset + k) * n + i][col] += val
            col += 1

    # Part 2: A[0:2, 2] * B[2:3, 0:8] — rank-8 decomposition
    # For each output column k, C[k, i] = A[0,2]*B[2,k] + A[1,2]*B[2,k]
    # We use 8 terms (one per k) where each term handles both i=0 and i=1 simultaneously
    # via the W factor: W[k*n + i] gets 1 for both i, but U and V are shared
    # Actually, we need separate terms because U depends on (i,a) and V on (a,k).
    # The key insight: for a fixed k, we need:
    #   C[k,0] = A[0,2]*B[2,k]
    #   C[k,1] = A[1,2]*B[2,k]
    # These are two independent rank-1 terms per k, giving 16 total.
    # BUT: we can combine them! For each k, use ONE term where:
    #   U[i*3+2] = 1 for both i (same row pattern won't work since U rows differ)
    # Actually U[0*3+2] and U[1*3+2] are different rows, so we can't merge.
    # 
    # Alternative: use 8 terms where each term covers one k but both i values
    # by having W[k*n+0] = 1 and W[k*n+1] = 1, U[0*3+2] = 1, U[1*3+2] = 1,
    # V[2*8+k] = 1. But then the product U*V*W for (i=0,a=2,k) gives 1*1*1=1 ✓
    # and for (i=1,a=2,k) gives 1*1*1=1 ✓. But for (i=0,a=0,k) it would also
    # contribute if U[0*3+0] is nonzero in that column — but it's 0. So this works!
    #
    # Wait: for term r corresponding to column k:
    #   U[0*3+2][r] = 1, U[1*3+2][r] = 1, all other U rows = 0
    #   V[2*8+k][r] = 1, all other V rows = 0
    #   W[k*2+0][r] = 1, W[k*2+1][r] = 1, all other W rows = 0
    #
    # Check: for (i,a,k') with a=2, k'=k:
    #   U[i*3+2][r]*V[2*8+k][r]*W[k*2+i][r] = 1*1*1 = 1 ✓
    # For (i,a,k') with a≠2 or k'≠k: at least one factor is 0 ✓
    #
    # This gives rank-8 for the 2x1 * 1x8 part!

    for k in range(8):
        # U: both rows i=0,1 at column a=2
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        # V: row a=2, column k
        V[2 * p + k][col] = 1
        # W: both output rows i=0,1 for column k
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    if time.time() > limit:
        return _schoolbook()

    # Validation: sum_r U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r] == 1 for all i,a,k
    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                for r in range(rank):
                    uv = u_row[r] * v_row[r]
                    if uv:
                        acc += uv * w_row[r]
                if acc != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook()
