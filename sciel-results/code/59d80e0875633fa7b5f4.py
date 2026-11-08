"""Combined strategy: Strassen-based decomposition with rank-reduction polish pass."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    N_U = n * m  # 6
    N_V = m * p  # 24
    N_W = p * n  # 16

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
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

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(N_U)]
    V = [[0] * rank for _ in range(N_V)]
    W = [[0] * rank for _ in range(N_W)]

    col = 0
    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if not _validate(candidate):
        return _schoolbook()

    # Polish pass: try to reduce rank by finding dependent terms
    # For each pair of columns, check if one can be expressed as a scalar multiple
    # of the other in all three matrices (exact rational check)
    def _try_reduce_rank(U, V, W):
        """Try to eliminate dependent rank-1 terms."""
        rank = len(U[0])
        used = [False] * rank
        
        # Check for scalar multiples
        for c1 in range(rank):
            if used[c1]:
                continue
            for c2 in range(c1 + 1, rank):
                if used[c2]:
                    continue
                # Check if column c2 is a scalar multiple of column c1
                # in all three matrices
                ratio = None
                is_multiple = True
                for idx in range(N_U):
                    if U[idx][c1] != 0:
                        if ratio is None:
                            ratio = Fraction(U[idx][c2], U[idx][c1])
                        elif U[idx][c2] != ratio * U[idx][c1]:
                            is_multiple = False
                            break
                    elif U[idx][c2] != 0:
                        is_multiple = False
                        break
                if not is_multiple:
                    continue
                for idx in range(N_V):
                    if V[idx][c1] != 0:
                        if ratio is None:
                            ratio = Fraction(V[idx][c2], V[idx][c1])
                        elif V[idx][c2] != ratio * V[idx][c1]:
                            is_multiple = False
                            break
                    elif V[idx][c2] != 0:
                        is_multiple = False
                        break
                if not is_multiple:
                    continue
                for idx in range(N_W):
                    if W[idx][c1] != 0:
                        if ratio is None:
                            ratio = Fraction(W[idx][c2], W[idx][c1])
                        elif W[idx][c2] != ratio * W[idx][c1]:
                            is_multiple = False
                            break
                    elif W[idx][c2] != 0:
                        is_multiple = False
                        break
                if is_multiple and ratio is not None:
                    # c2 = ratio * c1, so we can eliminate c2
                    used[c2] = True
                    break
        
        # Build reduced decomposition
        keep = [c for c in range(rank) if not used[c]]
        new_rank = len(keep)
        if new_rank >= rank:
            return None
        
        new_U = [[U[i][c] for c in keep] for i in range(N_U)]
        new_V = [[V[i][c] for c in keep] for i in range(N_V)]
        new_W = [[W[i][c] for c in keep] for i in range(N_W)]
        
        candidate = {"U": new_U, "V": new_V, "W": new_W}
        if _validate(candidate):
            return candidate
        return None

    reduced = _try_reduce_rank(U, V, W)
    if reduced is not None:
        return reduced

    return candidate

# EVOLVE-BLOCK-END
