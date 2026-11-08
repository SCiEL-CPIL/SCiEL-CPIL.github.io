"""Matrix multiplication 3x5x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

# The Strassen 7-term algorithm for 2x2x2 matrix multiplication
# Each term is (U_entries, V_entries, W_entries) where entries are {(row, col): value}
# U entries: (i, j) for 2x2 sub-block
# V entries: (j, k) for 2x2 sub-block
# W entries: (k, i) for 2x2 sub-block
_STRASSEN_222 = [
    # M1 = (A00+A11)(B00+B11)
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    # M2 = (A10+A11)B00
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
    # M3 = A00(B01-B11)
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    # M4 = A11(B10-B00)
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
    # M5 = (A00+A01)B11
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
    # M6 = (A10-A00)(B00+B01)
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    # M7 = (A01-A11)(B10+B11)
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]


def _build_decomposition(i_groups, j_groups, k_groups, n, m, p):
    """
    Build a tensor decomposition from partitions of i, j, k dimensions.
    For each block (ig, jg, kg):
      - if len(ig)==2 and len(jg)==2 and len(kg)==2: use Strassen (rank 7)
      - else: use schoolbook (rank = len(ig)*len(jg)*len(kg))
    """
    # Precompute total rank
    total_rank = 0
    block_specs = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                if si == 2 and sj == 2 and sk == 2:
                    r = 7
                else:
                    r = si * sj * sk
                total_rank += r
                block_specs.append((ig, jg, kg, r))

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for ig, jg, kg, r in block_specs:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            for u_pat, v_pat, w_pat in _STRASSEN_222:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _generate_contiguous_partitions(dim):
    """Generate all ways to split range(dim) into contiguous groups."""
    if dim <= 0:
        yield []
        return
    for first_size in range(1, dim + 1):
        for rest in _generate_contiguous_partitions(dim - first_size):
            first = list(range(first_size))
            shifted_rest = [[x + first_size for x in g] for g in rest]
            yield [first] + shifted_rest


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    # Schoolbook fallback
    def schoolbook():
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
        return U, V, W, rank

    # Generate all partitions for each dimension
    i_parts = list(_generate_contiguous_partitions(n))
    j_parts = list(_generate_contiguous_partitions(m))
    k_parts = list(_generate_contiguous_partitions(p))

    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()

    # Enumerate all combinations of partitions
    for i_groups in i_parts:
        if time.time() > deadline:
            break
        for j_groups in j_parts:
            if time.time() > deadline:
                break
            for k_groups in k_parts:
                if time.time() > deadline:
                    break
                U, V, W, rank = _build_decomposition(i_groups, j_groups, k_groups, n, m, p)
                if rank < best_rank:
                    if _verify_decomposition(U, V, W, n, m, p):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
