"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matmul tensor."""
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j2 != j:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k2 != k or i2 != i:
                    expected = 0
                else:
                    expected = 1
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    if not _verify(U, V, W, 2, 2, 2, 7):
        return _schoolbook(2, 2, 2)
    return U, V, W

def _combine_decompositions(parts, n, m, p):
    """Combine a list of (U_part, V_part, W_part, offset_i, offset_j, offset_k, ni, nj, np)
    into a single decomposition for the full n x m x p problem."""
    total_rank = sum(len(U_part[0]) for U_part, _, _, _, _, _, _, _, _ in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * m * 0 + m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for U_part, V_part, W_part, off_i, off_j, off_k, ni, nj, np in parts:
        rank_part = len(U_part[0])
        for i in range(ni):
            for j in range(nj):
                a_full = (off_i + i) * m + (off_j + j)
                a_part = i * nj + j
                for r in range(rank_part):
                    U[a_full][col + r] = U_part[a_part][r]
        for j in range(nj):
            for k in range(np):
                b_full = (off_j + j) * p + (off_k + k)
                b_part = j * np + k
                for r in range(rank_part):
                    V[b_full][col + r] = V_part[b_part][r]
        for k in range(np):
            for i in range(ni):
                c_full = (off_k + k) * n + (off_i + i)
                c_part = k * ni + i
                for r in range(rank_part):
                    W[c_full][col + r] = W_part[c_part][r]
        col += rank_part
    
    return U, V, W

def _block_rank(ni, nj, np):
    """Return the rank for a sub-block of size ni x nj x np."""
    if ni == 2 and nj == 2 and np == 2:
        return 7
    return ni * nj * np

def _gen_splits(total, max_parts=4):
    """Generate all compositions of 'total' into at most max_parts positive integers."""
    results = []
    def backtrack(remaining, parts, start):
        if not parts:
            if remaining > 0:
                results.append([remaining])
            return
        if len(parts) >= max_parts:
            results.append(list(parts))
            return
        for v in range(1, remaining + 1):
            parts.append(v)
            backtrack(remaining - v, parts, start + 1)
            parts.pop()
    backtrack(total, [], 1)
    # Deduplicate
    seen = set()
    unique = []
    for s in results:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

def _build_from_splits(n, m, p, i_sizes, j_sizes, k_sizes):
    """Build decomposition given specific split sizes for each dimension."""
    parts = []
    i_off = 0
    j_off = 0
    k_off = 0
    for ii, ni in enumerate(i_sizes):
        for jj, nj in enumerate(j_sizes):
            for kk, np in enumerate(k_sizes):
                if ni == 2 and nj == 2 and np == 2:
                    U_part, V_part, W_part = _strassen_2x2x2()
                else:
                    U_part, V_part, W_part = _schoolbook(ni, nj, np)
                parts.append((U_part, V_part, W_part, i_off, j_off, k_off, ni, nj, np))
                k_off += np
            k_off = 0
            j_off += nj
        j_off = 0
        i_off += ni
    
    U, V, W = _combine_decompositions(parts, n, m, p)
    return U, V, W

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Generate all possible splits
    i_splits = _gen_splits(n, max_parts=n)
    j_splits = _gen_splits(m, max_parts=m)
    k_splits = _gen_splits(p, max_parts=p)
    
    # Score each combination by total rank (without building)
    scored = []
    for isz in i_splits:
        for jsz in j_splits:
            for ksz in k_splits:
                total_r = 0
                for ni in isz:
                    for nj in jsz:
                        for np in ksz:
                            total_r += _block_rank(ni, nj, np)
                scored.append((total_r, isz, jsz, ksz))
    
    # Sort by rank ascending
    scored.sort(key=lambda x: x[0])
    
    # Try top candidates
    for total_r, isz, jsz, ksz in scored:
        if total_r >= best_rank:
            break
        if time.time() > deadline - 10:
            break
        U2, V2, W2 = _build_from_splits(n, m, p, isz, jsz, ksz)
        rank2 = len(U2[0]) if U2 else 0
        if rank2 < best_rank:
            if _verify(U2, V2, W2, n, m, p, rank2):
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    return best_result
# EVOLVE-BLOCK-END
