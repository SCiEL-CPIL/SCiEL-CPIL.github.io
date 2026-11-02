"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

# --- Strassen 2x2x2 (rank 7) ---
STRASSEN_222 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

def build_j_split(n, m, p, j_split, algo_222):
    """Build U,V,W for j-split decomposition."""
    rank = 0
    for grp in j_split:
        sz = len(grp)
        if sz == 1:
            rank += n * p  # 2*6 = 12
        elif sz == 2:
            rank += (p // 2) * len(algo_222)  # 3*7 = 21
        else:
            raise ValueError("group size not supported")
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for grp in j_split:
        sz = len(grp)
        if sz == 1:
            j0 = grp[0]
            for i in range(n):
                for k in range(p):
                    U[i * m + j0][col] = 1.0
                    V[j0 * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        elif sz == 2:
            j0, j1 = grp
            for k_pair in range(p // 2):
                k0 = k_pair * 2
                k1 = k_pair + 1
                for (u_list, v_list, w_list) in algo_222:
                    for (li, lj, val) in u_list:
                        U[li * m + (j0 + lj)][col] += val
                    for (lj, lk, val) in v_list:
                        V[(j0 + lj) * p + (k0 + lk)][col] += val
                    for (li, lk, val) in w_list:
                        W[(k0 + lk) * n + li][col] += val
                    col += 1
    return U, V, W, col

def verify(n, m, p, U, V, W):
    """Check exact decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ua = Ua.reshape(n, m, -1)
    Va = Va.reshape(m, p, -1)
    Wa = Wa.reshape(p, n, -1)
    S = np.einsum('ijt,jkt,kit->ijk', Ua, Va, Wa)
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Generate all j-splits (partitions of {0,1,2,3,4} into groups of size 1 or 2)
    def gen_splits(indices):
        if not indices:
            yield []
            return
        for i in range(len(indices)):
            # Try single
            rest = indices[:i] + indices[i+1:]
            for sub in gen_splits(rest):
                yield [[indices[i]]] + sub
            # Try pair with next
            if i + 1 < len(indices):
                rest = indices[:i] + indices[i+2:]
                for sub in gen_splits(rest):
                    yield [[indices[i], indices[i+1]]] + sub
    
    # Deduplicate splits (order of groups doesn't matter for rank, but does for construction)
    seen = set()
    splits = []
    for sp in gen_splits(list(range(m))):
        key = tuple(sorted(tuple(sorted(g)) for g in sp))
        if key not in seen:
            seen.add(key)
            splits.append(sp)
    
    best = None
    best_rank = None
    
    for sp in splits:
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
        U, V, W, col = build_j_split(n, m, p, sp, STRASSEN_222)
        if verify(n, m, p, U, V, W):
            if best_rank is None or col < best_rank:
                best_rank = col
                best = (U, V, W)
    
    if best is None:
        # Fallback: naive
        U = [[0.0] * 60 for _ in range(10)]
        V = [[0.0] * 60 for _ in range(30)]
        W = [[0.0] * 60 for _ in range(12)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        best = (U, V, W)
        best_rank = 60
    
    U, V, W = best
    def to_clean(mat):
        out = []
        for row in mat:
            out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
        return out
    return {"U": to_clean(U), "V": to_clean(V), "W": to_clean(W)}
# EVOLVE-BLOCK-END
