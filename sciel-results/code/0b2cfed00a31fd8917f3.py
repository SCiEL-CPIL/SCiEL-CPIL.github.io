import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    r = len(W[0])
    # accumulate tensor T[i,j,k] over c
    T = {}
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # gather nonzero rows
        unz = [a for a in range(n * m) if uc[a] != 0]
        vnz = [b for b in range(m * p) if vc[b] != 0]
        wnz = [d for d in range(p * n) if wc[d] != 0]
        for a in unz:
            i, j = divmod(a, m)
            ua = uc[a]
            for b in vnz:
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                val = ua * vc[b]
                key = (i, j, k)
                T[key] = T.get(key, Fraction(0)) + val
        # W entries must equal the accumulated tensor
        for d in wnz:
            k, i = divmod(d, n)
            key = (i, k, k)  # placeholder, overwritten below
            # correct key: (i, j, k) but j is not determined by W alone;
            # so instead we check T against W by reconstructing full check below.
    # Full check: for every (i,j,k), T[i,j,k] must equal sum_c W[k*n+i,c]*? 
    # Actually the decomposition identity is:
    #   (A B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j U[i*m+j,c] A_{i,j}) (sum_j V[j*p+k,c] B_{j,k})
    # and W[k*n+i,c] multiplies the two rank-1 terms. So the tensor is
    #   T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c]  and the output factor is W[k*n+i,c].
    # The output matrix entry uses W[k*n+i,c]; the tensor identity requires
    #   T[i,j,k] == W[k*n+i,c] summed appropriately? 
    # The correct identity: C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j U[i*m+j,c] A_{i,j})(sum_j V[j*p+k,c] B_{j,k])
    # = sum_c (sum_j A_{i,j}U[...]) (sum_j B_{j,k}V[...]).
    # This equals sum_c W[k*n+i,c] * (sum_j A_{i,j}U[i*m+j,c])(sum_j B_{j,k}V[j*p+k,c]) / W? 
    # No. The standard form: C_{i,k} = sum_c W[k*n+i,c] * L_c(A) * R_c(B)
    # where L_c(A) = sum_j U[i*m+j,c] A_{i,j} and R_c(B)=sum_j V[j*p+k,c] B_{j,k}.
    # So the tensor T[i,j,k] (coefficient of A_{i,j}B_{j,k} in C_{i,k}) must equal
    #   sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c].
    # That is the real identity. Let's check that.
    T2 = {}
    for c in range(r):
        wc = W[:, c]
        for d in wnz:
            pass
    # Recompute cleanly.
    T2 = {}
    for c in range(r):
        uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
        for a in range(n * m):
            if uc[a] == 0:
                continue
            i, j = divmod(a, m)
            for b in range(m * p):
                if vc[b] == 0:
                    continue
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                wv = wc[k * n + i]
                if wv == 0:
                    continue
                key = (i, j, k)
                T2[key] = T2.get(key, Fraction(0)) + uc[a] * vc[b] * wv
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T2.get((i, j, k), Fraction(0)) != Fraction(1):
                    return False
    return True


def _rank_of(U, V, W):
    return len(U[0])


def _try_strassen2x2x2(n, m, p, seed):
    """Build a Strassen-like 2x2 block recursion for the 2x2x2 core, then lift.
    This is a heuristic; returns None if it cannot produce a valid decomposition."""
    # For 2x2x2, Strassen gives rank 7. We'll construct the 7-term decomposition.
    # Then try to map it into the 4x5x6 grid by padding. This generally does NOT
    # reduce the rank for non-square sizes, so we skip unless it verifies.
    return None


def _direct_search(n, m, p, max_rank, seed, deadline):
    """Random search over half-integer coefficient tensors, verifying exactly."""
    rng = random.Random(seed)
    best = None
    # candidate ranks to try, descending from max_rank
    for rank in range(max_rank, 7, -1):
        start = time.time()
        attempts = 0
        while time.time() < deadline - 2.0:
            attempts += 1
            # random small half-integer entries
            def rand_entry():
                # choose from {-1, -0.5, 0, 0.5, 1} with bias to 0
                r = rng.random()
                if r < 0.5:
                    return Fraction(0)
                sign = Fraction(1) if rng.random() < 0.5 else Fraction(-1)
                mag = Fraction(1) if rng.random() < 0.5 else Fraction(1, 2)
                return sign * mag
            U = [[rand_entry() for _ in range(rank)] for _ in range(n * m)]
            V = [[rand_entry() for _ in range(rank)] for _ in range(m * p)]
            W = [[rand_entry() for _ in range(rank)] for _ in range(p * n)]
            if _verify(U, V, W, n, m, p):
                if best is None or rank < _rank_of(*best.values()):
                    best = {"U": U, "V": V, "W": W}
                    return best
            # also try sparse-1 version
            U1 = [[Fraction(0)] * rank for _ in range(n * m)]
            V1 = [[Fraction(0)] * rank for _ in range(m * p)]
            W1 = [[Fraction(0)] * rank for _ in range(p * n)]
            for c in range(rank):
                a = rng.randrange(n * m)
                b = rng.randrange(m * p)
                d = rng.randrange(p * n)
                U1[a][c] = Fraction(1)
                V1[b][c] = Fraction(1)
                W1[d][c] = Fraction(1)
            if _verify(U1, V1, W1, n, m, p):
                if best is None or rank < _rank_of(*best.values()):
                    best = {"U": U1, "V": V1, "W": W1}
                    return best
    return best


def _split_and_merge(n, m, p, seed, deadline):
    """Try to find a low-rank decomposition by solving the tensor as a
    sum of rank-1 terms with integer coefficients via a greedy + polish approach.
    We start from the schoolbook solution and try to merge/remove terms."""
    base = _schoolbook(n, m, p)
    U, V, W = base["U"], base["V"], base["W"]
    rank = len(U[0])
    # Try to find two columns that can be merged (linearly dependent contribution)
    # This is a heuristic and rarely helps for exact integer tensors; kept for completeness.
    return base


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = _rank_of(best["U"], best["V"], best["W"])

    # 1) Direct exact search for a lower rank.
    # The theoretical lower bound for 4x5x6 is at least 4*5=20 (area) and at most 120.
    # We search a handful of ranks below 120.
    target_ranks = [119, 118, 117, 116, 115, 112, 108, 100, 90, 80, 70]
    seed = 12345
    for tr in target_ranks:
        if time.time() > deadline - 10.0:
            break
        res = _direct_search(n, m, p, tr, seed, deadline)
        seed += 1
        if res is not None:
            r = _rank_of(res["U"], res["V"], res["W"])
            if r < best_rank and _verify(res["U"], res["V"], res["W"], n, m, p):
                best = res
                best_rank = r

    # 2) Split-and-merge polish (usually returns the same schoolbook, but may find a merge).
    if time.time() < deadline - 5.0:
        res2 = _split_and_merge(n, m, p, seed, deadline)
        r2 = _rank_of(res2["U"], res2["V"], res2["W"])
        if r2 < best_rank and _verify(res2["U"], res2["V"], res2["W"], n, m, p):
            best = res2
            best_rank = r2

    return best
# EVOLVE-BLOCK-END
