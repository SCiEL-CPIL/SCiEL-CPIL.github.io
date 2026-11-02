"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _target_tensor():
    n, m, p = DIMENSIONS
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    return T

def _key_from_factors(a, b, c):
    return (a[0], b[0], c[0]), (a[1], b[1], c[1]), (a[2], b[2], c[2])

def _candidate_pool(rng, n, m, p, pool_size=3500, max_terms=4):
    # Candidate vectors are sparse with entries in {0, 1/2, 1}.
    U = []
    V = []
    W = []

    # Deterministic basis-like vectors
    for i in range(n):
        for j in range(m):
            u = [0] * n
            u[i] = 1
            U.append(tuple(u))
    for j in range(m):
        for k in range(p):
            v = [0] * m
            v[j] = 1
            V.append(tuple(v))
    for k in range(p):
        for i in range(n):
            c = [0] * p
            c[k] = 1
            W.append(tuple(c))

    # Random sparse vectors
    def rand_vec(dim, rng, max_terms, half_prob):
        vec = [Fraction(0)] * dim
        if dim == 0:
            return tuple(vec)
        ts = min(max_terms, dim)
        idxs = rng.sample(range(dim), ts)
        for idx in idxs:
            val = Fraction(1, 2) if rng.random() < half_prob else Fraction(1)
            if rng.random() < 0.5:
                val = -val
            vec[idx] = val
        return tuple(vec)

    target_total = pool_size
    added = len(U) + len(V) + len(W)
    while added < target_total:
        rem = target_total - added
        u = rand_vec(n, rng, max_terms, 0.25)
        v = rand_vec(m, rng, max_terms, 0.25)
        c = rand_vec(p, rng, max_terms, 0.25)
        U.append(u)
        V.append(v)
        W.append(c)
        added += 3

    return U, V, W

def _greedy_construct(U, V, W, T, deadline, rng):
    # Greedily pick triples whose product has maximum agreement with unexplained target entries.
    n, m, p = DIMENSIONS
    remaining = dict(T)
    chosen = []

    # Index candidates by first coordinate to reduce combinations.
    U_by_i = defaultdict(list)
    for idx, u in enumerate(U):
        U_by_i[u[0]].append(idx)
    V_by_j = defaultdict(list)
    for idx, v in enumerate(V):
        V_by_j[v[0]].append(idx)
    W_by_k = defaultdict(list)
    for idx, c in enumerate(W):
        W_by_k[c[0]].append(idx)

    # Precompute nonzero entries for speed
    U_nz = [ [pos for pos, val in enumerate(vec) if val != 0] for vec in U ]
    V_nz = [ [pos for pos, val in enumerate(vec) if val != 0] for vec in V ]
    W_nz = [ [pos for pos, val in enumerate(vec) if val != 0] for vec in W ]

    while remaining:
        if time.time() > deadline:
            return None
        best_gain = 0
        best_combo = None
        # Randomize order of target keys to avoid deterministic bias
        keys = list(remaining.keys())
        rng.shuffle(keys)
        for i, j, k in keys:
            if time.time() > deadline:
                return None
            u_idxs = U_by_i[i]
            v_idxs = V_by_j[j]
            c_idxs = W_by_k[k]
            if not u_idxs or not v_idxs or not c_idxs:
                continue
            # Limit candidates per coordinate to control cost
            if len(u_idxs) > 60:
                u_idxs = rng.sample(u_idxs, 60)
            if len(v_idxs) > 60:
                v_idxs = rng.sample(v_idxs, 60)
            if len(c_idxs) > 60:
                c_idxs = rng.sample(c_idxs, 60)

            for ui in u_idxs:
                u = U[ui]
                u_nz = U_nz[ui]
                if not u_nz:
                    continue
                for vi in v_idxs:
                    v = V[vi]
                    v_nz = V_nz[vi]
                    if not v_nz:
                        continue
                    for ci in c_idxs:
                        c = W[ci]
                        c_nz = W_nz[ci]
                        if not c_nz:
                            continue
                        # Compute product entries where all three have nonzero
                        # We only care about keys in remaining
                        gain = 0
                        for ipos in u_nz:
                            uval = u[ipos]
                            for jpos in v_nz:
                                vval = v[jpos]
                                for kpos in c_nz:
                                    cval = c[kpos]
                                    key = (ipos, jpos, kpos)
                                    if key in remaining:
                                        prod = uval * vval * cval
                                        if prod == 1:
                                            gain += 1
                                        elif prod == 0:
                                            pass
                                        else:
                                            # wrong sign or magnitude: penalty
                                            gain -= 2
                        if gain > best_gain:
                            best_gain = gain
                            best_combo = (ui, vi, ci)
                            if gain >= 10:
                                # good enough, stop early for this iteration
                                break
                    if gain >= 10:
                        break
                if gain >= 10:
                    break

        if best_combo is None:
            return None
        ui, vi, ci = best_combo
        chosen.append((ui, vi, ci))
        u = U[ui]
        v = V[vi]
        c = W[ci]
        # Update remaining
        for ipos, uval in enumerate(u):
            if uval == 0:
                continue
            for jpos, vval in enumerate(v):
                if vval == 0:
                    continue
                for kpos, cval in enumerate(c):
                    if cval == 0:
                        continue
                    key = (ipos, jpos, kpos)
                    if key in remaining:
                        prod = uval * vval * cval
                        if prod == 1:
                            del remaining[key]
                        elif prod == 0:
                            pass
                        else:
                            # If product is -1 or 1/2 etc, we cannot cancel easily;
                            # mark as impossible for greedy (will fail later)
                            remaining[key] = prod
        # If remaining has non-1 values, greedy is stuck
        for val in remaining.values():
            if val != 1:
                return None

    return chosen

def _build_output(chosen, U, V, W):
    rank = len(chosen)
    n, m, p = DIMENSIONS
    Uout = [[Fraction(0)] * rank for _ in range(n * m)]
    Vout = [[Fraction(0)] * rank for _ in range(m * p)]
    Wout = [[Fraction(0)] * rank for _ in range(p * n)]

    for col, (ui, vi, ci) in enumerate(chosen):
        u = U[ui]
        v = V[vi]
        c = W[ci]
        for i in range(n):
            Uout[i * m + (0 if False else 0)][col] = u[i]  # placeholder, fix below
        # Correct indexing: U row is i*m + j
        for i in range(n):
            for j in range(m):
                Uout[i * m + j][col] = u[i] * v[j]
        for j in range(m):
            for k in range(p):
                Vout[j * p + k][col] = v[j] * c[k]
        for k in range(p):
            for i in range(n):
                Wout[k * n + i][col] = c[k] * u[i]

    # Convert to float or int if possible
    def convert_matrix(mat):
        out = []
        for row in mat:
            newrow = []
            for val in row:
                if val.denominator == 1:
                    newrow.append(int(val))
                else:
                    newrow.append(float(val))
            out.append(newrow)
        return out

    return {
        "U": convert_matrix(Uout),
        "V": convert_matrix(Vout),
        "W": convert_matrix(Wout)
    }

def construct():
    start = time.time()
    deadline = start + 120  # safety margin
    T = _target_tensor()
    n, m, p = DIMENSIONS
    rng = random.Random(42)

    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback

    # Try greedy search with increasing pool sizes
    pool_sizes = [500, 1000, 2000, 3500]
    for pool_size in pool_sizes:
        if time.time() > deadline - 10:
            break
        U, V, W = _candidate_pool(rng, n, m, p, pool_size)
        chosen = _greedy_construct(U, V, W, T, deadline - 5, rng)
        if chosen is not None:
            rank = len(chosen)
            if rank < best_rank:
                best_rank = rank
                best_result = _build_output(chosen, U, V, W)
                # If we found a very good rank, we can stop
                if rank <= 60:
                    break

    # Fallback: schoolbook
    if best_result is None or best_rank == n * m * p:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END
