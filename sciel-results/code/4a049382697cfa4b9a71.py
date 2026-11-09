# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_DEADLINE = time.time() + 315.0


def _neighbours(p):
    x, y, z = p
    return ((x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1))


# The 24 proper rotations of the cube as 3x3 integer matrices.
def _build_rotations():
    mats = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                # R = Rz(a) @ Ry(b) @ Rx(c)
                ax = a * 90; ay = b * 90; az = c * 90
                cx = 1 if ax == 0 else 0; sx = 0 if ax == 0 else 1
                cy = 1 if ay == 0 else 0; sy = 0 if ay == 0 else 1
                cz = 1 if az == 0 else 0; sz = 0 if az == 0 else 1
                # Rx
                Rx = ((cx, 0, 0), (0, cy, -sx), (0, sx, cy))
                # Ry
                Ry = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
                # Rz
                Rz = ((cz, -sz, 0), (sz, cz, 0), (0, 0, 1))
                # M = Rz @ Ry @ Rx
                def matmul(A, B):
                    C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    for i in range(3):
                        for k in range(3):
                            aik = A[i][k]
                            if aik:
                                for j in range(3):
                                    C[i][j] += aik * B[k][j]
                    return C
                M = matmul(Rz, matmul(Ry, Rx))
                mats.append(M)
    # deduplicate
    seen = set()
    uniq = []
    for M in mats:
        key = (M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2])
        if key not in seen:
            seen.add(key)
            uniq.append(M)
    return uniq


_ROTATIONS = _build_rotations()
_ROTATIONS = [m for m in _ROTATIONS if (m[0][0], m[0][1], m[0][2]) != (1, 0, 0)
              or (m[1][0], m[1][1], m[1][2]) != (0, 1, 0)]  # drop identity later if wanted


def _apply_rot(M, p):
    return (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2],
            M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2],
            M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2])


def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, rng, bias=0.5):
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 1
            score += rng.random() * 0.4
            options.append((score, site))
        if not options:
            return None
        if bias > 0.8:
            options.sort(reverse=True)
            site = options[0][1]
        else:
            top_k = min(3, len(options))
            site = options[rng.randint(0, top_k - 1)][1]
        chain.append(site)
        occupied[site] = i
    return chain


class _State:
    __slots__ = ('sequence', 'chain', 'occupied', 'energy', 'hset')

    def __init__(self, sequence, chain):
        self.sequence = sequence
        self.chain = chain
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.hset = {i for i, c in enumerate(sequence) if c == 'H'}
        self.energy = _compute_energy(sequence, chain)


def _try_crankshaft(st, start, length, rng, T):
    """Rotate chain[start:start+length] about anchor chain[start-1]. Returns new State or None."""
    seq = st.sequence
    chain = st.chain
    occupied = st.occupied
    n = len(chain)
    end = start + length
    if start < 1 or end > n:
        return None
    anchor = chain[start - 1]

    # Collect the segment points
    seg = chain[start:end]

    # Remove segment from occupied
    for p in seg:
        del occupied[p]

    # Relative coords to anchor
    rel = [(p[0] - anchor[0], p[1] - anchor[1], p[2] - anchor[2]) for p in seg]

    # Build candidate rotations: for each of 24 rotations, compute new segment
    candidates = []
    for M in _ROTATIONS:
        new_seg = []
        ok = True
        for rp in rel:
            np_ = _apply_rot(M, rp)
            np_ = (np_[0] + anchor[0], np_[1] + anchor[1], np_[2] + anchor[2])
            if np_ in occupied:
                ok = False
                break
            new_seg.append(np_)
        if not ok:
            continue
        # internal self-collision check within new_seg
        sset = set()
        for np_ in new_seg:
            if np_ in sset:
                ok = False
                break
            sset.add(np_)
        if not ok:
            continue
        # first step must differ from original (else identity)
        if new_seg[0] == seg[0]:
            continue
        candidates.append(new_seg)

    if not candidates:
        # restore
        for i, p in enumerate(seg):
            occupied[chain[start + i]] = start + i
        return None

    # Score each candidate by incremental energy delta
    hset = st.hset
    best_delta = None
    best_new_seg = None
    # For energy we need the full new chain; compute delta by contacts involving segment
    # Build base chain list
    base = chain  # full current
    deltas = []
    for new_seg in candidates:
        # new chain
        new_chain = base[:start] + new_seg + base[end:]
        # compute delta: contacts involving any segment index
        # removed: contacts where one endpoint in [start,end) and other not, non-adjacent, both H
        # We'll just compute the delta by scanning segment endpoints.
        # Build a where for new chain
        where_new = {p: i for i, p in enumerate(new_chain)}
        # removed contacts: pairs (a,b) both H, |a-b|>1, one in segment, that existed in old chain
        # and are not present in new chain.
        # Simpler: compute old and new contact sets restricted to segment, then delta = new - old.
        def seg_contacts(chain_list, where):
            c = 0
            for idx in range(start, end):
                if idx not in hset:
                    continue
                p = chain_list[idx]
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j not in hset:
                        continue
                    if abs(j - idx) > 1:
                        # count once: only when j>idx
                        if j > idx:
                            c += 1
            return c
        where_old = {p: i for i, p in enumerate(base)}
        old_c = seg_contacts(base, where_old)
        new_c = seg_contacts(new_chain, where_new)
        d = (new_c - old_c)  # contacts increase -> energy decrease
        # energy delta = - (new_c - old_c)
        ed = -(new_c - old_c)
        deltas.append((ed, new_seg, new_chain))

    if not deltas:
        for i, p in enumerate(seg):
            occupied[chain[start + i]] = start + i
        return None

    # Boltzmann choose
    if T < 0.05:
        deltas.sort(key=lambda x: x[0])
        chosen = deltas[0]
    else:
        max_e = max(d[0] for d in deltas)
        exps = [math.exp((d[0] - max_e) / T) for d in deltas]
        tot = sum(exps)
        r = rng.random() * tot
        acc = 0.0
        chosen = deltas[-1]
        for k, d in enumerate(deltas):
            acc += exps[k]
            if r <= acc:
                chosen = d
                break

    ed, new_seg, new_chain = chosen
    # commit
    st.chain = new_chain
    # rebuild occupied for segment
    for i, p in enumerate(new_seg):
        occupied[start + i] = p
    st.energy += ed
    return st


def _sa_fold(sequence, rng, time_limit):
    n = len(sequence)
    if n < 2:
        return [(0, 0, 0)] * n

    # best initial
    best_init = None
    best_init_e = float('inf')
    for _ in range(12):
        c = _grow_greedy(sequence, rng, bias=0.3 + rng.random() * 0.5)
        if c is None:
            continue
        e = _compute_energy(sequence, c)
        if e < best_init_e:
            best_init_e = e
            best_init = c
    if best_init is None:
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_e = _compute_energy(sequence, best_init)

    st = _State(sequence, best_init)
    best_energy = st.energy
    best_chain = list(st.chain)

    start_t = time.time()
    deadline = start_t + time_limit
    T = 2.5
    T_min = 0.03
    cool = 0.9975

    while T > T_min:
        if time.time() > deadline:
            break
        # choose crankshaft params
        if n <= 40:
            L = rng.randint(2, min(12, n - 1))
        else:
            L = rng.randint(2, min(16, n - 1))
        start = rng.randint(1, n - L - 1) if n - L - 1 > 1 else 1
        start = max(1, min(start, n - L - 1))
        if n - L - 1 < 1:
            L = 2
            start = 1
        new_st = _try_crankshaft(st, start, L, rng, T)
        if new_st is not None:
            st = new_st
            if st.energy < best_energy:
                best_energy = st.energy
                best_chain = list(st.chain)
        T *= cool
    return best_chain


def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining <= 1.0:
        # emergency: return a valid straight line
        return [[i, 0, 0] for i in range(n)]

    # per-sequence cap scaled by length, with safety margin
    if n <= 50:
        cap = 12.0
    elif n <= 80:
        cap = 14.0
    else:
        cap = 16.0
    cap = min(cap, remaining * 0.9)
    if cap < 2.0:
        cap = max(1.0, remaining * 0.5)

    if n <= 50:
        restarts = 4
    elif n <= 80:
        restarts = 3
    else:
        restarts = 2

    slice_time = cap / restarts
    rng = random.Random(hash(sequence) % (2 ** 32))

    best_chain = None
    best_energy = float('inf')
    for r in range(restarts):
        if time.time() > _GLOBAL_DEADLINE - 1.0:
            break
        c = _sa_fold(sequence, rng, slice_time)
        if c is not None:
            e = _compute_energy(sequence, c)
            if e < best_energy:
                best_energy = e
                best_chain = c

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
