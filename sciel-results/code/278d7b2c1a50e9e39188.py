import time
import random
from collections import deque
from itertools import product


def _luw_construct(k, g):
    # The original LUW construction as fallback (guaranteed valid, large n)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _check_girth(n, edges, g):
    """Full BFS girth check (used only to certify a final candidate)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


def _sl2q(q):
    """Return (mult, inv, det) for SL(2,q) over GF(q). q is a power of a prime."""
    if q & (q - 1) == 0:  # power of 2 -> GF(2^m)
        m = q.bit_length() - 1
        if m == 1:
            return (lambda a, b: a & b), (lambda a: a), (lambda a: a)
        # irreducible primitive polynomial for GF(2^m) (x^m + x + 1 family)
        polys = {
            2: 0b101, 3: 0b1011, 4: 0b10001, 5: 0b100111, 6: 0b1001011,
            7: 0b10000001, 8: 0b100010011, 9: 0b1000000011, 10: 0b10000100101,
            11: 0b100000000011, 12: 0b10000000000011, 13: 0b1000111010001,
            14: 0b100000000001011, 15: 0b10000000000000011,
        }
        p = polys[m]
        mask = (1 << m) - 1
        def _mul(a, b):
            r = 0
            for _ in range(m):
                if b & 1:
                    r ^= a
                b >>= 1
                a <<= 1
                if a & (1 << m):
                    a ^= p
                a &= mask
            return r
        def _inv(a):
            if a == 0:
                return 0
            # a^(2^m - 2)
            e = (1 << m) - 2
            r = 1
            while e:
                if e & 1:
                    r = _mul(r, a)
                a = _mul(a, a)
                e >>= 1
            return r
        return _mul, _inv, _mul
    else:  # odd prime power (only primes used here)
        return (lambda a, b: (a * b) % q), (lambda a: pow(a, q - 2, q) if a else 0), (lambda a: a % q)


def _girth_sl2(q, gens, g, deadline):
    """
    Fast girth test for the Cayley graph of SL(2,q) with generators gens.
    Girth >= g (g even) iff no non-backtracking closed walk of length < g.
    In a Cayley graph a shortest cycle has no consecutive equal/identity steps,
    so it corresponds to a word in the generator set of length < g whose
    product is the identity. We enumerate all such words (pruning identity
    prefixes) and check for the identity matrix.
    """
    mult, inv, det = _sl2q(q)
    I = (1, 0, 0, 1)
    L = g - 1  # max word length to check
    # precompute: for each matrix m (as 4-tuple) and each gen, product m*gen
    # but we build on the fly with memo of (m, gen) -> m*gen
    memo = {}

    def mmul(m, n):
        key = (m, n)
        r = memo.get(key)
        if r is not None:
            return r
        a00, a01, a10, a11 = m
        b00, b01, b10, b11 = n
        r = (
            mult(a00, b00) + mult(a01, b10),
            mult(a00, b01) + mult(a01, b11),
            mult(a10, b00) + mult(a11, b10),
            mult(a10, b01) + mult(a11, b11),
        )
        r = (r[0] % q, r[1] % q, r[2] % q, r[3] % q)
        memo[key] = r
        return r

    # words of length 1
    for gen in gens:
        if gen == I:
            return False
    # words of length >= 2, no consecutive identical (non-backtracking)
    # state: (matrix, last_gen_index)
    states = [(gen, i) for i, gen in enumerate(gens)]
    for length in range(2, L + 1):
        if time.time() > deadline:
            return None  # timeout -> treat as unknown, don't accept
        new_states = []
        for mat, last in states:
            for j, gen in enumerate(gens):
                if j == last:
                    continue
                nm = mmul(mat, gen)
                if nm == I:
                    return False
                new_states.append((nm, j))
        states = new_states
    return True


def _sl2_edges(q, gens):
    mult, inv, det = _sl2q(q)
    n = q * q * (q - 1)
    # enumerate all SL(2,q) matrices
    mats = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # d = (1 - a*b - c*0)/ (b*1) ... need det=1: a*d - b*c = 1
                # choose d such that a*d - b*c = 1
                # solve for d: a*d = 1 + b*c  =>  if a != 0: d = (1+b*c)*a^{-1}
                # handle a=0 separately
                pass
    # Simpler enumeration: iterate (a,b,c), compute d
    inv2 = inv
    mats = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # a*d - b*c = 1
                # if a != 0: d = (1 + b*c) * a^{-1}
                # if a == 0: -b*c = 1 => b*c = -1, then d free? No: a=0 => -b*c=1, d arbitrary
                if a != 0:
                    d = (1 + b * c) * inv2(a) % q
                    m = (a, b, c, d)
                    if m not in seen:
                        seen.add(m)
                        mats.append(m)
                else:
                    # a=0: -b*c = 1 => b*c = -1 mod q
                    # b must be nonzero
                    if b != 0 and (b * c) % q == (q - 1) % q:
                        # d free: iterate all d
                        for d in range(q):
                            m = (0, b, c, d)
                            if m not in seen:
                                seen.add(m)
                                mats.append(m)
    # map matrix -> index
    idx = {m: i for i, m in enumerate(mats)}
    mult2 = mult
    edges = []
    for i, m in enumerate(mats):
        for gen in gens:
            nm = (
                (mult2(m[0], gen[0]) + mult2(m[1], gen[2])) % q,
                (mult2(m[0], gen[1]) + mult2(m[1], gen[3])) % q,
                (mult2(m[2], gen[0]) + mult2(m[3], gen[2])) % q,
                (mult2(m[2], gen[1]) + mult2(m[3], gen[3])) % q,
            )
            j = idx[nm]
            if i < j:
                edges.append((i, j))
    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Fallback (always valid, large)
    n_luw, edges_luw = _luw_construct(3, 16)
    best = (n_luw, edges_luw)

    # Candidate generator sets for SL(2,16). We need 3 generators whose
    # Cayley graph has girth >= 16. Try several structured sets + random.
    q = 16
    mult, inv, det = _sl2q(q)

    # Standard-ish generators: two unipotents and one diagonal
    candidates = [
        ((1, 1, 0, 1), (1, 0, 1, 1), (2, 0, 0, 8)),
        ((1, 1, 0, 1), (1, 0, 1, 1), (3, 0, 0, 11)),
        ((1, 2, 0, 1), (1, 0, 3, 1), (2, 0, 0, 8)),
        ((1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 1, 0)),
        ((1, 2, 0, 1), (1, 0, 1, 1), (2, 0, 0, 9)),
    ]

    # add random generators (det=1 matrices)
    rng = random.Random(12345)
    for _ in range(40):
        while True:
            a = rng.randrange(q); b = rng.randrange(q)
            c = rng.randrange(q); d = rng.randrange(q)
            if (a * d - b * c) % q == 1:
                break
        candidates.append(((1, 1, 0, 1), (1, 0, 1, 1), (a, b, c, d)))

    for gens in candidates:
        if time.time() > deadline - 20:
            break
        # quick sanity: generators distinct and not identity
        if len(set(gens)) != 3:
            continue
        if any(m == (1, 0, 0, 1) for m in gens):
            continue
        res = _girth_sl2(q, gens, g, deadline - 20)
        if res is True:
            n, edges = _sl2_edges(q, gens)
            if n < best[0]:
                # certify with full BFS before accepting
                if _check_girth(n, edges, g):
                    best = (n, edges)
                    break
            else:
                # same size but valid; keep smaller
                if _check_girth(n, edges, g):
                    best = (n, edges)
                    break

    return best
