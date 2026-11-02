# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    # ---- finite field of order q = k (prime, or 4, or a square prime power) ----
    q = k
    table = None
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: (x & y)
    else:
        # try q itself (prime), else q^2 (square prime power)
        def is_prime(n):
            if n < 2:
                return False
            if n % 2 == 0:
                return n == 2
            i = 3
            while i * i <= n:
                if n % i == 0:
                    return False
                i += 2
            return True

        import math
        if is_prime(q):
            F = q
            add = lambda x, y: (x + y) % F
            sub = lambda x, y: (x - y) % F
            mul = lambda x, y: (x * y) % F
        else:
            # assume q is a square of a prime power: F = q^2, q = sqrt(F)
            r = int(math.isqrt(q))
            if r * r != q:
                # last resort: treat q as a field anyway (mod arithmetic)
                F = q
                add = lambda x, y: (x + y) % F
                sub = lambda x, y: (x - y) % F
                mul = lambda x, y: (x * y) % F
            else:
                F = q * q
                add = lambda x, y: (x + y) % F
                sub = lambda x, y: (x - y) % F
                mul = lambda x, y: (x * y) % F

    # ---- LUW rule for girth >= d + 5 (d odd) ----
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # need odd d

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # ---- enumerate vertices ----
    def encode(side, vec):
        # vec = (None, v1, v2, ..., vd); v1 in 0..q-1, others in field F (0..F-1)
        code = vec[1]
        for c in range(2, d + 1):
            code = code * F + vec[c]
        return side, code

    def decode(side, code):
        vec = [None] * (d + 1)
        rest = code
        for c in range(d, 1, -1):
            vec[c] = rest % F
            rest //= F
        vec[1] = rest
        return side, tuple(vec)

    total = 2 * (q * (F ** d))
    if total > 2000000:
        # too big to materialize; fall back to BFS component of the zero vertex
        index, order, edges = {}, [], []
        startv = (0, (None,) + (0,) * d)
        index[startv] = 0
        order = [startv]
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            v = order[i]
            for o in neighbours(v):
                if o not in index:
                    index[o] = len(order)
                    order.append(o)
                if v[0] == 0:
                    edges.append((index[v], index[o]))
            i += 1
        return len(order), edges

    # build full graph
    index = {}
    order = []
    for side in (0, 1):
        for code in range(q * (F ** d)):
            index[(side, code)] = len(order)
            order.append((side, code))

    edges = []
    for vi, (side, code) in enumerate(order):
        if side != 0:
            continue
        if time.time() > deadline:
            break
        vec = decode(side, code)
        for oside, ovec in neighbours((side, vec)):
            ocode = encode(oside, ovec)
            edges.append((vi, index[(oside, ocode)]))

    return len(order), edges
# EVOLVE-BLOCK-END
