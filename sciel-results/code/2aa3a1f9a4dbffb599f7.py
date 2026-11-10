import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 10
    p, q = 2, 5

    def decomp(i):
        return (i % p, i // p)

    def crt(x, y):
        return x * q + y

    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        seen = set()
        for i in range(n):
            row1 = sq1[i]
            row2 = sq2[i]
            for j in range(n):
                pair = (row1[j], row2[j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Build candidate pool
    candidates = []

    # Type A: L_k for k in Z_5 (5 squares): symbol ((a+c) mod 2, (b + k*d) mod 5)
    for k in range(5):
        sq = [[0] * n for _ in range(n)]
        for i in range(n):
            a, b = decomp(i)
            for j in range(n):
                c, d = decomp(j)
                x = (a + c) % p
                y = (b + k * d) % q
                sq[i][j] = crt(x, y)
        candidates.append(sq)

    # Type B: add offset depending on a (row's Z2 part) to the Z5 symbol
    # symbol ((a+c) mod 2, (b + d + t*a) mod 5) for t in 0..4
    for t in range(5):
        sq = [[0] * n for _ in range(n)]
        for i in range(n):
            a, b = decomp(i)
            for j in range(n):
                c, d = decomp(j)
                x = (a + c) % p
                y = (b + d + t * a) % q
                sq[i][j] = crt(x, y)
        candidates.append(sq)

    # Type C: swap roles - use d in Z2-part combination
    # symbol ((a + c + s*b) mod 2, (b + d) mod 5) for s in 0..1
    for s in range(2):
        sq = [[0] * n for _ in range(n)]
        for i in range(n):
            a, b = decomp(i)
            for j in range(n):
                c, d = decomp(j)
                x = (a + c + s * b) % p
                y = (b + d) % q
                sq[i][j] = crt(x, y)
        candidates.append(sq)

    # Type D: general linear: x = (a + c + r*b) mod 2, y = (b + k*d + t*a + u*c) mod 5
    # keep it bounded to avoid explosion
    for k in range(5):
        for t in range(5):
            sq = [[0] * n for _ in range(n)]
            for i in range(n):
                a, b = decomp(i)
                for j in range(n):
                    c, d = decomp(j)
                    x = (a + c) % p
                    y = (b + k * d + t * a) % q
                    sq[i][j] = crt(x, y)
            candidates.append(sq)

    # Deduplicate and filter Latin
    seen_keys = set()
    valid = []
    for sq in candidates:
        if not is_latin(sq):
            continue
        key = tuple(tuple(row) for row in sq)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        valid.append(sq)

    # Greedily select mutually orthogonal set
    mols = []
    for sq in valid:
        if all(are_orthogonal(sq, existing) for existing in mols):
            mols.append(sq)
        if len(mols) >= 9:
            break

    # Try to extend with random linear squares if under target
    if len(mols) < 8 and time.time() < deadline:
        import random
        rng = random.Random(12345)
        base = [[(i + j) % n for j in range(n)] for i in range(n)]
        for _ in range(20000):
            if time.time() > deadline:
                break
            perm = list(range(n))
            rng.shuffle(perm)
            sq = [[perm[base[i][j]] for j in range(n)] for i in range(n)]
            if not is_latin(sq):
                continue
            key = tuple(tuple(row) for row in sq)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if all(are_orthogonal(sq, existing) for existing in mols):
                mols.append(sq)
                if len(mols) >= 9:
                    break

    # Ensure pure Python int types and list-of-lists
    result = [[int(sq[i][j]) for j in range(n)] for i in range(n)] if mols else None
    out = []
    for sq in mols:
        out.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    return out
