import time, math, random

_MAX = 128
_START = time.time()
_DEADLINE = _START + 80.0

def construct():
    moduli = list(range(2, _MAX+1))
    n = len(moduli)
    # gcd table
    gcd_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_table[i][j] = math.gcd(moduli[i], moduli[j])
    idx = {m:i for i,m in enumerate(moduli)}

    def build(order):
        chosen = [None]*n
        chosen_moduli = []
        for m in order:
            i = idx[m]
            placed = False
            for r in range(m):
                ok = True
                for om in chosen_moduli:
                    j = idx[om]
                    g = gcd_table[i][j]
                    if g > 1 and (r % g) == (chosen[om] % g):
                        ok = False
                        break
                if ok:
                    chosen[i] = r
                    chosen_moduli.append(m)
                    placed = True
                    break
        return [[chosen[idx[m]], m] for m in chosen_moduli]

    # constraint count
    def ccount(m):
        i = idx[m]
        return sum(1 for j in range(n) if j!=i and gcd_table[i][j]>1)
    ccount_cache = {m: ccount(m) for m in moduli}

    best = []
    best_len = 0
    def consider(sol):
        nonlocal best, best_len
        if len(sol) > best_len:
            best_len = len(sol)
            best = sol

    # deterministic orderings
    orders = [
        sorted(moduli, key=lambda m: (-ccount_cache[m], m)),
        sorted(moduli, key=lambda m: (ccount_cache[m], m)),
        sorted(moduli, key=lambda m: m),
        sorted(moduli, key=lambda m: -m),
    ]
    for o in orders:
        consider(build(o))

    # random restarts
    random.seed(12345)
    while time.time() < _DEADLINE:
        o = moduli[:]
        random.shuffle(o)
        consider(build(o))

    if not best:
        best = [[1,2]]
    return best
