"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _popcount(x):
    return bin(x).count("1")

def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin under 90s
    N = 512

    def elapsed():
        return time.time() - t0

    def deadline():
        return elapsed() > DEADLINE

    # prime-power factorization sets for 1..N
    pp = [0] * (N + 1)
    for v in range(1, N + 1):
        x = v
        bits = 0
        p = 2
        while p * p <= x:
            if x % p == 0:
                pe = 1
                while x % p == 0:
                    x //= p
                    pe *= p
                bits |= (1 << pe)
            p += 1
        if x > 1:
            bits |= (1 << x)
        pp[v] = bits

    # adjacency bitsets: adj[i] has bit j set iff i != j and lcm(i, j) <= N
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        pi = pp[i]
        bits = 0
        for j in range(1, N + 1):
            if i != j and (pi | pp[j]) <= N:
                bits |= (1 << j)
        adj[i] = bits
        if deadline():
            break

    # candidate order by descending degree
    order = sorted(range(1, N + 1), key=lambda i: _popcount(adj[i]), reverse=True)
    # greedy upper bound
    greedy = 0
    avail = 0
    for i in range(1, N + 1):
        avail |= (1 << i)
    for i in order:
        if avail & (1 << i):
            greedy += 1
            avail &= adj[i]
    best = greedy
    best_clique = [i for i in order if (avail0 := 0) or True]  # placeholder, recompute below

    # recompute a concrete greedy clique of size `best`
    best_clique = []
    avail = 0
    for i in range(1, N + 1):
        avail |= (1 << i)
    for i in order:
        if avail & (1 << i):
            best_clique.append(i)
            avail &= adj[i]
    best = len(best_clique)

    # Bron-Kerbosch with pivot
    best_size = best
    best_set = set(best_clique)

    def bk(R, P, X):
        nonlocal best_size, best_set
        if deadline():
            return
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_set = set(R)
            return
        if len(R) + _popcount(P) <= best_size:
            return
        # choose pivot u in P|X maximizing |P & adj[u]|
        PX = P | X
        u = 0
        best_deg = -1
        tmp = PX
        while tmp:
            lb = tmp & (-tmp)
            j = lb.bit_length() - 1
            d = _popcount(P & adj[j])
            if d > best_deg:
                best_deg = d
                u = j
            tmp ^= lb
        candidates = P & ~adj[u]
        tmp = candidates
        while tmp:
            lb = tmp & (-tmp)
            j = lb.bit_length() - 1
            tmp ^= lb
            R.append(j)
            bk(R, P & adj[j], X & adj[j])
            R.pop()
            P ^= lb
            X |= lb
            if deadline():
                return

    bk([], 0, 0)
    # initial P: all nodes
    P0 = 0
    for i in range(1, N + 1):
        P0 |= (1 << i)
    bk([], P0, 0)

    if deadline():
        return sorted(best_set)

    return sorted(best_set)
# EVOLVE-BLOCK-END
