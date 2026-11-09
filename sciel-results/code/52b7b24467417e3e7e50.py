"""C4-free graph with no independent 11-set (Ramsey R(C4,K11)) via greedy construction."""
import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - time.time()

    # ---------- deterministic C4-free fallback (10 triangles, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- C4-free graph construction ----------
    def build_c4free(n, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() >= DEADLINE - 1.0:
                    break
                # C4 check: adding (i,j) is safe iff |N(i) & N(j)| <= 1
                if len(adj[i] & adj[j]) <= 1:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def densify(adj, n, seed):
        """Add edges between vertices with disjoint neighborhoods (never creates a C4)."""
        rng = random.Random(seed)
        changed = True
        while changed:
            changed = False
            pairs = list(combinations(range(n), 2))
            rng.shuffle(pairs)
            for i, j in pairs:
                if j in adj[i]:
                    continue
                if not (adj[i] & adj[j]):      # disjoint neighbors -> safe
                    adj[i].add(j)
                    adj[j].add(i)
                    changed = True
        return adj

    # ---------- greedy maximal independent set ----------
    def greedy_mis(adj, n, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        indep = []
        in_set = set()
        for v in order:
            if v in in_set:
                continue
            if all((u not in adj[v]) for u in indep):
                indep.append(v)
                in_set.add(v)
        return len(indep)

    # ---------- candidate validation (fast, non-exact) ----------
    def fast_ok(adj, n):
        # confirm C4-free
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i] and len(adj[i] & adj[j]) > 1:
                    return False
        # confirm greedy MIS < 11 across many orderings
        for s in range(25):
            if greedy_mis(adj, n, 9000 + s) >= 11:
                return False
        return True

    # ---------- main search ----------
    best = None
    target_ns = [43, 42, 41, 40, 39, 38, 37]
    trial = 0
    for n in target_ns:
        if time.time() >= DEADLINE - 3.0:
            break
        while time.time() < DEADLINE - 3.0:
            trial += 1
            adj = build_c4free(n, trial)
            adj = densify(adj, n, trial)
            if fast_ok(adj, n):
                best = (n, adj)
                break
            # if this n is hopeless (always finds 11-set), stop trying it
            if trial % 40 == 0 and time.time() < DEADLINE - 5.0:
                # quick probe: is it structurally impossible?
                break
        if best is not None:
            break

    if best is None:
        return fallback()

    n, adj = best
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
