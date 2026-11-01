"""Random 3-regular graph search with exact girth certification and local repair."""
import time
import random
from collections import deque

_DEADLINE = 150.0  # seconds; evaluator kills at 150s, keep safety margin
_MARGIN = 8.0      # return best before the kill

def _is_3_regular(adj, n):
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    return True

def _check_girth(adj, n, g):
    """Return True iff graph has no cycle of length < g (exact, non-backtracking)."""
    # For each directed edge (u->v), do a depth-limited non-backtracking search
    # up to depth g-2 looking for a return to u (which would form a cycle of length <= g-1).
    # A cycle of length L is found when a non-backtracking walk of length L returns to start
    # without immediately backtracking. We look for walks of length in [3, g-1] closing to start.
    for u in range(n):
        for v in adj[u]:
            # state: (current, prev) ; start (v, u), need to reach (u, x) with x != v after >=2 more steps
            stack = [(v, u, 1)]  # (cur, prev, length)
            while stack:
                cur, prev, length = stack.pop()
                if length >= g - 1:
                    # a non-backtracking walk of length >= g-1 that started at u->v;
                    # if it could close we'd have a cycle >= g, which is fine. We only care
                    # about cycles < g. A closing to u happens at length L in [3, g-1].
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    nl = length + 1
                    if nxt == u and 3 <= nl <= g - 1:
                        return False  # found a cycle shorter than g
                    if nl < g - 1:
                        stack.append((nxt, cur, nl))
    return True

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g, return list of vertices in cycle order, or None."""
    best = None
    for u in range(n):
        for v in adj[u]:
            # BFS non-backtracking
            # state (cur, prev)
            dist = {}
            parent = {}
            q = deque()
            q.append((v, u))
            dist[(v, u)] = 1
            parent[(v, u)] = None
            found = None
            while q:
                cur, prev = q.popleft()
                length = dist[(cur, prev)]
                if length >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    key = (nxt, cur)
                    if key not in dist:
                        dist[key] = length + 1
                        parent[key] = (cur, prev)
                        if nxt == u and 3 <= length + 1 <= g - 1:
                            # reconstruct
                            cyc = []
                            state = key
                            while state is not None:
                                cyc.append(state[0])
                                state = parent[state]
                            cyc.append(u)
                            if best is None or len(cyc) < len(best):
                                best = cyc
                            found = True
                            break
                    elif nxt == u and 3 <= length + 1 <= g - 1:
                        # closing to start
                        cyc = []
                        state = (cur, prev)
                        while state is not None:
                            cyc.append(state[0])
                            state = parent[state]
                        cyc.append(u)
                        if best is None or len(cyc) < len(best):
                            best = cyc
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if best is not None and len(best) <= 5:
            break
    return best

def _generate_random_3reg(n, rng):
    """Generate a random simple 3-regular graph on n vertices via configuration model,
    returning adjacency list or None if it fails to be simple."""
    if n < 4:
        return None
    if n % 2 != 0:
        return None
    half = 3 * n
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    ok = True
    for i in range(0, half, 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            ok = False
            break
        if b in adj[a]:
            ok = False
            break
        adj[a].add(b)
        adj[b].add(a)
    if not ok:
        return None
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return [list(s) for s in adj]

def _repair(adj, n, g, rng, iters=2000):
    """Try to remove short cycles via local edge rewiring. Return (adj, success)."""
    adj = [list(x) for x in adj]
    for _ in range(iters):
        cyc = _find_short_cycle(adj, n, g)
        if cyc is None:
            return adj, True
        # pick a random vertex in the cycle, rewire one of its edges to a random other vertex
        # to break the cycle. We'll re-randomize one vertex's neighbor set partially.
        # Strategy: pick two consecutive cycle vertices (a,b) that are adjacent; remove edge a-b,
        # then find a new partner for a and a new partner for b among non-neighbors to keep degree 3.
        # Simpler: pick a random vertex v in cycle; drop one of its incident edges (v,w) where w in cycle,
        # and reconnect v to a random vertex x not adjacent to v, and reconnect w's freed slot to a random y.
        # To keep it simple and degree-preserving: do a 2-opt style: choose edge (a,b) in cycle,
        # remove it, and add edge (a, c) and (b, d) is not degree preserving. Instead do: remove (a,b),
        # then a needs a new neighbor and b needs a new neighbor. Choose random x,y not in adj and x!=y,
        # x!=b, y!=a, and x not already neighbor of a, etc. But that changes two vertices' degrees by -1 each,
        # so we must add exactly one new edge to each. Add (a,x) and (b,y). Need x!=y, x not adj a, y not adj b,
        # and x,y not equal to a or b, and ensure no parallel edges.
        # Choose a,b as consecutive in cycle.
        L = len(cyc)
        if L < 3:
            return adj, False
        idx = rng.randrange(L)
        a = cyc[idx]
        b = cyc[(idx + 1) % L]
        if b not in adj[a]:
            # try another pair
            for t in range(L):
                a = cyc[t]
                b = cyc[(t + 1) % L]
                if b in adj[a]:
                    break
            else:
                return adj, False
        # remove edge a-b
        adj[a].remove(b)
        adj[b].remove(a)
        # candidate pool: all vertices except a,b and their current neighbors
        def candidates(v):
            return [x for x in range(n) if x != v and x not in adj[v]]
        cand_a = candidates(a)
        cand_b = candidates(b)
        if not cand_a or not cand_b:
            # revert
            adj[a].append(b)
            adj[b].append(a)
            return adj, False
        # pick x from cand_a, y from cand_b with x != y
        for _try in range(30):
            x = rng.choice(cand_a)
            y = rng.choice(cand_b)
            if x == y:
                continue
            # ensure adding (a,x) and (b,y) doesn't create parallel edges (already ensured by candidates)
            adj[a].append(x)
            adj[b].append(y)
            # verify degrees
            if len(adj[a]) == 3 and len(adj[b]) == 3:
                break
            else:
                adj[a].remove(x)
                adj[b].remove(y)
        else:
            # revert
            adj[a].append(b)
            adj[b].append(a)
            return adj, False
    return adj, False

def construct(k, g):
    """Build a k-regular graph of girth >= g with as few vertices as possible.
    Returns (n, edges). For the task k=3, g=16."""
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    if k != 3:
        # fallback: return a trivial valid-ish answer; task is k=3
        # For generality, just return empty if impossible
        return 0, []

    # Moore bound lower bound for (3,16): 2*sum_{i=0}^{7}(2^i) ... actually for girth 16,
    # Moore bound n >= 2*(2^{8}-1)/(2-1) = 2*255 = 510. Best lower bound 512.
    # Start search at a feasible size. Random 3-reg graphs with high girth are known to exist
    # for n up to some limit; girth 16 requires n >= 512. Known best is 936.
    # We'll try sizes from a low bound upward, but to keep within time, start near known-good.
    # Start at 512 and increase in steps; also try random sizes.
    start_n = 512
    # We'll attempt multiple random graphs at increasing n until girth >= g found.
    # To maximize chance, also do repair.

    best = None  # (n, adj)

    # Try a range of n values. Larger n makes girth easier. Known: random 3-reg on n vertices
    # has girth ~ log_2(n). For girth 16 need n ~ 2^16 = 65536 for random, but structured
    # cages do much better. So random alone at 936 may not hit girth 16. Repair may help.
    # Strategy: start at n=936 (known feasible), generate random 3-reg, check girth, repair.
    # If time permits, also try smaller n.

    candidate_ns = [936, 934, 932, 928, 924, 920, 900, 880, 856, 832, 800, 768, 736, 704, 672, 640, 608, 576, 544, 512]

    for n in candidate_ns:
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue
        # try several random graphs at this n
        attempts = 0
        while time.time() < deadline and attempts < 40:
            attempts += 1
            adj = _generate_random_3reg(n, rng)
            if adj is None:
                continue
            if _check_girth(adj, n, g):
                # found!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            # try repair
            if time.time() < deadline:
                adj2, ok = _repair(adj, n, g, rng, iters=300)
                if ok and _check_girth(adj2, n, g):
                    edges = []
                    for i in range(n):
                        for j in adj2[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    # If we didn't find a valid graph within time, return the best we can:
    # return a valid 3-regular graph (even if girth not certified) — but evaluator requires
    # girth >= g. So we must return something valid. Fallback: return the Lazebnik graph
    # (the seed) which is valid. But we removed it. Instead, return a known valid small graph
    # only if girth holds; otherwise we must guarantee validity.
    # Safer fallback: construct a cycle-like 3-reg graph with large girth is hard.
    # Use the seed's algebraic construction as a guaranteed-valid fallback.
    return _fallback_lazebnik(k, g)

def _fallback_lazebnik(k, g):
    """Guaranteed valid 3-regular graph (Lazebnik-Ustimenko-Woldar) for k=3."""
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
