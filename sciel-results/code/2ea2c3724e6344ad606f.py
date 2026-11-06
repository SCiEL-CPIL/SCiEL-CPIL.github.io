import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s

    def time_ok():
        return time.time() < deadline

    # Precompute all undirected 4-cycles (as frozensets of 4 edges)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_set = set(all_pairs)

    def cano_edge(a, b):
        return (a, b) if a < b else (b, a)

    def cano4(a, b, c, d):
        return (a, b, c, d) if a < b and b < c and c < d else None

    # For each pair (u,v): list of 4-cycles containing that edge, stored as
    # (the 3 other edges as a frozenset, and the 4th vertex info not needed).
    cycles_for_edge = {p: [] for p in all_pairs}

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # 4-cycles on {a,b,c,d}
                    for (p1, p2, p3) in [((a, b), (b, c), (c, d)),
                                          ((a, b), (b, d), (d, c)),
                                          ((a, c), (c, b), (b, d))]:
                        e1, e2, e3 = p1, p2, p3
                        # the 4th edge connects the two "free" endpoints
                        # endpoints of the path
                        # path vertices: derive
                        # we need the 4 edges forming a cycle; the 4th edge
                        # is the one not in {e1,e2,e3} among the 6 pairs
                        # that closes the cycle.
                        # Determine the cycle edges explicitly.
                        pass
    # Simpler: enumerate cycles by choosing the 4 distinct vertices and the
    # 3 possible pairings, building the actual 4-edge set.
    cycles_for_edge = {p: [] for p in all_pairs}

    def cycle_edge_set(a, b, c, d, pattern):
        # pattern: order of the 4 vertices around the cycle
        vs = [a, b, c, d]
        edges = [cano_edge(vs[i], vs[(i + 1) % 4]) for i in range(4)]
        return frozenset(edges)

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # 3 distinct 4-cycles on these 4 vertices
                    for perm in [
                        [a, b, c, d],
                        [a, b, d, c],
                        [a, c, b, d],
                    ]:
                        es = cycle_edge_set(perm[0], perm[1], perm[2], perm[3], None)
                        for e in es:
                            cycles_for_edge[e].append(es)

    # Map: edge -> list of 4-cycle edge-sets containing it
    # Also map: 3-edge "path" -> the edge that completes each cycle.

    best_edges = []
    best_count = 0

    class State:
        __slots__ = ('adj', 'color', 'nbrs', 'edges', 'cycle_colors', 'cycle_valid')

        def __init__(self):
            self.adj = [[False] * n for _ in range(n)]
            self.color = [[-1] * n for _ in range(n)]
            self.nbrs = [set() for _ in range(n)]
            self.edges = []
            self.cycle_colors = {}   # cycle -> dict edge->color for present edges
            self.cycle_valid = {}    # cycle -> bool (all 4 present & distinct)

        def add(self, u, v, c):
            self.adj[u][v] = True
            self.adj[v][u] = True
            self.color[u][v] = c
            self.color[v][u] = c
            self.nbrs[u].add(v)
            self.nbrs[v].add(u)
            self.edges.append((u, v, c))
            e = cano_edge(u, v)
            for cyc in cycles_for_edge[e]:
                d = self.cycle_colors.get(cyc)
                if d is None:
                    d = {}
                    self.cycle_colors[cyc] = d
                d[e] = c
                if len(d) == 4:
                    self.cycle_valid[cyc] = len(set(d.values())) == 4

        def remove(self, u, v):
            e = cano_edge(u, v)
            self.adj[u][v] = False
            self.adj[v][u] = False
            self.color[u][v] = -1
            self.color[v][u] = -1
            self.nbrs[u].discard(v)
            self.nbrs[v].discard(u)
            for i, ed in enumerate(self.edges):
                if cano_edge(ed[0], ed[1]) == e:
                    self.edges.pop(i)
                    break
            for cyc in cycles_for_edge[e]:
                d = self.cycle_colors.get(cyc)
                if d is not None:
                    d.pop(e, None)
                    if len(d) < 4:
                        self.cycle_valid.pop(cyc, None)

        def recolor(self, u, v, new_c):
            old = self.color[u][v]
            self.color[u][v] = new_c
            self.color[v][u] = new_c
            e = cano_edge(u, v)
            for i, ed in enumerate(self.edges):
                if cano_edge(ed[0], ed[1]) == e:
                    self.edges[i] = (ed[0], ed[1], new_c)
                    break
            for cyc in cycles_for_edge[e]:
                d = self.cycle_colors.get(cyc)
                if d is not None:
                    d[e] = new_c
                    if len(d) == 4:
                        self.cycle_valid[cyc] = len(set(d.values())) == 4

        def conflict_count(self):
            return sum(1 for ok in self.cycle_valid.values() if not ok)

    def valid_add_colors(st, u, v):
        """Colors for (u,v) that keep every 4-cycle rainbow, given (u,v) absent."""
        e = cano_edge(u, v)
        forbidden = 0
        for cyc in cycles_for_edge[e]:
            d = st.cycle_colors.get(cyc)
            if d is None:
                continue
            if len(d) == 4:
                # already a full cycle without (u,v)? impossible since e not in d
                pass
            # colors already on the other 3 edges of this cycle
            present = [d[ce] for ce in d if ce != e]
            if len(present) == 3:
                # the other 3 edges must be mutually distinct, else cycle can't be rainbow
                if len(set(present)) < 3:
                    return 0
                for cc in present:
                    forbidden |= (1 << cc)
            elif len(present) == 2:
                a, b = present
                if a == b:
                    return 0
                # third edge is (u,v) itself; the 4th edge of cycle must be absent
                # -> cycle not full, no constraint beyond distinctness with a,b
                forbidden |= (1 << a) | (1 << b)
            elif len(present) == 1:
                forbidden |= (1 << present[0])
        return ((1 << num_colors) - 1) & ~forbidden

    def valid_recolor(st, u, v, new_c):
        e = cano_edge(u, v)
        for cyc in cycles_for_edge[e]:
            d = st.cycle_colors.get(cyc)
            if d is None:
                continue
            if len(d) == 4:
                # cycle is full; replacing e's color must keep 4 distinct
                others = [d[ce] for ce in d if ce != e]
                if new_c in others or len(set(others)) < 3:
                    return False
        return True

    def run_greedy(seed, order='bipartite', prefer_formula=True):
        random.seed(seed)
        st = State()
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if order == 'bipartite':
            left = list(range(10))
            right = list(range(10, 20))
            bip = [(u, v) for u in left for v in right]
            intra = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if not (i < 10 and j >= 10)]
            random.shuffle(bip)
            random.shuffle(intra)
            all_edges = bip + intra
        else:
            random.shuffle(all_edges)
        for u, v in all_edges:
            if not time_ok():
                break
            valid = valid_add_colors(st, u, v)
            if valid:
                if prefer_formula:
                    pref = (u + v) % num_colors
                    if (valid >> pref) & 1:
                        c = pref
                    else:
                        c = random.choice([x for x in range(num_colors) if (valid >> x) & 1])
                else:
                    c = random.choice([x for x in range(num_colors) if (valid >> x) & 1])
                st.add(u, v, c)
        return st

    def local_search(st, seed):
        random.seed(seed)
        for _ in range(4000):
            if not time_ok():
                break
            # try to add a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if not st.adj[i][j]]
            if missing:
                u, v = random.choice(missing)
                valid = valid_add_colors(st, u, v)
                if valid:
                    c = random.choice([x for x in range(num_colors) if (valid >> x) & 1])
                    st.add(u, v, c)
                    continue
            # try a recolor
            if st.edges:
                eu, ev, ec = random.choice(st.edges)
                others = [c for c in range(num_colors) if c != ec]
                random.shuffle(others)
                for nc in others[:12]:
                    if valid_recolor(st, eu, ev, nc):
                        st.recolor(eu, ev, nc)
                        break
        return st

    def final_polish(st):
        # deterministic sweep: add every addable edge, repeat
        for _ in range(6):
            if not time_ok():
                break
            added = False
            for i in range(n):
                for j in range(i + 1, n):
                    if st.adj[i][j]:
                        continue
                    valid = valid_add_colors(st, i, j)
                    if valid:
                        c = (i + j) % num_colors
                        if not ((valid >> c) & 1):
                            c = random.choice([x for x in range(num_colors) if (valid >> x) & 1])
                        st.add(i, j, c)
                        added = True
            if not added:
                break
        return st

    iterations = 0
    orders = ['bipartite', 'random', 'random', 'bipartite']
    while time_ok():
        iterations += 1
        seed = iterations * 7 + 1
        order = orders[iterations % len(orders)]
        st = run_greedy(seed, order=order, prefer_formula=True)
        st = local_search(st, seed)
        st = final_polish(st)
        if st.conflict_count() == 0 and len(st.edges) > best_count:
            best_count = len(st.edges)
            best_edges = [[u, v, c] for (u, v, c) in st.edges]

    return best_edges
