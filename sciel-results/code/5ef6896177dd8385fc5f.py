import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before the 120s/150s kill

    n = 18

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~lsb):
                return False
            c ^= lsb
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_triangles(adj):
        """Return list of triangle bitmasks (each vertex appears once)."""
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles

    def evaluate(adj):
        """
        Largest triangle-free induced subgraph size == n - (min triangle transversal).
        A triangle transversal hits every triangle. We find the min hitting set over
        the triangle hypergraph using bitmasks + branch and bound (exact).
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n

        # Deduplicate (shouldn't be needed, but safe)
        uniq = []
        seen = set()
        for t in triangles:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        triangles = uniq
        num_tris = len(triangles)

        # vertex_hits[v]: bitmask of triangles that contain v
        vertex_hits = [0] * n
        for i, tri in enumerate(triangles):
            bits = tri
            while bits:
                lsb = bits & -bits
                v = lsb.bit_length() - 1
                vertex_hits[v] |= (1 << i)
                bits ^= lsb

        all_tris = (1 << num_tris) - 1

        # Greedy hitting set -> gives an initial upper bound on transversal size
        def greedy_ub():
            cov = all_tris
            cnt = 0
            while cov:
                best_v = -1
                best_gain = -1
                for v in range(n):
                    g = cov & vertex_hits[v]
                    if g:
                        bc = g.bit_count()
                        if bc > best_gain:
                            best_gain = bc
                            best_v = v
                if best_v < 0:
                    break
                cov &= ~vertex_hits[best_v]
                cnt += 1
            return cnt

        best = [greedy_ub()]

        # Lower bound: max number of vertex-disjoint triangles (each needs >=1 vertex)
        def disjoint_lb(remaining):
            # greedy maximal packing of disjoint triangles
            pack = 0
            used = 0
            for tri in triangles:
                if (tri & used) == 0 and (tri & remaining) == tri:
                    pack += 1
                    used |= tri
            return pack

        # Branch and bound over vertices (ordered by usefulness)
        order = list(range(n))
        order.sort(key=lambda v: -vertex_hits[v].bit_count())

        def bt(idx, chosen, cov):
            if cov == 0:
                if chosen < best[0]:
                    best[0] = chosen
                return
            if chosen >= best[0]:
                return
            if idx >= n:
                return
            # remaining triangles must be covered by remaining vertices
            rem_cov = 0
            for j in range(idx, n):
                rem_cov |= vertex_hits[order[j]]
            if (cov & ~rem_cov) != 0:
                return  # cannot cover all remaining triangles
            # lower bound: disjoint triangles among those still uncovered
            if chosen + disjoint_lb(cov) >= best[0]:
                return
            v = order[idx]
            hv = vertex_hits[v]
            # branch: take v
            if hv & cov:
                bt(idx + 1, chosen + 1, cov & ~hv)
            # branch: skip v
            bt(idx + 1, chosen, cov)

        bt(0, 0, all_tris)
        return n - best[0]

    # ---------- initial candidates ----------
    def init_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def init_turan4():
        # T_4(18): four parts of sizes 5,5,4,4, all cross-edges present.
        adj = [0] * n
        sizes = [5, 5, 4, 4]
        parts = []
        start = 0
        for s in sizes:
            parts.append(list(range(start, start + s)))
            start += s
        for i in range(4):
            for j in range(i + 1, 4):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def generate_random_regular_like(d):
        adj = [0] * n
        degrees = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if degrees[u] < d and degrees[v] < d:
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    degrees[u] += 1
                    degrees[v] += 1
        return adj

    best_adj = None
    best_score = n + 1

    candidates = []
    candidates.append(init_k666())
    candidates.append(init_turan4())
    for _ in range(8):
        candidates.append(generate_random_k4_free())
    for d in [4, 5, 6, 7]:
        candidates.append(generate_random_regular_like(d))

    for cand in candidates:
        s = evaluate(cand)
        if s < best_score:
            best_score = s
            best_adj = cand[:]

    current_adj = best_adj[:]
    current_score = best_score

    # ---------- local search (simulated annealing over single-edge moves) ----------
    t = 1.5
    cooling = 0.998
    min_t = 0.01

    def try_move_add():
        nonlocal current_score, best_score, best_adj
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or (current_adj[u] & (1 << v)):
            return
        if not can_add_edge(current_adj, u, v):
            return
        old_u = current_adj[u]
        old_v = current_adj[v]
        add_edge(current_adj, u, v)
        new_score = evaluate(current_adj)
        delta = new_score - current_score
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            current_adj[u] = old_u
            current_adj[v] = old_v

    def try_move_remove():
        nonlocal current_score, best_score, best_adj
        u = random.randint(0, n - 1)
        v_mask = current_adj[u]
        if not v_mask:
            return
        tmp = v_mask
        bits = []
        while tmp:
            b = (tmp & -tmp).bit_length() - 1
            bits.append(b)
            tmp &= tmp - 1
        v = random.choice(bits)
        old_u = current_adj[u]
        old_v = current_adj[v]
        remove_edge(current_adj, u, v)
        new_score = evaluate(current_adj)
        delta = new_score - current_score
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            current_adj[u] = old_u
            current_adj[v] = old_v

    while time.time() < deadline and t > min_t:
        try_move_add()
        if random.random() < 0.15:
            try_move_remove()
        t *= cooling

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges
