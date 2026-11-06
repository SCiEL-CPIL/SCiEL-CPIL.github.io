import time
import random
import math
import numpy as np

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 118  # Safety margin

    # ---- Quasi-random regular base: 20-regular circulant on 40 vertices ----
    # Each vertex i connects to i +/- k for k in 1..10  => 20 neighbors, 400 edges.
    def make_base():
        adj = [0] * n
        edges = set()
        for i in range(n):
            for k in range(1, 11):
                j = (i + k) % n
                a, b = min(i, j), max(i, j)
                if (a, b) not in edges:
                    edges.add((a, b))
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        return adj, edges

    def add_edge(adj, edges, u, v):
        u, v = min(u, v), max(u, v)
        if (u, v) in edges:
            return False
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, edges, u, v):
        u, v = min(u, v), max(u, v)
        if (u, v) not in edges:
            return False
        edges.discard((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True

    def build_books(adj, edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        return books

    def max_sum(books):
        if not books:
            return 0, 0
        return max(books.values()), sum(books.values())

    def local_search(adj, edges, seed, time_limit):
        rng = random.Random(seed)
        books = build_books(adj, edges)
        cur_max, cur_sum = max_sum(books)
        best_max, best_sum = cur_max, cur_sum
        best_edges = edges.copy()
        best_adj = adj[:]

        T = 3.0
        cooling = 0.9995
        min_T = 0.001
        stall = 0
        max_stall = 400
        end_time = time.time() + time_limit

        # Precompute non-edges list for the 4th vertex of a 4-cycle
        # We'll just sample candidates.

        while time.time() < end_time:
            if T <= min_T:
                break

            # Attempt a degree-preserving 4-cycle swap.
            # Pick a random edge (a,b); pick c in N(a)\{b}; pick d in N(b)\{a} with d != c,
            # such that (c,d) is an edge, and (a,d),(b,c) are non-edges.
            # Then swap: remove (a,c),(b,d)?? Let's define the 4-cycle properly.
            #
            # A 4-cycle: a - c - b - d - a  (edges a-c, c-b, b-d, d-a present)
            # Diagonals a-b and c-d absent.
            # Swap: remove a-c, b-d ; add a-b, c-d. Degrees preserved.
            #
            # So: choose edge (c,b) present. Choose a in N(c)\{b}, d in N(b)\{c},
            #     with a != d, (a,d) present, (a,b) absent, (c,d) absent.
            # Then remove (c,a) and (b,d), add (a,b) and (c,d).

            done = False
            for _ in range(40):  # try to find a valid 4-cycle
                c, b = rng.choice(list(books.keys()))
                # neighbors of c excluding b
                nc = adj[c] & ~(1 << b)
                if nc == 0:
                    continue
                a = (nc & -(nc)).bit_length() - 1  # lowest set bit
                # actually pick a random set bit
                bits = [x for x in range(n) if (nc >> x) & 1]
                a = rng.choice(bits)
                nb = adj[b] & ~(1 << c)
                bits_b = [x for x in range(n) if (nb >> x) & 1]
                if not bits_b:
                    continue
                d = rng.choice(bits_b)
                if d == a:
                    continue
                # need (a,d) edge, (a,b) non-edge, (c,d) non-edge
                if not (adj[a] & (1 << d)):
                    continue
                if (min(a, b), max(a, b)) in edges:
                    continue
                if (min(c, d), max(c, d)) in edges:
                    continue
                # valid 4-cycle: a-c, c-b, b-d, d-a
                done = True
                break

            if not done:
                T *= cooling
                continue

            # Backup
            adj_bak = adj[:]
            books_bak = books.copy()
            edges_bak = edges.copy()

            # Compute book changes from the swap:
            # Remove edges (c,a) and (b,d); add edges (a,b) and (c,d).
            # We'll just recompute books for the affected edges via direct formula,
            # but that's expensive. Instead do incremental update.
            #
            # Simpler and safe: recompute books for edges incident to {a,b,c,d}
            # plus the two new edges. Since only triangles involving these vertices
            # change. A triangle changes only if it uses at least one of the 4 changed edges.
            # Changed edges: (c,a) removed, (b,d) removed, (a,b) added, (c,d) added.
            # Any triangle containing one of these edges is affected.
            # Affected edges are those sharing a vertex with the changed edges, i.e.
            # edges incident to a, b, c, or d.

            # Remove (c,a)
            remove_edge(adj, edges, c, a)
            remove_edge(adj, edges, b, d)
            add_edge(adj, edges, a, b)
            add_edge(adj, edges, c, d)

            # Recompute books for all edges incident to a,b,c,d
            affected = set()
            for x in (a, b, c, d):
                bits = adj[x]
                while bits:
                    y = (bits & -bits).bit_length() - 1
                    bits &= bits - 1
                    affected.add((min(x, y), max(x, y)))
            # Also ensure the two new edges are present
            affected.add((min(a, b), max(a, b)))
            affected.add((min(c, d), max(c, d)))

            for e in affected:
                if e in edges:
                    u, v = e
                    books[e] = (adj[u] & adj[v]).bit_count()
                else:
                    books.pop(e, None)

            new_max, new_sum = max_sum(books)

            accept = False
            if new_max < cur_max:
                accept = True
                stall = 0
            elif new_max == cur_max and new_sum < cur_sum:
                accept = True
                stall = 0
            else:
                delta = (new_max - cur_max) * 10 + (new_sum - cur_sum)
                if delta < 0:
                    accept = True
                    stall = 0
                else:
                    if rng.random() < math.exp(-delta / T):
                        accept = True
                    else:
                        stall += 1
                        if stall > max_stall:
                            T = max(T * 0.5, min_T)
                            stall = 0

            if accept:
                cur_max, cur_sum = new_max, new_sum
                if cur_max < best_max or (cur_max == best_max and cur_sum < best_sum):
                    best_max, best_sum = cur_max, cur_sum
                    best_edges = edges.copy()
                    best_adj = adj[:]
            else:
                adj = adj_bak
                books = books_bak
                edges = edges_bak

            T *= cooling

            if best_max <= 7:
                break

        return best_max, best_sum, best_edges

    global_best = None
    global_best_max = 999
    global_best_sum = 999

    # Phase: multiple seeds, each base + a random extra edge, then local search.
    for seed in range(60):
        if time.time() > deadline - 4:
            break
        adj, edges = make_base()
        rng = random.Random(seed * 7919 + 13)
        # add one random edge
        for _ in range(50):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v and add_edge(adj, edges, u, v):
                break
        tm = max(0.8, (deadline - 4 - time.time()) / 60)
        bm, bs, be = local_search(adj, edges, seed, tm)
        if bm < global_best_max or (bm == global_best_max and bs < global_best_sum):
            global_best_max, global_best_sum = bm, bs
            global_best = be
        if global_best_max <= 7:
            break

    if global_best is None:
        # fallback: base + one edge
        adj, edges = make_base()
        add_edge(adj, edges, 0, 10)
        return [[u, v] for (u, v) in sorted(edges)]

    return [[u, v] for (u, v) in sorted(global_best)]
