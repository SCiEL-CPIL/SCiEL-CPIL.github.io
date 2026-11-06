"""Rewritten program: guaranteed valid fallback plus a time-budgeted search for a larger C5-free graph with alpha < 9."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # The evaluator kills after ~150s wall-clock; keep a hard safety margin.
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---- Guaranteed valid fallback: 8 disjoint K4s -> n=32, alpha=8, no C5 ----
    def k4_blocks(nblocks):
        n = 4 * nblocks
        edges = []
        for b in range(nblocks):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    fallback = k4_blocks(8)  # n = 32

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5(adj, n, tlimit):
        # C5 exists iff some vertex is the start of a 4-vertex path whose last vertex
        # is adjacent to the first (with distinct vertices). Bound the search by time.
        t0 = time.time()
        for s in range(n):
            sbit = 1 << s
            stack = [(s, sbit, 0)]
            while stack:
                v, used, depth = stack.pop()
                if depth == 3:
                    if adj[v] & sbit:
                        return True
                else:
                    cand = adj[v] & ~used
                    while cand:
                        lb = cand & (-cand)
                        w = lb.bit_length() - 1
                        stack.append((w, used | lb, depth + 1))
                        cand ^= lb
            if time.time() - t0 > tlimit:
                return None  # unknown
        return False

    def alpha_is_at_least_9(adj, n, tlimit):
        # alpha(G) >= 9  <=>  complement has a clique of size 9.
        # Branch-and-bound with greedy upper bound on remaining vertices.
        comp = [((1 << n) - 1) ^ (adj[i] | (1 << i)) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda i: -bin(comp[i]).count('1'))
        t0 = time.time()

        def rec(cand, k):
            if k >= 9:
                return True
            if time.time() - t0 > tlimit:
                return None
            # upper bound: |cand| + k
            if bin(cand).count('1') + k < 9:
                return False
            # greedy upper bound on clique size within cand
            cb = cand
            gc = 0
            while cb:
                lb = cb & (-cb)
                gc += 1
                cb &= (cb - 1)
                cb &= comp[lb.bit_length() - 1]
            if gc + k < 9:
                return False
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            r = rec(cand & comp[v], k + 1)
            if r is True:
                return True
            if r is None:
                return None
            return rec(cand ^ lb, k)

        return rec((1 << n) - 1, 0)

    def is_valid(n, edges, tlimit):
        adj = build_adj(n, edges)
        c5 = has_c5(adj, n, tlimit)
        if c5 is True:
            return False
        if c5 is None:
            return False  # inconclusive -> treat as invalid to be safe
        a9 = alpha_is_at_least_9(adj, n, tlimit)
        if a9 is True:
            return False
        if a9 is None:
            return False  # inconclusive -> treat as invalid to be safe
        return True

    best = fallback  # always valid

    # ---- Search: Randomized search for n=33 ----
    # The 8xK4 extension is proven impossible. Use random graphs with C5 removal.
    if remaining() > 20.0:
        n = 33
        t0 = time.time()
        limit = min(remaining() - 10.0, 60.0)
        found = None
        for attempt in range(1000):
            if time.time() - t0 > limit:
                break
            # Generate random graph with p=0.4
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.4:
                        edges.append([u, v])
            
            # Remove C5s iteratively
            # We need to find a C5 and remove one of its edges.
            # has_c5 returns True/False/None. We need the actual C5.
            # For speed, we just check validity. If invalid, we try to repair?
            # Simpler: Just check validity. If valid, we are done.
            # But random graphs with p=0.4 are likely to have C5s and alpha >= 9.
            # This might be too slow or unlikely to succeed.
            
            # Alternative: Use a "blow-up" of a small C5-free graph.
            # Let's try a blow-up of the 5-cycle? No, C5 is not C5-free.
            # Let's try a blow-up of the 3-prism? (6 vertices, C5-free? No, has C4s and C6s? No, 3-prism is 3x2. Cycles: 4s and 6s. No C5s.)
            # 3-Prism (6 vertices) is C5-free.
            # Alpha(3-Prism) = 3 (one from each square? No, independent set size 3? e.g. top triangle? No, top is triangle. Max independent set is 3: one from each vertical edge pair? No.
            # 3-Prism: vertices (0,1,2) top, (3,4,5) bottom. Edges: top triangle, bottom triangle, verticals.
            # Independent set: cannot pick adjacent.
            # Pick 0, 4, 2? 0-2 is edge.
            # Pick 0, 4. 0 is not adj to 4. 4 is not adj to 2? 4 is adj to 1, 3, 5. 2 is adj to 0, 1. So 0, 4, 2 is independent?
            # 0-2 is edge in top triangle. So no.
            # Max independent set of 3-prism is 3?
            # If we blow up 3-prism by factor k, n=6k.
            # For n=33, k=5.5. Not integer.
            
            # Let's stick to the random search but with a smarter check.
            ok = is_valid(n, edges, min(remaining() - 15.0, 2.0))
            if ok:
                found = {'n': n, 'edges': edges}
                break
        
        if found is not None:
            best = found

    # ---- Search: try to grow to n=34,35 via random C5-free graphs ----
    # (only if we still have a comfortable time budget)
    for target in (34, 35):
        if remaining() < 25.0:
            break
        # Build a random graph with edge probability tuned so it is likely C5-free
        # and dense enough that alpha < 9. Then verify exactly.
        t0 = time.time()
        limit = min(remaining() - 15.0, 20.0)
        found = None
        for _ in range(2000):
            if time.time() - t0 > limit:
                break
            # Random graph, then remove edges to break C5s is complex; instead use
            # a "blow-up of K4 with random cross edges" heuristic:
            # Partition into 8 blocks of ~target/8 vertices, make each block a clique,
            # and add random cross edges, then verify.
            n = target
            block_size = n // 8
            edges = []
            # clique within each block
            for b in range(8):
                lo = b * block_size
                hi = lo + block_size
                for i in range(lo, hi):
                    for j in range(i + 1, hi):
                        edges.append([i, j])
            # random cross edges
            for b in range(8):
                for c in range(b + 1, 8):
                    lo1 = b * block_size
                    hi1 = lo1 + block_size
                    lo2 = c * block_size
                    hi2 = lo2 + block_size
                    for i in range(lo1, hi1):
                        for j in range(lo2, hi2):
                            if random.random() < 0.5:
                                edges.append([i, j])
            ok = is_valid(n, edges, min(remaining() - 20.0, 5.0))
            if ok:
                found = {'n': n, 'edges': edges}
                break
        if found is not None:
            best = found
            break

    # Final safety: ensure we return before the hard deadline.
    if time.time() > deadline - 2.0:
        return fallback

    return best
# EVOLVE-BLOCK-END
