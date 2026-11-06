"""C4-free graph without independent 11-set (R(C4,K11) > n)."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    DEADLINE = start + 140.0  # safety margin under 150s

    # ---------- Fallback: 10 disjoint triangles (n=30, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b],[a,c],[b,c]]
        return {'n': 30, 'edges': edges}

    # ---------- Helpers ----------
    def has_indep_11(n, adj):
        """Return True if an independent set of size 11 exists (exact branch&bound)."""
        from functools import lru_cache
        import sys
        sys.setrecursionlimit(10000)
        # Work on the candidate set; use degree-based pruning.
        # Order vertices by degree descending for better pruning.
        degs = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        # Map to a candidate bitmask of size n
        # We use a list of candidate vertices (the "frontier").
        # Branch and bound on independence: pick vertex, include or exclude.
        # Use the fact: if len(candidates) < 11 -> False.
        # Prune: if we can pick 11 vertices all pairwise non-adjacent -> True.
        # Standard: independent set search via branching on a vertex v:
        #   include v -> recurse on candidates \ (N(v) ∪ {v}), need 10 more
        #   exclude v -> recurse on candidates \ {v}, need 11 more
        # Strong pruning: if the maximum possible independent set in candidates
        # is < needed -> False. Use a greedy upper bound: sort candidates by
        # degree-in-candidates ascending; the independence number is at most
        # the size of a maximal independent set found greedily? No, that's lower.
        # Use: alpha(G) <= n - matching? Hard. Use simple bound: alpha <= n.
        # Better: if needed > len(candidates): return False.
        # We'll implement a straightforward but pruned search.

        def _search(cands, need):
            # cands: tuple/list of candidate vertex ids
            if need <= 0:
                return True
            if len(cands) < need:
                return False
            # Upper bound: even if all cands were independent, need <= len(cands) (checked).
            # Try to find an independent set of size 'need' in cands.
            # Pick a vertex v with high degree within cands for strong pruning.
            # Compute degrees within cands.
            cand_set = set(cands)
            # Choose v in cands maximizing |N(v) ∩ cands|
            best_v = None
            best_deg = -1
            for v in cands:
                d = 0
                for u in adj[v]:
                    if u in cand_set:
                        d += 1
                        if d > best_deg:
                            best_deg = d
                            best_v = v
                            if d >= 10:  # early stop: strong pruning vertex
                                break
                if d > best_deg:
                    best_deg = d
                    best_v = v
            v = best_v
            # Include v: remove v and its neighbors from cands
            nbrs = adj[v]
            new_cands_inc = [x for x in cands if x != v and x not in nbrs]
            if _search(tuple(new_cands_inc), need - 1):
                return True
            # Exclude v
            new_cands_exc = [x for x in cands if x != v]
            if _search(tuple(new_cands_exc), need):
                return True
            return False

        cands = tuple(range(n))
        return _search(cands, 11)

    def is_c4_free(n, adj):
        """Check no two vertices share >=2 common neighbors."""
        # For each pair of vertices, count common neighbors.
        # Efficient: for each vertex, look at pairs of its neighbors.
        from collections import defaultdict
        pair_count = defaultdict(int)
        for v in range(n):
            nbrs = adj[v]
            for i in range(len(nbrs)):
                for j in range(i+1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    if a > b:
                        a, b = b, a
                    pair_count[(a,b)] += 1
                    if pair_count[(a,b)] >= 2:
                        return False
        return True

    def build_edges(n, adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # ---------- Random C4-free graph generation ----------
    def try_random(n, seed, time_limit):
        rng = random.Random(seed)
        # Start empty, add edges while preserving C4-free.
        adj = [set() for _ in range(n)]
        # Precompute all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(all_edges)
        t0 = time.time()
        for (u, v) in all_edges:
            if time.time() > time_limit:
                break
            # Check if adding (u,v) creates a C4.
            # C4 exists iff |N(u) ∩ N(v)| >= 2.
            common = adj[u] & adj[v]
            if len(common) >= 2:
                continue
            # Also, adding (u,v) must not create C4 involving u and some w,
            # where w is a neighbor of v... Actually the C4 condition is:
            # after adding (u,v), for every pair of vertices, common neighbors < 2.
            # The only new pairs that could get a new common neighbor are
            # (u, x) where x is a neighbor of v (x != u), and (v, x) where x is neighbor of u.
            # For (u, x): common neighbors = (N(u) ∩ N(x)) ∪ {v if v in N(x)}.
            # Since x is neighbor of v, v is in N(x). So common(u,x) gains v.
            # We need |N(u) ∩ N(x)| + (1 if v in N(x) else 0) < 2, i.e., |N(u) ∩ N(x)| < 2 (since v in N(x)).
            # So we need |N(u) ∩ N(x)| <= 0 for all x in N(v), x != u? No:
            # common(u,x) after add = (N(u) ∩ N(x)) ∪ ({v} if v ∈ N(x)).
            # v ∈ N(x) since x ∈ N(v). So common = (N(u)∩N(x)) ∪ {v}.
            # |common| = |N(u)∩N(x)| + 1 (since v not in N(u) before add, and v not in N(x)∩N(u) unless v∈N(u)).
            # v not in N(u) before add. So v is new. |common| = |N(u)∩N(x)| + 1.
            # Need < 2 => |N(u)∩N(x)| <= 0 => N(u) ∩ N(x) = empty for all x ∈ N(v)\{u}.
            # Similarly for (v, x) with x ∈ N(u)\{v}: need N(v) ∩ N(x) = empty.
            # This is expensive to check for all x. Let's just do the full check for safety
            # but only on affected pairs.
            ok = True
            # Check pairs (u, x) for x in N(v), x != u
            for x in adj[v]:
                if x == u:
                    continue
                # common(u,x) after add includes v and any common neighbor in N(u)∩N(x)
                cnt = 1  # for v
                for y in adj[u]:
                    if y != v and y in adj[x]:
                        cnt += 1
                        if cnt >= 2:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                for x in adj[u]:
                    if x == v:
                        continue
                    cnt = 1  # for u
                    for y in adj[v]:
                        if y != u and y in adj[x]:
                            cnt += 1
                            if cnt >= 2:
                                ok = False
                                break
                    if not ok:
                        break
            if ok:
                adj[u].add(v)
                adj[v].add(u)
        # Verify C4-free
        if not is_c4_free(n, adj):
            return None
        # Check independent set
        if has_indep_11(n, adj):
            return None
        edges = build_edges(n, adj)
        return {'n': n, 'edges': edges}

    # ---------- Search from high n down ----------
    best = None
    for n in range(43, 29, -1):
        if time.time() > DEADLINE - 5:
            break
        # Try several seeds for this n
        seeds = 0
        while time.time() < DEADLINE - 5:
            result = try_random(n, seeds, time.time())
            if result is not None:
                best = result
                break
            seeds += 1
            if seeds >= 200:  # cap per n
                break
        if best is not None:
            break

    if best is None:
        best = fallback()
    return best
