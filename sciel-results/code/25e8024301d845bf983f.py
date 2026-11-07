"""Rewritten program: clique partition of split graphs via bitset-accelerated greedy + local improvement."""
import time

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        # Bitset for uncovered clique-part edges: bit (i,j) for i<j in [0,k)
        # Use a set of frozensets for simplicity since k<=30
        # Actually, use a 2D boolean array via list of ints (bitmasks)
        # adj_mask[i] = bitmask of vertices j in [0,k) such that edge (i,j) is uncovered
        adj_mask = [0] * k
        for (a, b) in edge_set:
            if a < k and b < k:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)

        # Cross-edge tracking: for each independent vertex v, which clique-part neighbors are uncovered
        cross_uncovered = {}
        for v in range(k, n):
            cross_uncovered[v] = set(adj[v])  # neighbors in [0,k)

        cliques = []

        independent = list(range(k, n))
        independent.sort(key=lambda v: -len(adj[v]))

        def _find_max_clique_bitset(candidates_mask, adj_mask_local):
            """Find a maximal clique in the graph defined by adj_mask_local restricted to candidates_mask."""
            best = 0
            cand = candidates_mask
            while cand:
                # Pick vertex with max degree within cand
                best_v = -1
                best_deg = -1
                c = cand
                while c:
                    lb = c & (-c)
                    v = lb.bit_length() - 1
                    deg = bin(adj_mask_local[v] & cand).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    c ^= lb
                if best_v < 0:
                    break
                best |= (1 << best_v)
                # Intersect cand with neighbors of best_v
                cand &= adj_mask_local[best_v]
            return best

        for v in independent:
            if _deadline_hit():
                for u in list(cross_uncovered[v]):
                    cliques.append([v, u])
                    cross_uncovered[v].discard(u)
                    # Remove clique-part edge (u, w) if both in some set? No, just cross edge.
                    # Actually we need to also remove the clique-part edge if we form {v,u}
                    # {v,u} only covers edge (v,u), no clique-part edges.
                break

            remaining = set(cross_uncovered[v])
            while remaining:
                if _deadline_hit():
                    for u in list(remaining):
                        cliques.append([v, u])
                        remaining.discard(u)
                    break

                # Find a large clique S ⊆ remaining such that all edges within S are uncovered
                # Build bitmask of remaining
                rem_mask = 0
                for u in remaining:
                    rem_mask |= (1 << u)

                # Find maximal clique in uncovered-clique-part graph restricted to rem_mask
                S_mask = _find_max_clique_bitset(rem_mask, adj_mask)

                if S_mask == 0:
                    # Pick any single vertex
                    u = next(iter(remaining))
                    S_mask = 1 << u

                # Convert to list
                S = []
                m = S_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    S.append(u)
                    m ^= lb

                clique = [v] + S
                cliques.append(clique)

                # Remove covered edges
                for u in S:
                    cross_uncovered[v].discard(u)
                    remaining.discard(u)
                for i in range(len(S)):
                    a = S[i]
                    for j in range(i + 1, len(S)):
                        b = S[j]
                        # Remove clique-part edge (a,b)
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)

        # Cover remaining clique-part edges using bitset
        while True:
            if _deadline_hit():
                # Fallback: 2-cliques for all remaining
                for a in range(k):
                    m = adj_mask[a]
                    while m:
                        lb = m & (-m)
                        b = lb.bit_length() - 1
                        if a < b:
                            cliques.append([a, b])
                            adj_mask[a] &= ~(1 << b)
                            adj_mask[b] &= ~(1 << a)
                        m ^= lb
                break

            # Find vertex with max degree in remaining graph
            deg = [0] * k
            for a in range(k):
                deg[a] = bin(adj_mask[a]).count('1')
            start = max(range(k), key=lambda x: deg[x])
            if deg[start] == 0:
                break

            # Greedy clique from start using bitset
            S_mask = 1 << start
            cand = adj_mask[start]
            while cand:
                # Pick vertex with max degree within cand
                best_v = -1
                best_deg = -1
                c = cand
                while c:
                    lb = c & (-c)
                    v = lb.bit_length() - 1
                    d = bin(adj_mask[v] & cand).count('1')
                    if d > best_deg:
                        best_deg = d
                        best_v = v
                    c ^= lb
                if best_v < 0:
                    break
                S_mask |= (1 << best_v)
                cand &= adj_mask[best_v]

            S = []
            m = S_mask
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                S.append(u)
                m ^= lb

            cliques.append(S)
            # Remove covered edges
            for i in range(len(S)):
                a = S[i]
                for j in range(i+1, len(S)):
                    b = S[j]
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)

        # Local improvement: merge cliques if union is still a clique
        # For efficiency, only try merging cliques that share the same independent vertex
        # or are both pure clique-part cliques.
        # Build a set of all edges for quick lookup
        all_edges = set(edge_set)

        improved = True
        for _ in range(3):
            if _deadline_hit():
                break
            improved = False
            for i in range(len(cliques)):
                for j in range(i+1, len(cliques)):
                    c1 = set(cliques[i])
                    c2 = set(cliques[j])
                    # Quick check: if both contain independent vertices from different sets, skip
                    ind1 = [x for x in c1 if x >= k]
                    ind2 = [x for x in c2 if x >= k]
                    if ind1 and ind2:
                        continue  # Two different independent vertices can't be in same clique
                    union = c1 | c2
                    # Check if union is a clique
                    is_clique = True
                    for a in union:
                        for b in union:
                            if a >= b:
                                continue
                            if (a, b) not in all_edges:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        cliques[i] = sorted(union)
                        del cliques[j]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        # Validate and fix: ensure all edges covered, no double coverage
        seen = set()
        valid_cliques = []
        for c in cliques:
            c = list(dict.fromkeys(c))  # dedupe, preserve order
            if len(c) < 2:
                continue
            ok = True
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    if e in seen:
                        ok = False
                        break
                    if e not in edge_set:
                        ok = False
                        break
                    seen.add(e)
                if not ok:
                    break
            if ok:
                valid_cliques.append([int(x) for x in c])

        # Add any missing edges as 2-cliques
        for e in edge_set:
            if e not in seen:
                valid_cliques.append([int(e[0]), int(e[1])])
                seen.add(e)

        return valid_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
