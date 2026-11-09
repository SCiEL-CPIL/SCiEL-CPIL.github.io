import time
import sys
from collections import defaultdict

sys.setrecursionlimit(1000000)


def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_longest_cycle(adj, max_nodes=500000):
        """Find the longest simple cycle using iterative deepening with node budget."""
        n = len(adj)
        if n < 3:
            return None
        # Start vertices sorted by degree descending
        starts = sorted(adj.keys(), key=lambda v: -len(adj[v]))
        if not starts:
            return None
        # Try each start with a budget
        budget_per_start = max(5000, max_nodes // max(1, len(starts)))
        best_cycle = None
        best_len = 0
        for start in starts[:min(12, len(starts))]:
            if time.time() - t0 > DEADLINE:
                break
            # Iterative deepening: try to find cycle of length L
            # Use DFS with path tracking
            on_path = {start}
            path = [start]
            best_local = None
            best_local_len = 0
            nodes_visited = 0
            budget = budget_per_start

            def dfs(u, depth):
                nonlocal nodes_visited, best_local, best_local_len, budget
                if nodes_visited > budget:
                    return
                nbrs = sorted(adj[u], key=lambda x: -len(adj[x]))
                for w in nbrs:
                    nodes_visited += 1
                    if nodes_visited > budget:
                        return
                    if w == start and depth >= 2:
                        # Found a cycle of length depth+1
                        if depth + 1 > best_local_len:
                            best_local_len = depth + 1
                            best_local = list(path)
                    elif w not in on_path:
                        on_path.add(w)
                        path.append(w)
                        dfs(w, depth + 1)
                        path.pop()
                        on_path.discard(w)
                        if nodes_visited > budget:
                            return

            dfs(start, 0)
            if best_local is not None and best_local_len > best_len:
                best_len = best_local_len
                best_cycle = best_local
        return best_cycle

    def find_cycle_greedy(adj):
        """Fast greedy: find any cycle via DFS back-edge, then extend it."""
        color = {}
        parent = {}
        found_cycle = None

        def dfs(u):
            nonlocal found_cycle
            if found_cycle is not None:
                return
            color[u] = 1
            for w in adj[u]:
                if found_cycle is not None:
                    return
                if w == parent.get(u):
                    continue
                if color.get(w, 0) == 0:
                    parent[w] = u
                    dfs(w)
                elif color.get(w, 0) == 1:
                    cyc = []
                    cur = u
                    while cur != w:
                        cyc.append(cur)
                        cur = parent[cur]
                    cyc.append(w)
                    if len(cyc) >= 3:
                        found_cycle = cyc
                        return
            color[u] = 2

        for v in list(adj.keys()):
            if color.get(v, 0) == 0:
                dfs(v)
                if found_cycle is not None:
                    break
        return found_cycle

    def try_extend_cycle(cycle, adj, max_steps=200000):
        """Try to extend a cycle by inserting vertices."""
        if cycle is None:
            return None
        cycle_set = set(cycle)
        # Try to insert each non-cycle vertex into the cycle
        best = list(cycle)
        best_set = set(cycle)
        improved = True
        while improved:
            improved = False
            # Try each edge in the cycle, see if we can insert a vertex
            for i in range(len(best)):
                a = best[i]
                b = best[(i + 1) % len(best)]
                # Find common neighbors of a and b not in best
                common = adj[a] & adj[b] - best_set
                for w in common:
                    # Check if inserting w gives a valid cycle (it does by construction)
                    new_cycle = best[:i + 1] + [w] + best[i + 1:]
                    # Verify all edges exist
                    valid = True
                    for j in range(len(new_cycle)):
                        x, y = new_cycle[j], new_cycle[(j + 1) % len(new_cycle)]
                        if y not in adj[x]:
                            valid = False
                            break
                    if valid:
                        best = new_cycle
                        best_set.add(w)
                        improved = True
                        break
                if improved:
                    break
        return best

    def local_search_2opt(pieces, adj_original, edges_set):
        """Try to reduce piece count by merging cycles and absorbing edges."""
        # pieces: list of cycles (list of vertices, len>=3) and edges ([u,v], len=2)
        # Try to merge pairs of cycles that share a vertex
        changed = True
        while changed:
            changed = False
            new_pieces = []
            i = 0
            while i < len(pieces):
                p = pieces[i]
                if len(p) < 3:
                    new_pieces.append(p)
                    i += 1
                    continue
                # Try to merge with subsequent cycles
                merged = False
                for j in range(i + 1, len(pieces)):
                    q = pieces[j]
                    if len(q) < 3:
                        continue
                    # Check if they share a vertex
                    shared = set(p) & set(q)
                    if not shared:
                        continue
                    v = next(iter(shared))
                    # Try to splice: p = ...v... and q = ...v...
                    # Find positions of v in p and q
                    idx_p = p.index(v)
                    idx_q = q.index(v)
                    # Splice: take p from v around, then q from v around
                    # p: v, p[idx_p+1], ..., p[idx_p-1] (cycle back to v)
                    # q: v, q[idx_q+1], ..., q[idx_q-1]
                    # Merged: v, q[idx_q+1], ..., q[idx_q-1], p[idx_p+1], ..., p[idx_p-1]
                    # This requires edge q[idx_q-1] - p[idx_p+1] to exist
                    p_next = p[(idx_p + 1) % len(p)]
                    p_prev = p[(idx_p - 1) % len(p)]
                    q_next = q[(idx_q + 1) % len(q)]
                    q_prev = q[(idx_q - 1) % len(q)]
                    # Option 1: q_prev - p_next edge
                    if p_next in adj_original[q_prev]:
                        merged_cycle = [v] + q[(idx_q + 1):(idx_q + len(q))] + q[:idx_q] + p[(idx_p + 1):(idx_p + len(p))] + p[:idx_p]
                        # Remove duplicates (shouldn't be any since only shared vertex is v)
                        # Verify
                        valid = True
                        for k in range(len(merged_cycle)):
                            x, y = merged_cycle[k], merged_cycle[(k + 1) % len(merged_cycle)]
                            if y not in adj_original[x]:
                                valid = False
                                break
                        if valid and len(merged_cycle) >= 3:
                            # Check all vertices distinct
                            if len(set(merged_cycle)) == len(merged_cycle):
                                new_pieces.append(merged_cycle)
                                # Skip q
                                pieces[j] = None
                                merged = True
                                changed = True
                                break
                    # Option 2: q_next - p_prev edge
                    if p_prev in adj_original[q_next]:
                        merged_cycle = [v] + p[(idx_p + 1):(idx_p + len(p))] + p[:idx_p] + q[(idx_q + 1):(idx_q + len(q))] + q[:idx_q]
                        valid = True
                        for k in range(len(merged_cycle)):
                            x, y = merged_cycle[k], merged_cycle[(k + 1) % len(merged_cycle)]
                            if y not in adj_original[x]:
                                valid = False
                                break
                        if valid and len(merged_cycle) >= 3:
                            if len(set(merged_cycle)) == len(merged_cycle):
                                new_pieces.append(merged_cycle)
                                pieces[j] = None
                                merged = True
                                changed = True
                                break
                if not merged:
                    new_pieces.append(p)
                i += 1
            # Remove None entries
            pieces = [p for p in new_pieces if p is not None]
        # Try to absorb single edges into cycles
        changed = True
        while changed:
            changed = False
            new_pieces = []
            for p in pieces:
                if len(p) >= 3:
                    p_set = set(p)
                    # Try to extend this cycle
                    extended = try_extend_cycle(p, adj_original, max_steps=50000)
                    if extended is not None and len(extended) > len(p):
                        new_pieces.append(extended)
                        # Remove absorbed edges from remaining single edges
                        old_edges = set()
                        for k in range(len(p)):
                            x, y = p[k], p[(k + 1) % len(p)]
                            old_edges.add((min(x, y), max(x, y)))
                        new_edges = set()
                        for k in range(len(extended)):
                            x, y = extended[k], extended[(k + 1) % len(extended)]
                            new_edges.add((min(x, y), max(x, y)))
                        absorbed = new_edges - old_edges
                        # Remove absorbed single-edge pieces
                        pieces = [q for q in new_pieces if not (len(q) == 2 and (min(q[0], q[1]), max(q[0], q[1])) in absorbed)]
                        # Re-add extended cycle
                        pieces.append(extended)
                        changed = True
                        break
                    else:
                        new_pieces.append(p)
                else:
                    new_pieces.append(p)
            else:
                pieces = new_pieces
        return pieces

    def decompose(edges):
        adj = build_adj(edges)
        remaining = set(edges)
        pieces = []

        while time.time() - t0 < DEADLINE:
            if not remaining:
                break
            # Find longest cycle
            cycle = find_longest_cycle(adj)
            if cycle is None:
                cycle = find_cycle_greedy(adj)
            if cycle is None:
                break
            # Try to extend
            cycle = try_extend_cycle(cycle, adj, max_steps=100000)
            if cycle is None:
                cycle = find_cycle_greedy(adj)
                if cycle is None:
                    break
            # Remove cycle edges
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
            pieces.append(list(cycle))

        # Remaining edges as single pieces
        for e in remaining:
            pieces.append([e[0], e[1]])

        # Local search
        adj_full = build_adj(edges)
        pieces = local_search_2opt(pieces, adj_full, set(edges))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        results.append(decompose(edges))

    return results
