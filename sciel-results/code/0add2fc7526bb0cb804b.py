import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _canonical_key(cyc):
    """Canonical form of a cycle (as frozenset of edge keys)."""
    keys = []
    L = len(cyc)
    for i in range(L):
        a, b = cyc[i], cyc[(i + 1) % L]
        keys.append(_edge_key(a, b))
    return frozenset(keys)


def _precompute_all_hamiltonian_cycles(adj, deadline):
    """Enumerate all Hamiltonian cycles up-front using DFS."""
    n = len(adj)
    all_cycles = []
    seen_canonical = set()
    adj_sorted = [sorted(adj[i]) for i in range(n)]

    def dfs(path, visited):
        if time.time() > deadline:
            return
        if len(path) == n:
            last = path[-1]
            if 0 in adj[last]:
                # found a Hamiltonian cycle
                keys = set()
                ok = True
                for i in range(n):
                    a, b = path[i], path[(i + 1) % n]
                    k = _edge_key(a, b)
                    if k in keys:
                        ok = False
                        break
                    keys.add(k)
                if ok:
                    ck = frozenset(keys)
                    if ck not in seen_canonical:
                        seen_canonical.add(ck)
                        all_cycles.append(list(path))
            return
        u = path[-1]
        for w in adj_sorted[u]:
            if not visited[w]:
                visited[w] = True
                path.append(w)
                dfs(path, visited)
                path.pop()
                visited[w] = False

    visited = [False] * n
    visited[0] = True
    dfs([0], visited)
    return all_cycles


def _verify_cycles(cycles, full_edges):
    used = set()
    for cyc in cycles:
        if len(cyc) < 3:
            return False
        if len(set(cyc)) != len(cyc):
            return False
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            k = _edge_key(a, b)
            if k not in full_edges:
                return False
            if k in used:
                return False
            used.add(k)
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = _build_adjacency()

    # Precompute all Hamiltonian cycles (fast: ~26k cycles, takes < 1s)
    all_cycles = _precompute_all_hamiltonian_cycles(adj, deadline)
    if not all_cycles:
        # Fallback: return a single cycle if we found nothing
        return [[int(v) for v in range(n)]]

    # Precompute edge sets for each cycle
    cycle_edge_sets = []
    for cyc in all_cycles:
        keys = set()
        L = len(cyc)
        for i in range(L):
            a, b = cyc[i], cyc[(i + 1) % L]
            keys.add(_edge_key(a, b))
        cycle_edge_sets.append(keys)

    # Greedy selection with soft scoring
    def greedy_select(rng, max_cycles=5):
        remaining = set(range(len(all_cycles)))
        selected = []
        used_edges = set()
        for _ in range(max_cycles):
            best_idx = -1
            best_score = -1
            for idx in remaining:
                edges = cycle_edge_sets[idx]
                if edges & used_edges:
                    continue
                # soft score: prefer cycles that "spread out" edges
                # Count how many of this cycle's edges are adjacent to used edges
                # (i.e., share a vertex with used edges) - lower is better
                # Also prefer cycles with more "new" edges
                score = len(edges)  # all same (21)
                # Penalty: edges that share a vertex with used edges
                penalty = 0
                for (a, b) in edges:
                    for (ca, cb) in used_edges:
                        if a in (ca, cb) or b in (ca, cb):
                            penalty += 1
                            break
                score = len(edges) - 0.01 * penalty + rng.random() * 0.001
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(all_cycles[best_idx])
            used_edges |= cycle_edge_sets[best_idx]
            remaining.discard(best_idx)
        return selected

    # Local search polish: try to swap one selected cycle for an unselected one
    def local_polish(selected, rng, max_iters=50):
        used_edges = set()
        for cyc in selected:
            used_edges |= cycle_edge_sets[selected.index(cyc)] if False else set()
        # Recompute properly
        sel_indices = []
        for cyc in selected:
            for i, c in enumerate(all_cycles):
                if c == cyc:
                    sel_indices.append(i)
                    break
        used_edges = set()
        for i in sel_indices:
            used_edges |= cycle_edge_sets[i]

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for si, idx in enumerate(sel_indices):
                # Try removing idx and adding a different one
                new_used = set()
                for j in sel_indices:
                    if j != idx:
                        new_used |= cycle_edge_sets[j]
                # Find best replacement
                best_alt = -1
                best_alt_score = -1
                for alt in range(len(all_cycles)):
                    if alt in sel_indices:
                        continue
                    if cycle_edge_sets[alt] & new_used:
                        continue
                    score = rng.random() * 0.001
                    if score > best_alt_score:
                        best_alt_score = score
                        best_alt = alt
                if best_alt != -1:
                    sel_indices[si] = best_alt
                    used_edges = set()
                    for j in sel_indices:
                        used_edges |= cycle_edge_sets[j]
                    improved = True
                    break
            if not improved:
                break
        return [all_cycles[i] for i in sel_indices]

    # Run multiple greedy attempts with different seeds
    best_result = []
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        result = greedy_select(rng, max_cycles=5)
        if len(result) > len(best_result):
            best_result = result
            # Polish
            polished = local_polish(result, rng, max_iters=20)
            if len(polished) > len(best_result):
                best_result = polished
        if len(best_result) >= 5:
            break

    # Final verification
    if not best_result or not _verify_cycles(best_result, full_edges):
        # Fallback: just return one cycle
        best_result = [all_cycles[0]]

    return [[int(v) for v in cyc] for cyc in best_result]
