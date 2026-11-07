"""Rewritten solver for the cochromatic partition task."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
FULL = (1 << N) - 1


def _build_adj():
    adj = [0] * N
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _precompute(adj):
    n = N
    valid = [False] * (1 << n)
    valid[0] = True
    max_clique = [0] * (1 << n)
    max_clique[0] = 0
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        nb = adj[v]
        cm = rest & nb
        im = rest & ~nb
        valid[mask] = (cm == rest) or (im == rest)
        max_clique[mask] = 1 + max_clique[cm]
    # maximal valid masks: a valid mask that is not a proper subset of any valid mask
    valid_masks = [m for m in range(1, 1 << n) if valid[m]]
    maximal = [True] * len(valid_masks)
    for i in range(len(valid_masks)):
        mi = valid_masks[i]
        for j in range(len(valid_masks)):
            if i == j:
                continue
            mj = valid_masks[j]
            if mi != mj and (mi & mj) == mi:
                maximal[i] = False
                break
    maximal_valid = [valid_masks[i] for i in range(len(valid_masks)) if maximal[i]]
    # for each vertex, the maximal valid masks containing it, largest first
    by_vertex = [[] for _ in range(n)]
    for m in maximal_valid:
        mm = m
        while mm:
            lsb = mm & (-mm)
            v = lsb.bit_length() - 1
            by_vertex[v].append(m)
            mm ^= lsb
    for v in range(n):
        by_vertex[v].sort(key=lambda m: m.bit_count(), reverse=True)
    return valid, max_clique, by_vertex


def _solve(adj, valid, max_clique, by_vertex):
    n = N
    full = FULL
    deadline = time.time() + 58.0
    nodes = [0]
    state = {'best_k': n, 'best_masks': None}

    def dfs(remaining, k_used, path):
        nodes[0] += 1
        if time.time() > deadline or nodes[0] > 3000000:
            return False
        if remaining == 0:
            if k_used < state['best_k']:
                state['best_k'] = k_used
                state['best_masks'] = list(path)
            return True
        if k_used >= state['best_k']:
            return False
        # clique-size lower bound on the number of parts still needed
        omega = max_clique[remaining]
        if omega == 0:
            return False
        need = (remaining.bit_count() + omega - 1) // omega
        if k_used + need >= state['best_k']:
            return False
        # deterministic branching vertex: lowest-index free vertex
        v = (remaining & (-remaining)).bit_length() - 1
        for m in by_vertex[v]:
            if (m & remaining) != m:
                continue
            path.append(m)
            dfs(remaining ^ m, k_used + 1, path)
            path.pop()
        return False

    for target in range(1, n + 1):
        state['best_k'] = target
        state['best_masks'] = None
        nodes[0] = 0
        dfs(full, 0, [])
        if state['best_masks'] is not None:
            return state['best_masks']
    return None


def _greedy(adj):
    n = N
    edges = {(u, v) for u, v in P['edges']} | {(v, u) for u, v in P['edges']}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            ok = True
            for u in part:
                e = (u, v) in edges
                if e and part == []:
                    continue
            # check all pairs
            trial = part + [v]
            allc = True
            alli = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    if (trial[i], trial[j]) in edges:
                        alli = False
                    else:
                        allc = False
            if allc or alli:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts


def construct():
    adj = _build_adj()
    valid, max_clique, by_vertex = _precompute(adj)
    masks = _solve(adj, valid, max_clique, by_vertex)
    if masks is None:
        parts = _greedy(adj)
    else:
        parts = []
        for m in masks:
            verts = []
            mm = m
            while mm:
                lsb = mm & (-mm)
                verts.append(lsb.bit_length() - 1)
                mm ^= lsb
            parts.append([int(v) for v in verts])
    return parts

# EVOLVE-BLOCK-END
