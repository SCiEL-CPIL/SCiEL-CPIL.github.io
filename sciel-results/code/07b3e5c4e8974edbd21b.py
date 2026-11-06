"""Stronger decomposition: ILP max edge-disjoint cycle packing + greedy local search."""
import time
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE = 120.0
SAFETY = 18.0
START = time.time()


def _time_left():
    return DEADLINE - SAFETY - (time.time() - START)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _cycles_from_ilp(edges, n, budget):
    """Exact max edge-disjoint cycle packing via integer programming (HiGHS)."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        return None
    if budget <= 2:
        return None
    m = len(edges)
    if m < 6:
        return None

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cycles = []
    for u in range(n):
        for v in sorted(adj[u]):
            if u >= v:
                continue
            # find simple paths from v back to u using DFS (bounded)
            stack = [(v, [v], {v})]
            while stack:
                node, path, seen = stack.pop()
                if node == u and len(path) >= 3:
                    cycles.append(path)
                    continue
                for w in sorted(adj[node]):
                    if w not in seen:
                        stack.append((w, path + [w], seen | {w}))

    if not cycles:
        return None

    # cap number of cycles to keep the LP tractable
    max_cycles = 3000
    if len(cycles) > max_cycles:
        random.Random(0).shuffle(cycles)
        cycles = cycles[:max_cycles]

    K = len(cycles)
    # incidence matrix: rows = edges, cols = cycles
    inc = np.zeros((m, K), dtype=np.int8)
    for j, cyc in enumerate(cycles):
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i + 1) % len(cyc)]
            ei = edges.index((min(a, b), max(a, b)))
            inc[ei, j] = 1

    c = np.ones(K, dtype=float)  # maximize number of cycles
    A = inc.astype(np.float64)
    cons = LinearConstraint(A, ub=np.ones(m))
    integrality = np.ones(K)
    bounds = Bounds(0.0, 1.0)

    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
               options={"time_limit": budget, "mip_rel_gap": 0.0,
                        "presolve": True, "thread_count": 4})
    if res.status not in (0, 2) or res.x is None:
        return None
    sel = [j for j in range(K) if res.x[j] > 0.5]
    return [cycles[j] for j in sel]


def _greedy_decompose(edges, n, budget):
    """Greedy longest-cycle extraction + edge-flip local search to maximize covered edges."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # ---- Phase A: greedy longest cycle extraction ----
    cycles = []
    used = set()
    remaining = set(edges)

    def longest_cycle(start):
        # iterative DFS for longest simple path that can close to start
        best = None
        best_len = 0
        stack = [(start, [start], {start})]
        while stack:
            node, path, seen = stack.pop()
            if len(path) > best_len:
                if start in adj[node] and len(path) >= 3:
                    e = (min(start, node), max(start, node))
                    if e in remaining and e not in used:
                        best = list(path)
                        best_len = len(path)
            for w in sorted(adj[node]):
                if w == start:
                    if len(path) >= 3:
                        e = (min(start, node), max(start, node))
                        if e in remaining and e not in used and len(path) > best_len:
                            best = list(path)
                            best_len = len(path)
                    continue
                if w not in seen:
                    stack.append((w, path + [w], seen | {w}))
        return best

    while remaining and _time_left() > 1:
        # candidate starts: vertices with degree >= 2 in remaining
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        cands = [v for v in deg if deg[v] >= 2]
        random.shuffle(cands)
        cands = cands[:min(len(cands), 25)]
        best_cycle = None
        best_len = 0
        for s in cands:
            if _time_left() <= 1:
                break
            cyc = longest_cycle(s)
            if cyc and len(cyc) > best_len:
                best_len = len(cyc)
                best_cycle = cyc
        if best_cycle:
            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(a, b), max(a, b))
                used.add(e)
                remaining.discard(e)
        else:
            break

    # ---- Phase B: local search to increase total covered edges ----
    # covered edges = union of cycle edges
    def covered_edges(cycs):
        s = set()
        for cyc in cycs:
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                s.add((min(a, b), max(a, b)))
        return s

    cur_covered = covered_edges(cycles)
    cur_score = len(cur_covered)  # maximize edges covered by cycles

    # build candidate cycles from remaining edges for insertion
    # (simple 3-cycles and 4-cycles from remaining)
    rem_adj = defaultdict(set)
    for u, v in remaining:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    insert_cands = []
    for u in rem_adj:
        for v in sorted(rem_adj[u]):
            if u >= v:
                continue
            # 3-cycle u-v-w
            for w in sorted(rem_adj[u] & rem_adj[v]):
                if w != u and w != v:
                    insert_cands.append([u, v, w])
    # dedupe
    seen_cyc = set()
    uniq = []
    for cyc in insert_cands:
        key = frozenset(tuple(sorted(cyc)))
        if key not in seen_cyc:
            seen_cyc.add(key)
            uniq.append(cyc)
    insert_cands = uniq

    improved = True
    while improved and _time_left() > 2:
        improved = False
        # try inserting a candidate cycle (if edge-disjoint)
        random.shuffle(insert_cands)
        for cyc in insert_cands:
            if _time_left() <= 2:
                break
            ce = covered_edges([cyc])
            if ce & cur_covered:
                continue
            new_score = cur_score + len(ce)
            if new_score > cur_score:
                cycles.append(cyc)
                cur_covered |= ce
                cur_score = new_score
                improved = True
                break
        if improved:
            continue
        # try dropping a cycle that uses few unique edges (free up edges)
        # not needed for pure increase; skip to save time

    # ---- Build pieces: cycles + leftover single edges ----
    pieces = [list(c) for c in cycles]
    for e in remaining:
        if e not in cur_covered:
            pieces.append(list(e))
    return pieces


def _verify(pieces, edges):
    all_edges = set(tuple(sorted(e)) for e in edges)
    covered = set()
    for piece in pieces:
        if len(piece) == 2:
            covered.add((min(piece[0], piece[1]), max(piece[0], piece[1])))
        else:
            if len(set(piece)) < 3:
                return False
            for i in range(len(piece)):
                u = piece[i]
                v = piece[(i + 1) % len(piece)]
                covered.add((min(u, v), max(u, v)))
    return covered == all_edges


def _solve_one(spec):
    n, p, seed = spec
    edges = _instance(n, p, seed)
    if not edges:
        return []
    n = max(v for e in edges for v in e) + 1
    # try ILP if enough time
    ilp_budget = _time_left() * 0.4
    ilp_cycles = None
    if ilp_budget > 3:
        ilp_cycles = _cycles_from_ilp(edges, n, ilp_budget)
    if ilp_cycles is not None:
        # build pieces from ILP cycles + leftover edges
        cyc_edges = set()
        for cyc in ilp_cycles:
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                cyc_edges.add((min(a, b), max(a, b)))
        pieces = [list(c) for c in ilp_cycles]
        for e in edges:
            if e not in cyc_edges:
                pieces.append(list(e))
        if _verify(pieces, edges):
            return pieces
    # fallback: greedy + local search
    pieces = _greedy_decompose(edges, n, _time_left())
    if _verify(pieces, edges):
        return pieces
    # ultimate fallback: all single edges
    return [list(e) for e in edges]


def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if _time_left() <= 1:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        results.append(_solve_one(spec))
    return results
