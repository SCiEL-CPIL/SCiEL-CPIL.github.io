# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_candidates():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Bron-Kerbosch with pivot to enumerate all maximal cliques
    cliques = []
    def bronk(R, P_set, X):
        if not P_set and not X:
            if R:
                cliques.append(frozenset(R))
            return
        union = P_set | X
        if union:
            pivot = max(union, key=lambda v: len(adj[v] & P_set))
            candidates = [v for v in P_set if v not in adj[pivot]]
        else:
            candidates = list(P_set)
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bronk(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}

    bronk(frozenset(), frozenset(range(n)), frozenset())

    # All maximal independent sets = maximal cliques of complement
    non_adj = [set() for _ in range(n)]
    for i in range(n):
        non_adj[i] = set(range(n)) - adj[i] - {i}

    indep_sets = []
    def bronk_is(R, P_set, X):
        if not P_set and not X:
            if R:
                indep_sets.append(frozenset(R))
            return
        union = P_set | X
        if union:
            pivot = max(union, key=lambda v: len(non_adj[v] & P_set))
            candidates = [v for v in P_set if v not in non_adj[pivot]]
        else:
            candidates = list(P_set)
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & non_adj[v]
            new_X = X & non_adj[v]
            bronk_is(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}

    bronk_is(frozenset(), frozenset(range(n)), frozenset())

    # Union of all maximal cliques and maximal independent sets
    all_cands = list(set(cliques) | set(indep_sets))
    all_cands = [c for c in all_cands if len(c) >= 1]

    # Remove dominated candidates: if A ⊆ B, A is never needed
    all_cands.sort(key=len, reverse=True)
    non_dominated = []
    for c in all_cands:
        dominated = False
        for d in non_dominated:
            if c <= d:
                dominated = True
                break
        if not dominated:
            non_dominated.append(c)
    all_cands = non_dominated
    all_cands.sort(key=len, reverse=True)
    return all_cands

def _solve_milp(candidates, n, deadline):
    """Solve cochromatic partition via MILP using scipy.optimize.milp."""
    t0 = time.time()
    num_cands = len(candidates)
    if num_cands == 0:
        return None

    # Decision variables: x[i] = 1 if candidate i is selected
    # Objective: minimize sum(x[i])
    c = np.ones(num_cands)

    # Constraint 1: Each vertex must be covered exactly once
    # For each vertex v: sum of x[i] for candidates i containing v = 1
    A_rows = []
    b_lower = []
    b_upper = []

    for v in range(n):
        row = np.zeros(num_cands)
        for i, cand in enumerate(candidates):
            if v in cand:
                row[i] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)

    if not A_rows:
        return None

    A = np.array(A_rows)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)

    constraints = [
        LinearConstraint(A, b_lower, b_upper)
    ]

    integrality = np.ones(num_cands)  # all binary
    bounds = Bounds(0, 1)

    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        return None

    if time.time() - t0 > deadline:
        return None

    if not res.success:
        return None

    # Extract solution
    x = res.x
    parts = []
    for i, xi in enumerate(x):
        if xi > 0.5:
            parts.append(sorted([int(v) for v in candidates[i]]))

    if not parts:
        return None

    # Verify each vertex appears exactly once
    all_used = []
    for part in parts:
        all_used.extend(part)
    if len(all_used) != n or len(set(all_used)) != n:
        return None

    return parts

def construct():
    t0 = time.time()
    deadline = 75.0  # safety margin before 90s kill

    candidates = _build_candidates()
    parts = _solve_milp(candidates, P['n'], deadline)

    if parts is None or not parts:
        parts = _greedy_fallback()

    # Verify
    n = P['n']
    all_used = []
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    valid = True
    for part in parts:
        all_used.extend(part)
        adj_pairs = 0
        total_pairs = len(part) * (len(part) - 1) // 2
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if (part[i], part[j]) in es:
                    adj_pairs += 1
        if not (adj_pairs == 0 or adj_pairs == total_pairs):
            valid = False
            break

    if not valid or len(all_used) != n or len(set(all_used)) != n:
        parts = _greedy_fallback()

    result = [list(sorted([int(v) for v in part])) for part in parts]
    return result

def _greedy_fallback():
    n = P['n']
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return [[int(v) for v in p] for p in parts]

# EVOLVE-BLOCK-END
