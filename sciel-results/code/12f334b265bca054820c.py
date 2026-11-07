# EVOLVE-BLOCK-START
import time
from functools import lru_cache
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
        # Choose pivot from P_set ∪ X to minimize branching
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

def _solve(candidates, n, deadline):
    full_mask = (1 << n) - 1
    cand_masks = []
    for c in candidates:
        m = 0
        for v in c:
            m |= (1 << v)
        cand_masks.append(m)

    num_cands = len(cand_masks)
    timed_out = [False]

    # Precompute: for each mask, which candidate indices are subsets
    subset_cands = [[] for _ in range(1 << n)]
    for mask in range(1 << n):
        valid = []
        for i, cm in enumerate(cand_masks):
            if (cm & mask) == cm:
                valid.append(i)
        subset_cands[mask] = valid

    # Lower bound: minimum number of parts needed
    # Each part covers at most max_candidate_size vertices from remaining
    max_cand_size = max(len(c) for c in candidates) if candidates else 1

    @lru_cache(maxsize=None)
    def min_parts(rem_mask):
        if rem_mask == 0:
            return 0
        if timed_out[0]:
            return 100
        if time.time() > deadline:
            timed_out[0] = True
            return 100

        # Lower bound: popcount / max_cand_size
        popcount = bin(rem_mask).count('1')
        lb = (popcount + max_cand_size - 1) // max_cand_size

        best_add = 100
        for i in subset_cands[rem_mask]:
            cm = cand_masks[i]
            r = rem_mask ^ cm
            add = 1 + min_parts(r)
            if add < best_add:
                best_add = add
                if best_add <= lb:
                    break
        return best_add

    result = min_parts(full_mask)

    if timed_out[0]:
        return None

    # Reconstruct the partition
    def reconstruct(rem_mask):
        if rem_mask == 0:
            return []
        target = min_parts(rem_mask)
        for i in subset_cands[rem_mask]:
            cm = cand_masks[i]
            r = rem_mask ^ cm
            if 1 + min_parts(r) == target:
                rest = reconstruct(r)
                part = sorted([v for v in range(n) if (cm >> v) & 1])
                return [part] + rest
        return []

    parts = reconstruct(full_mask)
    return parts

def _milp_solve(candidates, n, deadline):
    """Try to solve via MILP. Returns list of parts or None."""
    t0 = time.time()
    if time.time() - t0 > deadline:
        return None
    
    num_cands = len(candidates)
    if num_cands == 0:
        return None
    
    # Build candidate masks
    cand_masks = []
    for c in candidates:
        m = 0
        for v in c:
            m |= (1 << v)
        cand_masks.append(m)
    
    # Variables: x[i][v] = 1 if vertex v is assigned to candidate i
    # We want to minimize sum_i (max over v of x[i][v]) which is equivalent to
    # introducing y[i] >= x[i][v] for all v, minimize sum y[i]
    # Total variables: n * num_cands (x) + num_cands (y)
    
    # Actually, let's use a simpler formulation:
    # x[i][v] binary: vertex v is in part i
    # y[i] binary: part i is used
    # Constraints:
    # 1. sum_i x[i][v] = 1 for all v (each vertex in exactly one part)
    # 2. x[i][v] <= y[i] for all i, v (if vertex in part, part is used)
    # 3. For each candidate i, the set of vertices assigned to it must be subset of candidate i
    #    i.e., if x[i][v] = 1 then v must be in candidate i
    #    This is handled by only allowing x[i][v] = 1 if v is in candidate i
    
    # For constraint 3, we can just set x[i][v] = 0 if v not in candidate i
    # So we only have variables x[i][v] where v is in candidate i
    
    # Let's build the variable index mapping
    var_idx = {}  # (i, v) -> variable index
    var_list = []
    for i in range(num_cands):
        for v in candidates[i]:
            var_idx[(i, v)] = len(var_list)
            var_list.append((i, v))
    
    num_x = len(var_list)
    num_y = num_cands
    total_vars = num_x + num_y
    
    # Objective: minimize sum y[i]
    c_obj = np.zeros(total_vars)
    for i in range(num_cands):
        c_obj[num_x + i] = 1.0
    
    # Constraints:
    # 1. For each vertex v: sum over i where v in candidate i of x[i][v] = 1
    # 2. For each (i,v): x[i][v] - y[i] <= 0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # Vertex assignment constraints
    for v in range(n):
        row = np.zeros(total_vars)
        for (i, vv) in var_list:
            if vv == v:
                row[var_idx[(i, v)]] = 1.0
        if row.sum() > 0:
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)
    
    # x[i][v] <= y[i] constraints
    for (i, v) in var_list:
        row = np.zeros(total_vars)
        row[var_idx[(i, v)]] = 1.0
        row[num_x + i] = -1.0
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(0.0)
    
    if not A_rows:
        return None
    
    A = np.array(A_rows)
    lb = np.array(b_lower)
    ub = np.array(b_upper)
    
    # All variables are binary
    integrality = np.ones(total_vars)
    
    # Bounds: all in [0, 1]
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    constraint = LinearConstraint(A, lb, ub)
    
    try:
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
    except Exception:
        return None
    
    if res.status != 0:  # 0 means optimal
        return None
    
    # Extract solution
    x_sol = res.x[:num_x]
    y_sol = res.x[num_x:]
    
    # Build parts from x_sol
    parts = []
    for i in range(num_cands):
        part_vertices = []
        for (ii, v) in var_list:
            if ii == i and x_sol[var_idx[(i, v)]] > 0.5:
                part_vertices.append(v)
        if part_vertices:
            parts.append(sorted(part_vertices))
    
    # Verify it's a valid partition
    all_used = []
    for part in parts:
        all_used.extend(part)
    
    if len(all_used) != n or len(set(all_used)) != n:
        return None
    
    return parts

def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before 90s kill

    candidates = _build_candidates()
    
    # First try MILP
    milp_parts = _milp_solve(candidates, P['n'], deadline)
    
    # Then try DP
    dp_parts = _solve(candidates, P['n'], deadline)
    
    # Pick the better solution (fewer parts)
    parts = None
    if milp_parts and dp_parts:
        if len(milp_parts) <= len(dp_parts):
            parts = milp_parts
        else:
            parts = dp_parts
    elif milp_parts:
        parts = milp_parts
    elif dp_parts:
        parts = dp_parts
    
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
