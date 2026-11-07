"""Cochromatic partition via branch-and-bound set cover over maximal cliques/ind-sets."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _maximal_cliques(adj, n):
    result = []
    def bronk(R, P_set, X):
        if not P_set and not X:
            result.append(frozenset(R))
            return
        candidates = P_set | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = set()
        for v in list(candidates):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    bronk(frozenset(), set(range(n)), set())
    return result

def _maximal_independent_sets(adj, n):
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp_adj[u].add(v)
    return _maximal_cliques(comp_adj, n)

def _greedy_set_cover(universe, candidates):
    remaining = set(universe)
    parts = []
    while remaining:
        best_c = None
        best_overlap = -1
        for cs in candidates:
            overlap = len(cs & remaining)
            if overlap > best_overlap:
                best_overlap = overlap
                best_c = cs
        if best_c is None or best_overlap == 0:
            v = min(remaining)
            best_c = {v}
        parts.append(set(best_c))
        remaining -= best_c
    return parts

def construct():
    start = time.time()
    deadline = start + 85.0
    
    adj = _build_adj()
    n = P['n']
    
    max_cliques = _maximal_cliques(adj, n)
    max_indep = _maximal_independent_sets(adj, n)
    
    candidates = []
    for mc in max_cliques:
        candidates.append(set(mc))
    for mi in max_indep:
        candidates.append(set(mi))
    for v in range(n):
        s = {v}
        if s not in candidates:
            candidates.append(s)
    
    # Deduplicate
    unique_cands = list(set(frozenset(c) for c in candidates))
    candidates = [set(c) for c in unique_cands]
    
    # Greedy initial solution for upper bound
    greedy_parts = _greedy_set_cover(set(range(n)), candidates)
    ub = len(greedy_parts)
    
    uidx = {v: i for i, v in enumerate(range(n))}
    
    cands_with_sets = []
    for cs in candidates:
        mask = 0
        for v in cs:
            mask |= (1 << uidx[v])
        cands_with_sets.append((mask, cs))
    
    full_mask = (1 << n) - 1
    max_size = max(len(cs) for _, cs in cands_with_sets) if cands_with_sets else 1
    
    best_count = [ub]
    best_parts = [None]
    
    # Store best solution from greedy as initial
    best_parts[0] = greedy_parts
    
    def lb(remaining):
        if remaining == 0:
            return 0
        return (bin(remaining).count('1') + max_size - 1) // max_size
    
    def bnb(remaining, chosen, depth):
        if time.time() > deadline:
            return
        if remaining == 0:
            if depth < best_count[0]:
                best_count[0] = depth
                best_parts[0] = list(chosen)
            return
        if depth >= best_count[0]:
            return
        if depth + lb(remaining) >= best_count[0]:
            return
        
        rem_bits = [i for i in range(n) if (remaining >> i) & 1]
        if not rem_bits:
            return
        
        min_count = len(cands_with_sets) + 1
        chosen_elem = rem_bits[0]
        for i in rem_bits:
            cnt = 0
            for mask, cs in cands_with_sets:
                if (mask >> i) & 1:
                    cnt += 1
            if cnt < min_count:
                min_count = cnt
                chosen_elem = i
        
        covering = [(mask, cs) for mask, cs in cands_with_sets if (mask >> chosen_elem) & 1]
        if not covering:
            return
        
        covering.sort(key=lambda x: -len(x[1]))
        
        for mask, cs in covering:
            new_rem = remaining & ~mask
            chosen.append(cs)
            bnb(new_rem, chosen, depth + 1)
            chosen.pop()
            if time.time() > deadline:
                return
    
    bnb(full_mask, [], 0)
    
    if best_parts[0] is not None:
        result = []
        for part in best_parts[0]:
            result.append(list(sorted(part)))
        return result
    
    # Ultimate fallback
    return [[v] for v in range(n)]
