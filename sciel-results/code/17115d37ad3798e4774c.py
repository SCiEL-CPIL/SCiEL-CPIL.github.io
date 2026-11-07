# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_candidates():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    cliques = []
    def gen_cliques(cand, cur):
        if cur:
            cliques.append(frozenset(cur))
        for i in range(len(cand)):
            v = cand[i]
            new_cand = [x for x in cand[i+1:] if v in adj[x]]
            gen_cliques(new_cand, cur + [v])
    gen_cliques(list(range(n)), [])

    non_adj = [set() for _ in range(n)]
    for i in range(n):
        non_adj[i] = set(range(n)) - adj[i] - {i}
    indep_sets = []
    def gen_is(cand, cur):
        if cur:
            indep_sets.append(frozenset(cur))
        for i in range(len(cand)):
            v = cand[i]
            new_cand = [x for x in cand[i+1:] if v in non_adj[x]]
            gen_is(new_cand, cur + [v])
    gen_is(list(range(n)), [])

    all_cands = list(set(cliques) | set(indep_sets))
    all_cands = [c for c in all_cands if len(c) >= 1]
    all_cands.sort(key=len, reverse=True)
    return all_cands

def _greedy_fallback():
    n = P['n']
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(a, b) in es for i, a in enumerate(trial) for b in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [[int(v) for v in p] for p in parts]

def _solve(candidates, n, deadline):
    full_mask = (1 << n) - 1
    cand_masks = []
    for c in candidates:
        m = 0
        for v in c:
            m |= (1 << v)
        cand_masks.append(m)
    
    # Remove duplicate masks
    cand_masks = list(set(cand_masks))
    
    # Filter to maximal candidates
    cand_masks_sorted = sorted(cand_masks, key=lambda m: bin(m).count('1'), reverse=True)
    filtered = []
    for cm in cand_masks_sorted:
        is_subset = False
        for fm in filtered:
            if (cm & fm) == cm:
                is_subset = True
                break
        if not is_subset:
            filtered.append(cm)
    cand_masks = filtered
    
    from collections import deque
    q = deque()
    q.append(full_mask)
    
    parent = {full_mask: None}
    
    while q:
        if time.time() > deadline:
            return None
            
        mask = q.popleft()
        
        for cm in cand_masks:
            if (cm & mask) == cm:
                new_mask = mask ^ cm
                if new_mask == 0:
                    # Found solution
                    parts = []
                    part = [v for v in range(n) if (cm >> v) & 1]
                    part.sort()
                    parts.append(part)
                    
                    cur = mask
                    while cur != full_mask:
                        prev, cm2 = parent[cur]
                        part2 = [v for v in range(n) if (cm2 >> v) & 1]
                        part2.sort()
                        parts.append(part2)
                        cur = prev
                    return parts
                else:
                    if new_mask not in parent:
                        parent[new_mask] = (mask, cm)
                        q.append(new_mask)
                        
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s kill

    candidates = _build_candidates()
    parts = _solve(candidates, P['n'], deadline)

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

# EVOLVE-BLOCK-END
