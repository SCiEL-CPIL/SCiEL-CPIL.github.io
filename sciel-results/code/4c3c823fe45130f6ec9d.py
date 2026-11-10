import random
import time
from collections import Counter

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all distinct triples (c, a, b) with c < a < b
    such that 1/c = 1/a + 1/b and a, b, c <= N.
    Returns a list of tuples (c, a, b).
    """
    triples = []
    for c in range(1, N + 1):
        c_sq = c * c
        for u in range(1, c + 1):
            if c_sq % u != 0:
                continue
            v = c_sq // u
            a = c + u
            b = c + v
            if a <= N and b <= N:
                if a != b and a != c and b != c:
                    triples.append((c, a, b))
    return triples

def _build_conflict_graph(triples, N=180):
    """
    Build conflict graph: edge between x,y if there exists z such that
    {x,y,z} forms a forbidden triple.
    Returns adjacency list.
    """
    conflicts = {i: set() for i in range(1, N + 1)}
    for (c, a, b) in triples:
        conflicts[c].add(a)
        conflicts[c].add(b)
        conflicts[a].add(b)
    return conflicts

def _greedy_min_degree(conflicts, N=180, seed=42):
    """
    Greedy: repeatedly add element with fewest conflicts to current set.
    Returns a valid subset.
    """
    rng = random.Random(seed)
    in_set = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # Maintain dynamic conflict counts
    remaining_conflicts = {i: len(conflicts[i]) for i in range(1, N + 1)}
    
    while candidates:
        # Find candidate with minimum remaining conflicts
        min_conf = min(remaining_conflicts.values())
        # Among those with min conflicts, pick random for diversity
        min_cands = [x for x in candidates if remaining_conflicts[x] == min_conf]
        x = rng.choice(min_cands)
        
        # Check if x conflicts with any element already in set
        if not (conflicts[x] & in_set):
            in_set.add(x)
            # Remove x from candidates and update conflict counts
            candidates.remove(x)
            for y in conflicts[x]:
                if y in candidates:
                    remaining_conflicts[y] -= 1
    
    return in_set

def _greedy_max_independence(triples, N=180, seed=42):
    """
    Alternative greedy: start with all elements, remove those in most triples.
    """
    rng = random.Random(seed)
    S = set(range(1, N + 1))
    elem_to_triples = {i: set() for i in range(1, N + 1)}
    for i, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].add(i)
    
    active_triples = set(range(len(triples)))
    
    while active_triples:
        # Pick a random active triple
        t_idx = rng.choice(list(active_triples))
        t = triples[t_idx]
        
        # Remove the element in t with highest degree
        best_elem = max(t, key=lambda x: len(elem_to_triples[x] & active_triples))
        S.discard(best_elem)
        
        # Remove all triples containing best_elem
        to_remove = list(elem_to_triples[best_elem] & active_triples)
        for tri_idx in to_remove:
            active_triples.discard(tri_idx)
    
    return S

def _local_search_add_back(S, removed, triples, elem_to_triples, deadline):
    """
    Try to add back removed elements if they don't create new triples.
    """
    S = set(S)
    removed = set(removed)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in list(removed):
            if time.time() >= deadline:
                break
            # Check if adding x creates any triple
            can_add = True
            for tri_idx in elem_to_triples[x]:
                t = triples[tri_idx]
                others = [y for y in t if y != x]
                if others[0] in S and others[1] in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                removed.discard(x)
                changed = True
    return S, removed

def _local_search_swap(S, removed, triples, elem_to_triples, deadline, max_swaps=100):
    """
    Try 1-swaps: remove y from S, add x from removed if it helps.
    """
    S = set(S)
    removed = set(removed)
    swaps_made = 0
    
    for _ in range(max_swaps):
        if time.time() >= deadline:
            break
        if not removed:
            break
        
        # Try random swap
        x = random.choice(list(removed))
        y = random.choice(list(S))
        
        # Check if swapping y->x is valid
        new_S = S - {y} | {x}
        valid = True
        for t in triples:
            if t[0] in new_S and t[1] in new_S and t[2] in new_S:
                valid = False
                break
        
        if valid:
            S = new_S
            removed.discard(x)
            removed.add(y)
            swaps_made += 1
    
    return S, removed

def _is_valid_subset(S, triples):
    """Return True if S contains no complete precomputed triple."""
    S = set(S)
    for t in triples:
        if t[0] in S and t[1] in S and t[2] in S:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    N = 180
    
    triples = _precompute_triples(N)
    conflicts = _build_conflict_graph(triples, N)
    
    elem_to_triples = {i: set() for i in range(1, N + 1)}
    for i, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].add(i)
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Multiple greedy min-degree with different seeds
    for seed in range(20):
        if time.time() >= deadline:
            break
        S = _greedy_min_degree(conflicts, N, seed)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Greedy max independence with different seeds
    for seed in range(10):
        if time.time() >= deadline:
            break
        S = _greedy_max_independence(triples, N, seed)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search on best set
    if time.time() < deadline:
        removed = set(range(1, N + 1)) - best_set
        best_set, removed = _local_search_add_back(best_set, removed, triples, elem_to_triples, deadline)
        best_size = len(best_set)
    
    # More local search with swaps
    if time.time() < deadline - 10:
        removed = set(range(1, N + 1)) - best_set
        best_set, removed = _local_search_swap(best_set, removed, triples, elem_to_triples, deadline - 5, max_swaps=200)
        best_size = len(best_set)
        
        # Add back after swaps
        removed = set(range(1, N + 1)) - best_set
        best_set, removed = _local_search_add_back(best_set, removed, triples, elem_to_triples, deadline)
        best_size = len(best_set)
    
    # Final verification
    if not _is_valid_subset(best_set, triples):
        # Fallback: simple safe set (odd numbers)
        best_set = set(range(1, N + 1, 2))
    
    return list(best_set)

# EVOLVE-BLOCK-END
