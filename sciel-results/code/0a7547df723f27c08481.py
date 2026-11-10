"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_forbidden(N=180):
    """
    Precompute for each a in [1, N], the set of pairs (b, c) with b < c
    such that 1/a = 1/b + 1/c, i.e., (b-a)(c-a) = a^2.
    b, c must be in [1, N] and distinct from a and each other.
    """
    forbidden_pairs = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(a2 ** 0.5) + 1):
            if a2 % d != 0:
                continue
            for dd in (d, a2 // d):
                if dd == 0:
                    continue
                # Case 1: d > 0
                b1 = a + dd
                c1 = a + a2 // dd
                if b1 == c1: continue
                if 1 <= b1 <= N and 1 <= c1 <= N and b1 != a and c1 != a:
                    if b1 < c1:
                        forbidden_pairs[a].add((b1, c1))
                    else:
                        forbidden_pairs[a].add((c1, b1))
                
                # Case 2: d < 0
                b2 = a - dd
                c2 = a - a2 // dd
                if b2 == c2: continue
                if 1 <= b2 <= N and 1 <= c2 <= N and b2 != a and c2 != a:
                    if b2 < c2:
                        forbidden_pairs[a].add((b2, c2))
                    else:
                        forbidden_pairs[a].add((c2, b2))
    
    return forbidden_pairs

def _can_add(S_set, elem, forbidden_pairs):
    """
    Check if adding `elem` to `S_set` (a set) creates any forbidden triple.
    """
    for (b, c) in forbidden_pairs[elem]:
        if b in S_set and c in S_set:
            return False
    
    for a in S_set:
        for (b, c) in forbidden_pairs[a]:
            if b == elem and c in S_set:
                return False
            if c == elem and b in S_set:
                return False
                
    return True

def _build_conflict_graph(S_set, forbidden_pairs):
    """
    Build a conflict graph on S_set where edges represent forbidden triples.
    Returns: dict mapping each element to set of elements it conflicts with,
             and list of all conflicting triples.
    """
    conflict_adj = defaultdict(set)
    conflicting_triples = []
    
    S_list = sorted(S_set)
    S_set_local = set(S_list)
    
    for a in S_list:
        for (b, c) in forbidden_pairs[a]:
            if b in S_set_local and c in S_set_local:
                # Triple (a, b, c) is a conflict
                conflicting_triples.append((a, b, c))
                conflict_adj[a].add(b)
                conflict_adj[a].add(c)
                conflict_adj[b].add(a)
                conflict_adj[c].add(a)
                conflict_adj[b].add(c)
                conflict_adj[c].add(b)
    
    return conflict_adj, conflicting_triples

def _subtractive_greedy(S_set, forbidden_pairs, deadline):
    """
    Subtractive greedy: repeatedly remove the element that appears in the most
    conflicting triples until no conflicts remain.
    """
    S = set(S_set)
    
    while time.time() < deadline:
        conflict_adj, conflicting_triples = _build_conflict_graph(S, forbidden_pairs)
        
        if not conflicting_triples:
            break
        
        # Count how many conflicts each element is involved in
        conflict_count = defaultdict(int)
        for (a, b, c) in conflicting_triples:
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
        
        # Remove the element with the highest conflict count
        # Tie-break by smaller value (remove smaller numbers first to keep larger ones)
        max_conflicts = max(conflict_count.values())
        candidates_to_remove = [x for x, cnt in conflict_count.items() if cnt == max_conflicts]
        elem_to_remove = min(candidates_to_remove)
        
        S.remove(elem_to_remove)
    
    return S

def _additive_greedy(S_set, candidates, forbidden_pairs, deadline):
    """
    Additive greedy: try to add candidates in order, keeping those that don't create conflicts.
    """
    S = set(S_set)
    for elem in candidates:
        if time.time() > deadline:
            break
        if _can_add(S, elem, forbidden_pairs):
            S.add(elem)
    return S

def _local_search_add(S_set, all_elements, forbidden_pairs, deadline, rng):
    """
    Try to add any missing element that doesn't create conflicts.
    """
    S = set(S_set)
    missing = list(all_elements - S)
    rng.shuffle(missing)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in missing:
            if time.time() > deadline:
                break
            if _can_add(S, elem, forbidden_pairs):
                S.add(elem)
                missing.remove(elem)
                improved = True
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    # Precompute forbidden pairs
    forbidden_pairs = _precompute_forbidden(N)
    
    all_elements = set(range(1, N + 1))
    rng = random.Random(42)
    
    # Strategy 1: Subtractive greedy from full set
    # Start with all elements, remove the most problematic ones
    S_sub = _subtractive_greedy(set(range(1, N + 1)), forbidden_pairs, deadline - 5)
    
    # Then try to add back elements
    S_sub = _local_search_add(S_sub, all_elements, forbidden_pairs, deadline - 3, rng)
    
    # Strategy 2: Additive greedy from upper half
    S_add = set(range(91, N + 1))
    candidates = list(range(89, 0, -2))
    S_add = _additive_greedy(S_add, candidates, forbidden_pairs, deadline - 5)
    S_add = _local_search_add(S_add, all_elements, forbidden_pairs, deadline - 3, rng)
    
    # Strategy 3: Hybrid - start with upper half, add some lower, then subtractive repair
    S_hybrid = set(range(91, N + 1))
    # Add odd numbers from 1-90
    for a in range(89, 0, -2):
        if time.time() > deadline - 10:
            break
        if _can_add(S_hybrid, a, forbidden_pairs):
            S_hybrid.add(a)
    
    # Apply subtractive repair to hybrid
    S_hybrid = _subtractive_greedy(S_hybrid, forbidden_pairs, deadline - 5)
    S_hybrid = _local_search_add(S_hybrid, all_elements, forbidden_pairs, deadline - 3, rng)
    
    # Pick the best result
    best = max([S_sub, S_add, S_hybrid], key=len)
    
    # Final local search on the best
    best = _local_search_add(best, all_elements, forbidden_pairs, deadline - 2, rng)
    
    return list(best)

# EVOLVE-BLOCK-END
