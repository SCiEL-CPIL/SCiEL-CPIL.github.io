"""Optimized construction for K(2,2,2)-free 3-graphs on 16 vertices."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)

# Precompute triple indices for fast lookup
# Map each sorted triple to an index 0..559
_triple_to_idx = {}
for i, t in enumerate(ALL_TRIPLES):
    _triple_to_idx[t] = i

def _tidx(a, b, c):
    """Get index of sorted triple (a,b,c)."""
    if a > b:
        a, b = b, a
    if a > c:
        a, c = c, a
    if b > c:
        b, c = c, b
    return _triple_to_idx[(a, b, c)]

def _check_conflict(a, b, c, present):
    """
    Check if adding triple (a,b,c) would create a K(2,2,2).
    present is a set of triple indices already in the graph.
    Returns True if there's a conflict.
    """
    # For K(2,2,2), we need 6 distinct vertices partitioned into 3 pairs
    # such that all 8 transversal triples are present.
    # If we're adding (a,b,c), then for any a', b', c' distinct from {a,b,c}
    # and from each other, we check if all 8 triples are present.
    
    V_minus = [x for x in range(V) if x != a and x != b and x != c]
    
    for a_p in V_minus:
        # Check (a_p, b, c) must be present
        if _tidx(a_p, b, c) not in present:
            continue
        for b_p in V_minus:
            if b_p == a_p:
                continue
            # Check (a, b_p, c) must be present
            if _tidx(a, b_p, c) not in present:
                continue
            for c_p in V_minus:
                if c_p == a_p or c_p == b_p:
                    continue
                # Check the remaining 5 triples (a,b,c is being added, 
                # (a_p,b,c) and (a,b_p,c) already checked)
                if (_tidx(a, b, c_p) in present and
                    _tidx(a_p, b_p, c) in present and
                    _tidx(a_p, b, c_p) in present and
                    _tidx(a, b_p, c_p) in present and
                    _tidx(a_p, b_p, c_p) in present):
                    return True
    return False

def _greedy(order, deadline):
    """Greedy construction with given triple order."""
    present = set()
    for t in order:
        if time.time() >= deadline:
            break
        a, b, c = t
        if not _check_conflict(a, b, c, present):
            present.add(_tidx(a, b, c))
    return present

def _local_search(present, deadline):
    """
    Local search: try adding triples that were skipped, and try swapping.
    Returns improved set of triple indices.
    """
    present = set(present)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any triple not in present
        for t in ALL_TRIPLES:
            if time.time() >= deadline:
                break
            a, b, c = t
            idx = _tidx(a, b, c)
            if idx in present:
                continue
            if not _check_conflict(a, b, c, present):
                present.add(idx)
                improved = True
        
        # Try removing one triple and adding two others (swap-like moves)
        # This is expensive, so only do a limited number
        if time.time() < deadline:
            # Try single removals to see if we can add more
            # Actually, let's try a different approach: random restarts within local search
            pass
    
    return present

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin before 120s limit
    
    best_present = set()
    best_count = 0
    
    # Strategy 1: Multiple randomized greedy passes
    for attempt in range(50):
        if time.time() >= deadline:
            break
        
        order = list(ALL_TRIPLES)
        random.shuffle(order)
        
        present = _greedy(order, deadline)
        
        if len(present) > best_count:
            best_count = len(present)
            best_present = set(present)
    
    # Strategy 2: Local search on the best result
    if time.time() < deadline:
        improved = _local_search(best_present, deadline)
        if len(improved) > best_count:
            best_count = len(improved)
            best_present = improved
    
    # Strategy 3: More greedy passes with different heuristics
    # Try greedy but prioritize triples that appear in many potential K(2,2,2)
    if time.time() < deadline:
        # Compute for each triple how many K(2,2,2) it participates in
        # (i.e., how many ways it can be part of a K(2,2,2))
        # This is a proxy for "criticality" - triples that are more critical
        # should be added earlier to block more K(2,2,2)s
        
        # Actually, let's try a different approach: start from a random subset
        # and greedily add
        for attempt in range(30):
            if time.time() >= deadline:
                break
            
            # Start with a small random seed
            seed_size = random.randint(10, 50)
            seed_triples = random.sample(ALL_TRIPLES, seed_size)
            present = set()
            
            # Add seed triples if they don't conflict
            for t in seed_triples:
                a, b, c = t
                if not _check_conflict(a, b, c, present):
                    present.add(_tidx(a, b, c))
            
            # Then greedily add the rest
            remaining = [t for t in ALL_TRIPLES if _tidx(*t) not in present]
            random.shuffle(remaining)
            
            for t in remaining:
                if time.time() >= deadline:
                    break
                a, b, c = t
                if not _check_conflict(a, b, c, present):
                    present.add(_tidx(a, b, c))
            
            if len(present) > best_count:
                best_count = len(present)
                best_present = set(present)
    
    # Strategy 4: Local search again
    if time.time() < deadline:
        improved = _local_search(best_present, deadline)
        if len(improved) > best_count:
            best_count = len(improved)
            best_present = improved
    
    # Convert back to list of triples
    idx_to_triple = {i: t for i, t in enumerate(ALL_TRIPLES)}
    result = [list(idx_to_triple[i]) for i in best_present]
    
    return result
