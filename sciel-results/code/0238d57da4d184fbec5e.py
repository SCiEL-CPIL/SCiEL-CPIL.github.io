"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
DEADLINE = 140.0  # Safety margin before 150s kill

def _check_conflict(triples_set, t):
    """Check if adding triple t creates a K(2,2,2).
    K(2,2,2) = 8 transversal triples on 6 vertices forming 3 pairs.
    If t={x,y,z} is added, we need x',y',z' such that all 8 of:
    {x,y,z}, {x,y,z'}, {x,y',z}, {x,y',z'}, {x',y,z}, {x',y,z'}, {x',y',z}, {x',y',z'}
    are present (the other 7 must be in triples_set).
    """
    x, y, z = t
    for xp in range(V):
        if xp in (x, y, z):
            continue
        for yp in range(V):
            if yp in (x, y, z, xp):
                continue
            for zp in range(V):
                if zp in (x, y, z, xp, yp):
                    continue
                # Check the other 7 transversal triples
                if ((x, y, zp) in triples_set and
                    (x, yp, z) in triples_set and
                    (x, yp, zp) in triples_set and
                    (xp, y, z) in triples_set and
                    (xp, y, zp) in triples_set and
                    (xp, yp, z) in triples_set and
                    (xp, yp, zp) in triples_set):
                    return True
    return False

def _greedy_randomized(triples_order, deadline):
    """Greedy: add triples in given order if no conflict."""
    triples_set = set()
    for t in triples_order:
        if time.time() > deadline:
            break
        if not _check_conflict(triples_set, t):
            triples_set.add(t)
    return triples_set

def _local_search(triples_set, deadline):
    """Local search: try to add triples, or swap (remove one, add another)."""
    triples_set = set(triples_set)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try adding any missing triple
        for t in ALL_TRIPLES:
            if t in triples_set:
                continue
            if not _check_conflict(triples_set, t):
                triples_set.add(t)
                changed = True
        # Try 2-swap: remove one triple, try to add a different one
        if not changed:
            current_list = list(triples_set)
            random.shuffle(current_list)
            for t_remove in current_list:
                if time.time() > deadline:
                    return triples_set
                triples_set.discard(t_remove)
                # Try to add multiple triples after removal
                added_any = False
                for t_add in ALL_TRIPLES:
                    if t_add == t_remove or t_add in triples_set:
                        continue
                    if not _check_conflict(triples_set, t_add):
                        triples_set.add(t_add)
                        added_any = True
                if not added_any:
                    triples_set.add(t_remove)  # revert
    return triples_set

def construct():
    start_time = time.time()
    deadline = start_time + (DEADLINE - (time.time() - start_time))
    
    best_set = set()
    best_count = 0
    
    # First, do a simple greedy to get a baseline
    base_order = ALL_TRIPLES[:]
    random.shuffle(base_order)
    base_set = _greedy_randomized(base_order, deadline)
    best_set = base_set
    best_count = len(base_set)
    
    # Multiple randomized restarts with local search
    restart = 0
    while time.time() < deadline - 5.0:  # 5s safety margin
        restart += 1
        # Create a randomized order, biasing toward triples that were in best
        order = ALL_TRIPLES[:]
        random.shuffle(order)
        
        # Greedy phase
        current_set = _greedy_randomized(order, deadline - 10.0)
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
        
        # Local search phase
        if time.time() < deadline - 3.0:
            improved = _local_search(current_set, deadline - 2.0)
            if len(improved) > best_count:
                best_count = len(improved)
                best_set = improved
        
        # Also try local search on best_set itself
        if restart % 3 == 0 and time.time() < deadline - 3.0:
            improved_best = _local_search(best_set, deadline - 2.0)
            if len(improved_best) > best_count:
                best_count = len(improved_best)
                best_set = improved_best
    
    return [[a, b, c] for a, b, c in best_set]
