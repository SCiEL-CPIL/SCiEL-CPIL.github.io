"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples via bitmask + local search."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N = len(ALL_TRIPLES)  # 560

# Index each triple
TRIPLE_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}

# For each triple, precompute the set of triples that form a K(2,2,2) with it
# K(2,2,2) = 8 triples forming a complete 3-partite with parts {x1,x2},{y1,y2},{z1,z2}
# Two triples conflict if they are part of the same 8-set K(2,2,2)

def _build_conflict_masks():
    """Build bitmask of conflicts for each triple.
    Two triples conflict if they are both in some K(2,2,2) 8-set.
    """
    # For each set of 6 distinct vertices, there are 3 ways to partition into 3 pairs
    # Each partition gives one K(2,2,2) with 8 triples
    # We need: for each pair of triples (t1, t2), check if they co-occur in some K(2,2,2)
    
    # Precompute: for each 6-tuple of distinct vertices, generate the 3 K(2,2,2) structures
    # A K(2,2,2) is defined by partitioning 6 vertices into 3 pairs: {a,b}, {c,d}, {e,f}
    # The 8 triples are all (one from each pair)
    
    # For each triple t, find all other triples t' such that t and t' are in the same K(2,2,2)
    # t = (x, y, z) with x in pair1, y in pair2, z in pair3
    # t' must have one vertex from each of the 3 pairs, and be different from t
    # The pairs are determined by: t uses one vertex from each pair, and the other vertices
    # in each pair must be distinct from t's vertices and from each other
    
    # For triple (a,b,c):
    # Pair containing a: {a, a'} where a' != a, b, c
    # Pair containing b: {b, b'} where b' != a, b, c, a'
    # Pair containing c: {c, c'} where c' != a, b, c, a', b'
    # Then the 8 triples are all combinations of {a,a'}, {b,b'}, {c,c'}
    # t' is any other combination
    
    conflict = [0] * N  # bitmask, but N=560 > 64, so use Python int (arbitrary precision)
    
    for i, (a, b, c) in enumerate(ALL_TRIPLES):
        used = {a, b, c}
        for a_prime in range(V):
            if a_prime in used:
                continue
            for b_prime in range(V):
                if b_prime in used or b_prime == a_prime:
                    continue
                for c_prime in range(V):
                    if c_prime in used or c_prime == a_prime or c_prime == b_prime:
                        continue
                    # The 8 triples of this K(2,2,2):
                    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
                    eight = [
                        (a, b, c),
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime),
                    ]
                    for t2 in eight:
                        j = TRIPLE_IDX[tuple(sorted(t2))]
                        if j != i:
                            conflict[i] |= (1 << j)
    
    return conflict

CONFLICT = _build_conflict_masks()

def _is_valid_add_bitmask(selected_mask, t_idx):
    """Check if adding triple t_idx to selected_mask creates a K(2,2,2).
    t_idx conflicts with some selected triple if there exists a K(2,2,2) containing both.
    But wait: the conflict mask says t and t' are in the SAME K(2,2,2).
    Adding t creates a K(2,2,2) iff there exist 7 other triples all in selected_mask
    that together with t form a K(2,2,2).
    
    Actually, the conflict relation is: t and t' conflict if they are in the same K(2,2,2).
    But that's not sufficient - we need ALL 8 triples of that K(2,2,2) to be present.
    
    Let me reconsider. The correct check: adding t creates a K(2,2,2) iff there exists
    a set of 8 triples forming a K(2,2,2) that includes t and the other 7 are in selected.
    
    This is more complex than just checking conflict mask. Let me use a different approach:
    For the new triple t=(a,b,c), a K(2,2,2) containing it must have t as one of the 8.
    t = (a,b,c) means a is in pair1, b in pair2, c in pair3.
    The other vertices: a' in pair1, b' in pair2, c' in pair3.
    The 8 triples are all combos of {a,a'}, {b,b'}, {c,c'}.
    We need the other 7 to be in selected.
    
    So: for each (a', b', c') with all distinct from a,b,c and from each other,
    check if the 7 other triples are in selected_mask.
    """
    a, b, c = ALL_TRIPLES[t_idx]
    used = {a, b, c}
    
    for a_prime in range(V):
        if a_prime in used:
            continue
        for b_prime in range(V):
            if b_prime in used or b_prime == a_prime:
                continue
            for c_prime in range(V):
                if c_prime in used or c_prime == a_prime or c_prime == b_prime:
                    continue
                # Check the 7 other triples
                others = [
                    (a, b, c_prime),
                    (a, b_prime, c),
                    (a, b_prime, c_prime),
                    (a_prime, b, c),
                    (a_prime, b, c_prime),
                    (a_prime, b_prime, c),
                    (a_prime, b_prime, c_prime),
                ]
                ok = True
                for t2 in others:
                    j = TRIPLE_IDX[tuple(sorted(t2))]
                    if not (selected_mask >> j) & 1:
                        ok = False
                        break
                if ok:
                    return False
    return True


def _greedy(order, deadline):
    """Greedily add triples in given order."""
    selected_mask = 0
    result = []
    for t_idx in order:
        if time.time() >= deadline:
            break
        if (selected_mask >> t_idx) & 1:
            continue
        if _is_valid_add_bitmask(selected_mask, t_idx):
            selected_mask |= (1 << t_idx)
            result.append(list(ALL_TRIPLES[t_idx]))
    return result


def _local_search(result, deadline, max_iters=5000):
    """Try to improve by swapping: remove one triple, add another.
    Or: remove two, add three. Start with simple 1-for-1 and 1-for-2 swaps.
    """
    selected_set = set(tuple(t) for t in result)
    selected_mask = 0
    for t in result:
        idx = TRIPLE_IDX[tuple(sorted(t))]
        selected_mask |= (1 << idx)
    
    # Precompute for each triple in result, which triples it conflicts with (in the K222 sense)
    # Actually, let's just try random swaps
    
    best = result[:]
    best_len = len(result)
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Try to add a random triple not in selected
        # Pick a random unselected triple
        while True:
            cand = random.randrange(N)
            if not (selected_mask >> cand) & 1:
                break
        
        if _is_valid_add_bitmask(selected_mask, cand):
            selected_mask |= (1 << cand)
            best = result[:]
            best.append(list(ALL_TRIPLES[cand]))
            result = best
            best_len += 1
        else:
            # Try swapping: remove one selected triple, add cand
            # Pick a random selected triple to remove
            if not result:
                continue
            remove_idx = random.randrange(len(result))
            removed = result[remove_idx]
            removed_idx = TRIPLE_IDX[tuple(sorted(removed))]
            
            # Remove it
            new_mask = selected_mask & ~(1 << removed_idx)
            
            if _is_valid_add_bitmask(new_mask, cand):
                # Do the swap
                selected_mask = new_mask | (1 << cand)
                result[remove_idx] = list(ALL_TRIPLES[cand])
                # Length stays the same, but composition changed
                # This might enable further improvements
    
    return best


def _full_validation(triples):
    """Check that no K(2,2,2) exists in the triple set."""
    tset = set(tuple(sorted(t)) for t in triples)
    for x1, x2, y1, y2, z1, z2 in combinations(range(V), 6):
        eight = [
            (x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
            (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2),
        ]
        if all(tuple(sorted(t)) in tset for t in eight):
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best = None
    best_len = -1
    
    # Strategy: multiple greedy restarts with different random orders, then local search
    for attempt in range(300):
        if time.time() >= deadline - 10:  # Leave 10s for local search on best
            break
        
        # Random order
        order = list(range(N))
        random.shuffle(order)
        
        result = _greedy(order, deadline - 10)
        
        if len(result) > best_len:
            best_len = len(result)
            best = result
        
        # If we have time, do local search on the best
        if best is not None and time.time() < deadline - 5:
            improved = _local_search(best, deadline - 2, max_iters=2000)
            if len(improved) > best_len:
                best_len = len(improved)
                best = improved
    
    # Final local search on the best
    if best is not None and time.time() < deadline - 2:
        improved = _local_search(best, deadline - 1, max_iters=5000)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved
    
    if best is None:
        # Fallback: star at vertex 0
        best = [list(t) for t in [(0, a, b) for a in range(1, V) for b in range(a + 1, V)]]
    
    # Validate
    if not _full_validation(best):
        # Remove triples until valid (shouldn't happen often)
        best_set = set(tuple(sorted(t)) for t in best)
        # Just return the star as fallback
        best = [list(t) for t in [(0, a, b) for a in range(1, V) for b in range(a + 1, V)]]
    
    return best
