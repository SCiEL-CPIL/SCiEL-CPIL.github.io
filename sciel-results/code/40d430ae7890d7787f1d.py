"""
Maximize the number of triples in a 3-uniform hypergraph on 16 vertices
with no K(2,2,2) (i.e., no three disjoint pairs {a,a'}, {b,b'}, {c,c'}
such that all 8 transversal triples are present).

Strategy:
1. Precompute all 560 triples and all K(2,2,2) constraint groups (sets of 8 triples).
2. Greedy initial construction with a fast O(1)-ish validity check using precomputed conflict maps.
3. Simulated-annealing / local search to escape local optima.
"""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# ── Precomputation ──────────────────────────────────────────────────────────

_ALL_TRIPLES = list(itertools.combinations(range(16), 3))
_N = len(_ALL_TRIPLES)
_TRIPLE_IDX = {t: i for i, t in enumerate(_ALL_TRIPLES)}

# For each pair (x,y), list of triples containing both x and y
# For each triple index, the set of "conflict" triple indices that would complete a K(2,2,2)
# Actually, we need: for a given triple t, which other triples are "close" to forming a K(2,2,2) with it?
# A K(2,2,2) consists of 8 triples on 6 vertices. A new triple t can only create a K(2,2,2)
# if there exist two other triples t1, t2 already in the set such that {t, t1, t2, ...} complete an 8-set.
# More precisely: t can create a K(2,2,2) iff there exist 6 distinct vertices and a partition into 3 pairs
# such that t is one of the 8 transversals, and the other 7 transversals are all present.
#
# Better approach: For each triple t, precompute the set of "witness" triples.
# When we add t, we need to check if there exists a K(2,2,2) containing t where all other 7 are present.
# Number of K(2,2,2)s containing a given triple:
#   A triple uses 3 vertices. Pick 3 more vertices from the remaining 13: C(13,3)=286.
#   For each such 6-set, count how many pairings have the triple as a transversal.
#   For a fixed 6-set and a fixed triple (one transversal), the number of pairings that include it:
#   The triple picks one from each pair. Given the triple and the 6-set, the number of ways to partition
#   the remaining 3 vertices (one per pair) such that the triple is a transversal = 2^3 = 8? No.
#   Actually, given 6 vertices and a specific transversal triple (one vertex from each of 3 pairs),
#   the number of pairings where this triple is a transversal is:
#   We need to assign the 3 chosen vertices to 3 pairs, and the 3 unchosen vertices to the other 3 pairs.
#   The chosen vertices must go to distinct pairs, and unchosen to distinct pairs.
#   Number of bijections from chosen to pairs * bijections from unchosen to pairs / (order of pairs)
#   = 3! * 3! / 3! = 6. So 6 pairings per 6-set.
#   Total K(2,2,2)s per triple ≈ 286 * 6 = 1716. That's a lot but manageable for precomputation.
#
# Alternative: for each K(2,2,2) (8-set of triples), add each member to a conflict list.
# Then checking if triple t can be added: for each K(2,2,2) group G containing t, check if all other 7 are in current_set.

# Build all K(2,2,2) groups
_K222_GROUPS = []  # list of frozensets of 8 triple-indices
_seen = set()

# Generate all 6-vertex subsets
for six in itertools.combinations(range(16), 6):
    a, b, c, d, e, f = six
    # 15 perfect matchings of 6 elements into 3 pairs
    # Standard enumeration:
    matchings = [
        (frozenset({a,b}), frozenset({c,d}), frozenset({e,f})),
        (frozenset({a,b}), frozenset({c,e}), frozenset({d,f})),
        (frozenset({a,b}), frozenset({c,f}), frozenset({d,e})),
        (frozenset({a,c}), frozenset({b,d}), frozenset({e,f})),
        (frozenset({a,c}), frozenset({b,e}), frozenset({d,f})),
        (frozenset({a,c}), frozenset({b,f}), frozenset({d,e})),
        (frozenset({a,d}), frozenset({b,c}), frozenset({e,f})),
        (frozenset({a,d}), frozenset({b,e}), frozenset({c,f})),
        (frozenset({a,d}), frozenset({b,f}), frozenset({c,e})),
        (frozenset({a,e}), frozenset({b,c}), frozenset({d,f})),
        (frozenset({a,e}), frozenset({b,d}), frozenset({c,f})),
        (frozenset({a,e}), frozenset({b,f}), frozenset({c,d})),
        (frozenset({a,f}), frozenset({b,c}), frozenset({d,e})),
        (frozenset({a,f}), frozenset({b,d}), frozenset({c,e})),
        (frozenset({a,f}), frozenset({b,e}), frozenset({c,d})),
    ]
    for m in matchings:
        p1, p2, p3 = m
        # 8 transversal triples
        trans = []
        for x in p1:
            for y in p2:
                for z in p3:
                    trans.append(tuple(sorted((x, y, z))))
        group = frozenset(_TRIPLE_IDX[t] for t in trans)
        if group not in _seen:
            _seen.add(group)
            _K222_GROUPS.append(group)

# For each triple index, list of K222 groups it belongs to
_triple_to_groups = [[] for _ in range(_N)]
for gi, group in enumerate(_K222_GROUPS):
    for ti in group:
        _triple_to_groups[ti].append(gi)

# ── Fast validity check ─────────────────────────────────────────────────────

def _can_add(tri_idx, present):
    """Check if adding triple `tri_idx` to `present` (a set of indices) keeps the set K(2,2,2)-free."""
    for gi in _triple_to_groups[tri_idx]:
        group = _K222_GROUPS[gi]
        # Check if all other 7 are present
        # Optimized: iterate and check
        ok = True
        for ti in group:
            if ti != tri_idx and ti not in present:
                ok = False
                break
        if not ok:
            continue
        else:
            # All 7 others present → would create K(2,2,2)
            return False
    return True

# ── Greedy construction ─────────────────────────────────────────────────────

def _greedy(deadline):
    present = set()
    order = list(range(_N))
    random.shuffle(order)
    for ti in order:
        if time.time() > deadline - 3.0:
            break
        if _can_add(ti, present):
            present.add(ti)
    return present

# ── Local search (simulated annealing style) ────────────────────────────────

def _local_search(initial, deadline):
    current = set(initial)
    best = set(current)
    best_size = len(current)
    
    all_triples = list(range(_N))
    missing = [t for t in all_triples if t not in current]
    
    # Phase 1: try adding any missing triple
    random.shuffle(missing)
    for ti in missing:
        if time.time() > deadline - 2.0:
            break
        if _can_add(ti, current):
            current.add(ti)
            if len(current) > best_size:
                best_size = len(current)
                best = set(current)
            # Try to add more
            new_missing = [t for t in missing if t not in current]
            random.shuffle(new_missing)
            for ti2 in new_missing:
                if _can_add(ti2, current):
                    current.add(ti2)
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
    
    # Phase 2: swap-based local search
    while time.time() < deadline - 2.0:
        improved = False
        # Try removing one and adding two
        candidates = list(current)
        random.shuffle(candidates)
        
        for ti_rem in candidates[:30]:  # limit work per iteration
            if time.time() > deadline - 2.0:
                break
            current.remove(ti_rem)
            
            added = []
            missing_now = [t for t in all_triples if t not in current]
            random.shuffle(missing_now)
            
            for ti_add in missing_now:
                if _can_add(ti_add, current):
                    current.add(ti_add)
                    added.append(ti_add)
            
            if len(added) >= 2:
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                break
            else:
                for ti_add in added:
                    current.discard(ti_add)
                current.add(ti_rem)
        
        if not improved:
            # Try a more aggressive move: remove 2, add 3
            if time.time() > deadline - 2.0:
                break
            pair_cands = list(itertools.combinations(current, 2))
            random.shuffle(pair_cands)
            
            done = False
            for rem_pair in pair_cands[:20]:
                if time.time() > deadline - 2.0:
                    break
                r1, r2 = rem_pair
                current.discard(r1)
                current.discard(r2)
                
                added = []
                missing_now = [t for t in all_triples if t not in current]
                random.shuffle(missing_now)
                
                for ti_add in missing_now:
                    if _can_add(ti_add, current):
                        current.add(ti_add)
                        added.append(ti_add)
                
                if len(added) >= 3:
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                    done = True
                    break
                else:
                    for ti_add in added:
                        current.discard(ti_add)
                    current.add(r1)
                    current.add(r2)
            
            if not done:
                break
    
    return best

# ── Multi-start with diverse seeds ──────────────────────────────────────────

def _multi_start(deadline):
    best = set()
    best_size = 0
    
    num_starts = 8
    time_per_start = (deadline - time.time() - 5.0) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        t_start = time.time()
        t_alloc = min(time_per_start, deadline - t_start - 3.0)
        
        # Diverse random seed for this start
        rng = random.Random(s * 7919 + 13)
        old_seed = random.getstate()
        random.seed(s * 7919 + 13)
        
        g = _greedy(t_start + t_alloc * 0.5)
        ls = _local_search(g, t_start + t_alloc)
        
        random.setstate(old_seed)
        
        if len(ls) > best_size:
            best_size = len(ls)
            best = ls
    
    return best

# ── Main entry point ────────────────────────────────────────────────────────

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    # First quick greedy
    g = _greedy(deadline - 60.0)
    best = g
    best_size = len(g)
    
    # Local search on greedy result
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        ls = _local_search(best, start_time + remaining)
        if len(ls) > best_size:
            best = ls
            best_size = len(ls)
    
    # Multi-start if time permits
    remaining = deadline - time.time() - 5.0
    if remaining > 30.0:
        ms = _multi_start(deadline)
        if len(ms) > best_size:
            best = ms
            best_size = len(ms)
    
    # Final polish
    remaining = deadline - time.time() - 2.0
    if remaining > 2.0:
        final = _local_search(best, start_time + remaining)
        if len(final) > best_size:
            best = final
            best_size = len(final)
    
    return [list(_ALL_TRIPLES[i]) for i in best]
