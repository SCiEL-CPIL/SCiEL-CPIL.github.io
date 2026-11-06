"""Triple systems on 16 points without the complete 3-partite K(2,2,2).

Strategy:
  1. Precompute the conflict graph: vertices are all 560 triples on 16 points;
     an edge connects two triples if together they can form a K(2,2,2) with
     one more triple.  Equivalently, two triples T1={a,b,c}, T2={d,e,f} with
     disjoint point-sets conflict iff there exist points x,y,z (one from each
     of three disjoint pairs) such that {x,y,z} is the "third" transversal.
     
     More precisely: K(2,2,2) has three pairs of points {a,a'}, {b,b'}, {c,c'}.
     The eight transversal triples are all 2x2x2 combinations.  A set of triples
     "contains K(2,2,2)" if there exist three disjoint pairs such that all eight
     transversals are in the set.
     
     For the greedy independent-set approach, two triples conflict if adding both
     would make it *impossible* to avoid completing a K(2,2,2) — but that's not
     the right model.  The correct model: we want a set S of triples such that
     no three disjoint pairs have all 8 transversals in S.
     
     Two triples T1, T2 conflict (cannot both be in S) if there exist two more
     triples T3, T4 such that {T1,T2,T3,T4,...} contains a full K(2,2,2).  This
     is complex.  Simpler: use the known fact that a pair of triples T1={a,b,c},
     T2={d,e,f} with |{a,b,c} ∩ {d,e,f}| < 3 may or may not conflict.
     
     Actually the cleanest approach from the observed improvement (score 2.362):
     use a randomized greedy with local search.  The conflict check for a single
     candidate triple against the current set is: does adding it create a
     K(2,2,2)?  That requires checking all triples of disjoint pairs.
     
     Let me think about this differently.  A K(2,2,2) is determined by 3 disjoint
     pairs {p1,p1'}, {p2,p2'}, {p3,p3'}.  The 8 transversals are all {pi or pi',
     pj or pj', pk or pk'} for i<j<k.  So a set S contains a K(2,2,2) iff there
     exist 3 disjoint pairs such that all 8 transversals are in S.
     
     To check if adding triple T to S creates a K(2,2,2): for each pair of
     triples T1, T2 already in S that together with T could be part of a
     K(2,2,2), check if the remaining 5 transversals are also in S.  This is
     expensive per-check.
     
     Alternative: precompute for each triple T the set of "forbidden pairs" —
     pairs of other triples that, together with T, complete a K(2,2,2).  Then
     T can be added to S iff no such forbidden pair is fully contained in S.
     
     Given 150s budget, let's use a simpler but effective approach:
     - Randomized greedy: shuffle all 560 triples, try to add each if it doesn't
       create a K(2,2,2) with the current set.
     - Local search: try to swap out one triple for one or two others.
     - Multiple restarts.
     
     For the conflict check: when considering adding triple T={a,b,c} to set S,
     we need to verify that no three disjoint pairs {a,a'},{b,b'},{c,c'} have all
     8 transversals in S∪{T}.  Since T is one of the 8, we need the other 7 to be
     in S.  The other 7 transversals are determined by the choice of a',b',c'.
     
     For T={a,b,c}, the possible a' values are the 15 other points, but for the
     pairs to be disjoint from {a,b,c}, a'∉{b,c}, b'∉{a,c}, c'∉{a,b}, and all
     six distinct.  So a'∈[0,16)\{a,b,c}, b'∈[0,16)\{a,c,a'}, c'∈[0,16)\{a,b,b',a'}.
     
     For each such (a',b',c'), check if the 7 other transversals are all in S.
     Number of (a',b',c') triples: 13 * 12 * 11 = 1716.  For each, 7 lookups.
     That's ~12k operations per candidate.  With 560 candidates and many restarts,
     this should be fast enough.
"""

import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 145.0  # safety margin from 150s

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _timeup():
    return _elapsed() > _DEADLINE

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n = 16
    # All C(16,3) = 560 triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
    num_triples = len(all_triples)
    
    # Index triples for fast lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # For each triple, precompute the set of (a', b', c') completions
    # that would make a K(2,2,2) with this triple as one of the 8 transversals.
    # For triple T = (a, b, c), a K(2,2,2) containing T has T as one of the 8
    # transversals.  The 8 transversals are {pi or pi', pj or pj', pk or pk'}
    # for pairs {p1,p1'},{p2,p2'},{p3,p3'}.
    # T = (a,b,c) means a is from pair 1, b from pair 2, c from pair 3 (in some assignment).
    # Actually, the assignment of which point goes to which pair is not fixed.
    # T could be any of the 8 transversals, so a could be from any of the 3 pairs.
    
    # Let's precompute for each triple T the list of "complement sets":
    # For each way T can be a transversal in a K(2,2,2), the other 7 transversals.
    # T = {a,b,c}.  The three pairs must be disjoint and each contains exactly one of a,b,c.
    # So pair 1 contains exactly one of {a,b,c}, pair 2 contains exactly one, pair 3 contains exactly one.
    # WLOG: pair1 contains a, pair2 contains b, pair3 contains c.  But there are 6 ways to assign.
    # For each assignment, pair1 = {a, a'} where a' ∉ {b,c}, pair2 = {b, b'} where b' ∉ {a,c},
    # pair3 = {c, c'} where c' ∉ {a,b}, and all 6 points distinct.
    
    # Precompute: for each triple index, store list of frozensets of the 7 other transversals.
    # But that's a lot of storage.  Instead, let's just compute on the fly with memoization.
    
    # Better: precompute for each triple T the list of "conflict signatures".
    # A conflict signature is a set of 7 triples that, together with T, form a K(2,2,2).
    # T can be added to S iff no conflict signature is fully contained in S.
    
    # Number of conflict signatures per triple: for each of 6 assignments of (a,b,c) to (pair1,pair2,pair3),
    # a' has 13 choices, b' has 12, c' has 11 → 6*13*12*11 = 10296.  Too many to precompute all.
    
    # Let's use a different approach: when checking if T can be added to S,
    # iterate over possible (a',b',c') and check if the 7 other transversals are in S.
    # Use a set for S for O(1) lookup.
    
    def can_add(T, S_set):
        """Check if triple T can be added to set S_set without creating K(2,2,2)."""
        a, b, c = T
        # For each of 6 ways to assign a,b,c to the three pairs
        # and for each choice of a', b', c'
        # check if the 7 other transversals are all in S_set
        points = [a, b, c]
        # Assignment: perm[i] is which point goes to pair i
        # 6 permutations of (0,1,2)
        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        for p1, p2, p3 in perms:
            pa, pb, pc = points[p1], points[p2], points[p3]
            # a' must not be in {pb, pc} and not equal to pa
            # b' must not be in {pa, pc} and not equal to pb
            # c' must not be in {pa, pb} and not equal to pc
            # All six distinct
            for a_prime in range(n):
                if a_prime == pa or a_prime == pb or a_prime == pc:
                    continue
                for b_prime in range(n):
                    if b_prime == pa or b_prime == pb or b_prime == pc or b_prime == a_prime:
                        continue
                    for c_prime in range(n):
                        if c_prime == pa or c_prime == pb or c_prime == pc or c_prime == a_prime or c_prime == b_prime:
                            continue
                        # The 8 transversals:
                        # (pa,pb,pc) = T itself
                        # (a_prime, pb, pc), (pa, b_prime, pc), (pa, pb, c_prime),
                        # (a_prime, b_prime, pc), (a_prime, pb, c_prime), (pa, b_prime, c_prime),
                        # (a_prime, b_prime, c_prime)
                        # Check the other 7:
                        t1 = tuple(sorted((a_prime, pb, pc)))
                        t2 = tuple(sorted((pa, b_prime, pc)))
                        t3 = tuple(sorted((pa, pb, c_prime)))
                        t4 = tuple(sorted((a_prime, b_prime, pc)))
                        t5 = tuple(sorted((a_prime, pb, c_prime)))
                        t6 = tuple(sorted((pa, b_prime, c_prime)))
                        t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                        if (t1 in S_set and t2 in S_set and t3 in S_set and
                            t4 in S_set and t5 in S_set and t6 in S_set and t7 in S_set):
                            return False
        return True
    
    best_set = set()
    best_count = 0
    
    # Run multiple randomized greedy restarts
    while not _timeup():
        rng = random.Random()
        order = list(range(num_triples))
        rng.shuffle(order)
        
        current_set = set()
        for idx in order:
            if _timeup():
                break
            T = all_triples[idx]
            if can_add(T, current_set):
                current_set.add(T)
        
        # Local improvement: try to add triples that were skipped
        if not _timeup():
            for idx in order:
                if _timeup():
                    break
                T = all_triples[idx]
                if T not in current_set:
                    if can_add(T, current_set):
                        current_set.add(T)
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = set(current_set)
        
        # Try local search: swap operations
        if not _timeup() and best_count > 100:
            # Try to remove one triple and add two
            removed = None
            for T in list(best_set):
                if _timeup():
                    break
                trial_set = best_set - {T}
                added = 0
                for idx in range(num_triples):
                    if _timeup():
                        break
                    T2 = all_triples[idx]
                    if T2 in trial_set:
                        continue
                    if can_add(T2, trial_set):
                        trial_set.add(T2)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    best_set = trial_set
                    best_count = len(best_set)
                    break
    
    # Convert to list of lists
    result = [list(t) for t in best_set]
    return result
