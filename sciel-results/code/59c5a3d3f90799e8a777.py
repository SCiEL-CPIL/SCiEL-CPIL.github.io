"""Triple systems on 16 points without K(2,2,2) - maximum independent set approach."""
import time
import random
from itertools import combinations

# Deadline management
_DEADLINE = 145.0  # safety margin before 150s kill
_start_time = None

def _time_left():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)

# Precompute all C(16,3) = 560 triples
_ALL_TRIPLES = list(combinations(range(16), 3))
N_TRIPLES = len(_ALL_TRIPLES)

# Map triple to index
_triple_to_idx = {t: i for i, t in enumerate(_ALL_TRIPLES)}

# Conflict graph: two triples conflict if they form a K(2,2,2) configuration
# K(2,2,2) = complete tripartite graph with parts of size 2.
# Three disjoint pairs (a,b), (c,d), (e,f) - the 8 transversal triples are:
# (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f)
# So two triples conflict if they are part of the same K(2,2,2) configuration.
# More precisely: triple T1=(x,y,z) and T2=(x',y',z') conflict if there exist
# three disjoint pairs such that both T1 and T2 are transversal triples of that K(2,2,2).
#
# Actually, the condition is: no three disjoint pairs have ALL EIGHT transversal triples present.
# So we need to ensure that for any three disjoint pairs, not all 8 transversal triples are in our set.
#
# Two triples conflict if they are both transversal triples of some K(2,2,2) with three disjoint pairs.
# Let's think about when two triples T1 and T2 are both in a K(2,2,2).
# T1 = (a, c, e), T2 = (b, d, f) where {a,b}, {c,d}, {e,f} are the three pairs.
# So T1 and T2 conflict if they share no common point and together determine three disjoint pairs.
# Wait, that's not quite right either. Let me reconsider.
#
# A K(2,2,2) is determined by three disjoint pairs. The 8 transversal triples are all choices of one from each pair.
# Two triples T1, T2 are in the same K(2,2,2) if there exist three disjoint pairs P1, P2, P3 such that
# T1 picks one from each Pi and T2 picks one from each Pi.
# This means T1 and T2 must have no common elements, and for each position i, the two elements (from T1 and T2)
# must be paired together.
#
# More concretely: T1=(x1,x2,x3), T2=(y1,y2,y3). They conflict if after some permutation of positions,
# {x1,y1}, {x2,y2}, {x3,y3} are three disjoint pairs.
# Since triples are unordered, we need: the 6 elements of T1 ∪ T2 can be partitioned into 3 pairs
# such that T1 contains exactly one from each pair and T2 contains exactly one from each pair.
# This is equivalent to: T1 ∩ T2 = ∅ and there exists a perfect matching between T1 and T2
# such that each matched pair is disjoint from the others (which is automatic since T1∩T2=∅).
# Actually any bijection between T1 and T2 gives 3 pairs, and since T1∩T2=∅, all 6 elements are distinct.
# So T1 and T2 conflict iff T1 ∩ T2 = ∅.
#
# Wait, that's too broad. Let me re-examine.
# If T1=(a,c,e) and T2=(b,d,f) with a≠b, c≠d, e≠f and all 6 distinct, then yes they form a K(2,2,2).
# But what if T1=(a,c,e) and T2=(a,d,f)? They share 'a', so they can't be in the same K(2,2,2) with disjoint pairs.
#
# So: two triples conflict iff they are disjoint (share no elements).
# Wait no! That's not right either. If T1=(a,c,e) and T2=(b,d,f) are disjoint, they form ONE K(2,2,2).
# But the constraint is that NO K(2,2,2) has all 8 transversal triples present.
# So the conflict isn't pairwise between triples - it's that we can't have all 8 triples of any K(2,2,2).
#
# This means the problem is NOT a simple maximum independent set problem! It's a hypergraph independence problem
# where the "hyperedges" are the sets of 8 transversal triples for each K(2,2,2).
#
# Let me reconsider. The constraint is: for every set of 3 disjoint pairs, not all 8 transversal triples are present.
# This is a 8-uniform hypergraph constraint.
#
# For a practical approach, let's use a greedy + local search approach:
# 1. Start with a random subset of triples
# 2. Add triples greedily while maintaining the constraint
# 3. Use local search to swap triples for improvements

def _count_k222_violations(triple_set):
    """Count number of K(2,2,2) configurations where all 8 transversal triples are present."""
    if not triple_set:
        return 0
    tset = set(triple_set)
    violations = 0
    # Iterate over all possible sets of 3 disjoint pairs
    # A set of 3 disjoint pairs is determined by choosing 6 distinct points and partitioning into 3 pairs
    # Number of ways: C(16,6) * (number of ways to partition 6 elements into 3 unordered pairs) = 8008 * 15 = 120120
    # That's too many to check each time. Let's use a different approach.
    return 0

def _is_k222_free(triples):
    """Check if a set of triples is K(2,2,2)-free."""
    tset = set(triples)
    # For each triple, check all K(2,2,2)s that contain it
    # A triple (a,c,e) is in K(2,2,2) with pairs {a,b},{c,d},{e,f} for any b≠a, d≠c, f≠e with all 6 distinct
    # For triple (a,c,e), the K(2,2,2)s containing it have pairs {a,b},{c,d},{e,f} where b,d,f are chosen
    # from the remaining 13 points, with b∉{a,c,e}, d∉{a,c,e,b}, f∉{a,c,e,b,d}
    # The 8 transversal triples are: (a,c,e),(a,c,f),(a,d,e),(a,d,f),(b,c,e),(b,c,f),(b,d,e),(b,d,f)
    for (a, c, e) in tset:
        # Choose b from remaining 13 points (not a,c,e)
        for b in range(16):
            if b in (a, c, e):
                continue
            for d in range(16):
                if d in (a, c, e, b):
                    continue
                for f in range(16):
                    if f in (a, c, e, b, d):
                        continue
                    # Check if all 8 transversal triples are present
                    if (a, c, f) in tset and (a, d, e) in tset and (a, d, f) in tset and \
                       (b, c, e) in tset and (b, c, f) in tset and (b, d, e) in tset and (b, d, f) in tset:
                        return False
    return True

def _count_violations(triples):
    """Count K(2,2,2) violations (for scoring in local search)."""
    tset = set(triples)
    violations = 0
    seen_configs = set()
    for (a, c, e) in tset:
        for b in range(16):
            if b in (a, c, e):
                continue
            for d in range(16):
                if d in (a, c, e, b):
                    continue
                for f in range(16):
                    if f in (a, c, e, b, d):
                        continue
                    # Normalize the config: sort the three pairs
                    config = frozenset([frozenset((a,b)), frozenset((c,d)), frozenset((e,f))])
                    if config in seen_configs:
                        continue
                    seen_configs.add(config)
                    if (a, c, f) in tset and (a, d, e) in tset and (a, d, f) in tset and \
                       (b, c, e) in tset and (b, c, f) in tset and (b, d, e) in tset and (b, d, f) in tset:
                        violations += 1
    return violations

def _greedy_construct(order):
    """Greedy: add triples in given order if it doesn't create a violation."""
    result = []
    tset = set()
    for t in order:
        # Check if adding t creates a violation
        new_set = tset | {t}
        if _is_k222_free(new_set):
            result.append(t)
            tset = new_set
    return result

def _local_search(triples, max_iter=5000):
    """Local search: try to add triples or swap to improve."""
    triples = list(triples)
    tset = set(triples)
    best = len(triples)
    
    for it in range(max_iter):
        if _time_left() < 2.0:
            break
        improved = False
        # Try adding a random triple not in the set
        candidates = [t for t in _ALL_TRIPLES if t not in tset]
        random.shuffle(candidates)
        for t in candidates[:50]:  # Try up to 50 candidates
            new_set = tset | {t}
            if _is_k222_free(new_set):
                triples.append(t)
                tset = new_set
                improved = True
                best = len(triples)
                break
        if improved:
            continue
        
        # Try swapping: remove one, add another
        if len(triples) > 0:
            for _ in range(20):
                if _time_left() < 2.0:
                    break
                i = random.randint(0, len(triples) - 1)
                removed = triples[i]
                candidates = [t for t in _ALL_TRIPLES if t not in tset]
                random.shuffle(candidates)
                for t in candidates[:30]:
                    new_set = (tset - {removed}) | {t}
                    if _is_k222_free(new_set):
                        triples[i] = t
                        tset = new_set
                        improved = True
                        break
                if improved:
                    break
    return triples

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    best_triples = []
    best_size = 0
    
    # Multiple restarts with different orderings
    for restart in range(200):
        if _time_left() < 5.0:
            break
        
        # Create a random ordering
        order = list(_ALL_TRIPLES)
        random.shuffle(order)
        
        # Greedy construction
        triples = _greedy_construct(order)
        size = len(triples)
        
        if size > best_size:
            best_triples = triples[:]
            best_size = size
        
        # Local search improvement
        triples = _local_search(triples, max_iter=1000)
        size = len(triples)
        
        if size > best_size:
            best_triples = triples[:]
            best_size = size
    
    # Final polish on the best solution
    if _time_left() > 10.0:
        best_triples = _local_search(best_triples, max_iter=10000)
        best_size = len(best_triples)
    
    # Verify the result is valid
    if not _is_k222_free(best_triples):
        # Fallback: use the initial seed solution
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return [list(t) for t in best_triples]
