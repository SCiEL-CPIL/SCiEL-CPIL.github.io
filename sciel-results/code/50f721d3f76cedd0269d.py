import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin

    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)

    # Precompute conflict graph: for each triple t, find all triples u such that
    # {t, u} together with some other 6 triples would form a K(2,2,2)
    # Actually, we need: t conflicts with u if there exist 3 disjoint pairs
    # such that t and u are two of the 8 transversal triples.
    # 
    # A K(2,2,2) has 6 vertices: {a,b,c} x {a',b',c'} forming 8 triples.
    # For triple t=(a,b,c) and u=(a',b',c'), they conflict if there exist
    # 3 more disjoint pairs from the remaining 10 points such that all 8
    # transversals are "possible" (i.e., the 8 triples form a K(2,2,2)).
    #
    # Actually, the conflict is: t and u are in the same K(2,2,2) if there
    # exist 6 more points forming 3 pairs such that all 8 transversals exist.
    # But for can_add, we need: adding t creates a K(2,2,2) if there exist
    # 3 disjoint pairs from remaining 13 points such that all 7 other
    # transversals are already in the set.
    #
    # Precompute: for each triple t, for each pair of disjoint pairs (p1, p2, p3)
    # from the 13 remaining points, the 7 "other" triples. If any of those 7
    # is in the current set, adding t is blocked.
    #
    # Better approach: for each triple t, precompute the set of "witness" triples
    # that would block t. t is blocked if ANY of its blocking triples is present.
    #
    # For triple t = (a,b,c), a blocking triple is any triple u = (x,y,z) where
    # {x,y,z} intersects {a,b,c} in exactly 1 point, and there exist 2 more disjoint
    # pairs from the remaining 11 points such that all 8 transversals are formed.
    #
    # Let me think more carefully. A K(2,2,2) on 6 points {v1,v2,v3,v4,v5,v6}
    # partitioned as {v1,v2,v3} and {v4,v5,v6} has 8 triples: all combinations
    # of one from each part... no, that's not right either.
    #
    # K(2,2,2) is the complete 3-partite 3-uniform hypergraph with parts of size 2.
    # Parts: A={a1,a2}, B={b1,b2}, C={c1,c2}. The 8 triples are all (ai, bj, ck).
    #
    # So for t=(a,b,c) to be part of a K(2,2,2), there must exist a',b',c' all
    # distinct from a,b,c and from each other, such that all 8 triples
    # (a or a', b or b', c or c') are in the system.
    #
    # When checking if t can be added, we need: for all choices of a',b',c'
    # (distinct, not in {a,b,c}), the 7 other transversals are NOT all present.
    #
    # Precompute for each t the set of "critical" triples: u is critical for t
    # if there exists a K(2,2,2) containing both t and u, and u is one of the
    # 7 other transversals. Then t can be added iff no critical triple for t
    # is in the current set... wait, that's not quite right either.
    #
    # t can be added iff for ALL (a',b',c'), at least one of the 7 other
    # transversals is absent. Equivalently, t is blocked iff there EXISTS
    # (a',b',c') such that ALL 7 other transversals are present.
    #
    # Let me precompute: for each t, the list of "witness sets" - each witness
    # set is a frozenset of 7 triples. t is blocked if any witness set is
    # a subset of the current set.
    #
    # This is still expensive. Let me use a different approach:
    # For each t, precompute all 7-tuples of transversals for each (a',b',c').
    # Then use a bitset or sorted list for fast subset checking.
    #
    # Actually, the simplest speedup: precompute for each t, for each (a',b',c'),
    # the 7 other triples as a frozenset. Store these as a list. Then can_add
    # just checks if any frozenset is a subset of current_set.
    #
    # Even better: use a counter approach. For each t, maintain a count of how
    # many witness sets are fully contained. But that's complex.
    #
    # Let me just precompute the witness frozensets and use efficient checking.

    # Precompute witness sets for each triple
    triple_index = {t: i for i, t in enumerate(all_triples)}
    witness_sets = [[] for _ in all_triples]

    for t_idx, t in enumerate(all_triples):
        a, b, c = t
        remaining = [x for x in range(n) if x not in (a, b, c)]
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # The 7 other transversals
                    others = frozenset([
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime)
                    ])
                    witness_sets[t_idx].append(others)

    def can_add(t, current_set):
        """Check if triple t can be added to current_set without creating K(2,2,2)."""
        t_idx = triple_index[t]
        for ws in witness_sets[t_idx]:
            # Check if all 7 triples in ws are in current_set
            # Use a fast check: count how many are in current_set
            count = 0
            for tr in ws:
                if tr in current_set:
                    count += 1
                    if count < 7:
                        continue
                    else:
                        return False
        return True

    def greedy_pass(order):
        current = set()
        for t in order:
            if time.time() > DEADLINE:
                break
            if can_add(t, current):
                current.add(t)
        return current

    def fill_pass(current):
        """Try to add any missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t in all_triples:
                if time.time() > DEADLINE:
                    return current
                if t not in current and can_add(t, current):
                    current.add(t)
                    improved = True
        return current

    def simulated_annealing_polish(current, rng):
        """Try to improve current set via swaps and additions."""
        current = set(current)
        best_size = len(current)
        T = 1.0
        T_min = 0.01
        cooling = 0.95

        while T > T_min and time.time() < DEADLINE:
            # Try random operations
            for _ in range(50):
                if time.time() > DEADLINE:
                    return current

                op = rng.random()
                if op < 0.3:
                    # Try to add a random missing triple
                    missing = [t for t in all_triples if t not in current]
                    if not missing:
                        continue
                    t = rng.choice(missing)
                    if can_add(t, current):
                        current.add(t)
                        if len(current) > best_size:
                            best_size = len(current)
                elif op < 0.6:
                    # Try to remove a random present triple
                    present = list(current)
                    if not present:
                        continue
                    t = rng.choice(present)
                    current.discard(t)
                else:
                    # Try a swap: remove one, try to add another
                    present = list(current)
                    missing = [t for t in all_triples if t not in current]
                    if not present or not missing:
                        continue
                    t_remove = rng.choice(present)
                    t_add = rng.choice(missing)
                    current.discard(t_remove)
                    if can_add(t_add, current):
                        current.add(t_add)
                        if len(current) > best_size:
                            best_size = len(current)
                    else:
                        current.add(t_remove)

            T *= cooling

        # Final fill
        current = fill_pass(current)
        return current

    best = set()

    # Multiple randomized greedy passes
    rng = random.Random(42)
    num_passes = 15

    for i in range(num_passes):
        if time.time() > DEADLINE:
            break
        order = list(all_triples)
        rng.shuffle(order)
        result = greedy_pass(order)
        result = fill_pass(result)
        if len(result) > len(best):
            best = result

    # Apply simulated annealing polish to the best result
    if time.time() < DEADLINE:
        polished = simulated_annealing_polish(best, rng)
        if len(polished) > len(best):
            best = polished

    # A few more passes with different seeds
    for i in range(5):
        if time.time() > DEADLINE:
            break
        rng2 = random.Random(100 + i)
        order = list(all_triples)
        rng2.shuffle(order)
        result = greedy_pass(order)
        result = fill_pass(result)
        if len(result) > len(best):
            best = result
        if time.time() < DEADLINE:
            polished = simulated_annealing_polish(result, rng2)
            if len(polished) > len(best):
                best = polished

    return [list(t) for t in best]
