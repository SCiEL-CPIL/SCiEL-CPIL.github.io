"""Linear triple systems on 16 points with small independence number.
Strategy: generate cyclic candidate triples on Z_16, then simulated annealing
to select a subset minimizing the independence number subject to linearity
(any two triples share at most one point)."""

import random
import time
from itertools import combinations

N = 16
DEADLINE = 140.0  # safety margin before 150s kill


def _build_candidates():
    """Generate candidate triples: all cyclic shifts of base patterns, plus randoms."""
    cands = set()
    # Cyclic: for each base (a,b,c) with a<b<c and small differences, add all shifts
    base_patterns = [
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4),
        (0, 1, 5), (0, 2, 5), (0, 3, 5), (0, 4, 5),
        (0, 1, 6), (0, 2, 6), (0, 3, 6), (0, 4, 6), (0, 5, 6),
        (0, 1, 7), (0, 2, 7), (0, 3, 7), (0, 4, 7), (0, 5, 7), (0, 6, 7),
        (0, 1, 8), (0, 2, 8), (0, 3, 8), (0, 4, 8), (0, 5, 8), (0, 6, 8), (0, 7, 8),
    ]
    for (a, b, c) in base_patterns:
        for s in range(N):
            t = tuple(sorted(((a + s) % N, (b + s) % N, (c + s) % N)))
            cands.add(t)
    # Add some random triples for diversity
    for _ in range(300):
        t = tuple(sorted(random.sample(range(N), 3)))
        cands.add(t)
    return list(cands)


def _independence_number(triples):
    """Compute the independence number (largest set with no triple)."""
    if not triples:
        return N
    # Build a bitset representation: for each triple, a 16-bit mask
    masks = [0] * len(triples)
    for i, t in enumerate(triples):
        m = 0
        for v in t:
            m |= (1 << v)
        masks[i] = m
    # A subset S (bitmask) is independent if no triple is fully contained in S
    # i.e., for all i, masks[i] & S != masks[i]  <=>  (masks[i] | S) == S is false
    # Equivalently: (masks[i] & ~S) != 0 for all i
    # We want max popcount(S) such that for all i, (masks[i] & ~S) != 0
    # Brute force over all 2^16 subsets
    best = 0
    # Precompute: for each subset S, check independence
    # Use the property: S is independent iff no mask[i] is subset of S
    # i.e., for all i: (masks[i] & S) != masks[i]
    for S in range(1 << N):
        ok = True
        for m in masks:
            if (m & S) == m:
                ok = False
                break
        if ok:
            pc = bin(S).count('1')
            if pc > best:
                best = pc
                if best == N:
                    break
    return best


def _independence_number_fast(triples):
    """Faster independence number using complement: find smallest hitting set... 
    Actually, alpha = N - tau where tau is the transversal number (min hitting set).
    But computing tau is also hard. Use the subset enumeration but with early exit."""
    if not triples:
        return N
    masks = [0] * len(triples)
    for i, t in enumerate(triples):
        m = 0
        for v in t:
            m |= (1 << v)
        masks[i] = m
    best = 0
    # Iterate subsets in decreasing order of popcount for early termination
    # But 2^16 = 65536 is small, just do it all
    for S in range(1 << N):
        ok = True
        for m in masks:
            if (m & S) == m:
                ok = False
                break
        if ok:
            pc = bin(S).count('1')
            if pc > best:
                best = pc
    return best


def _is_linear(triple_set):
    """Check that any two triples share at most one point."""
    triples = list(triple_set)
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _can_add(triples_list, new_triple):
    """Check if new_triple can be added without violating linearity."""
    new_set = set(new_triple)
    for t in triples_list:
        common = new_set & set(t)
        if len(common) > 1:
            return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    candidates = _build_candidates()
    rng.shuffle(candidates)
    
    best_triples = []
    best_alpha = N  # worst case
    
    # Initial greedy: add triples that reduce independence number
    current = []
    current_alpha = N
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
        if _can_add(current, cand):
            new_current = current + [cand]
            a = _independence_number_fast(new_current)
            if a < current_alpha:
                current = new_current
                current_alpha = a
                if current_alpha <= 6:
                    break
            # Accept even if same alpha? No, only if it improves or we're building structure
            # Actually, adding a triple that doesn't reduce alpha might still help later
            # For now, only add if it reduces alpha or if we have few triples
            elif len(current) < 10:
                # Add to build structure even if alpha doesn't drop immediately
                current = new_current
    
    best_triples = list(current)
    best_alpha = current_alpha
    
    # Simulated annealing to improve
    if time.time() - start_time < DEADLINE - 20:
        T = 5.0
        cooling = 0.999
        current = list(best_triples)
        current_alpha = best_alpha
        
        while time.time() - start_time < DEADLINE - 5 and T > 0.01:
            # Pick a random move: add, remove, or swap a triple
            move = rng.random()
            if move < 0.4 and len(current) < 40:
                # Add a random candidate
                cand = rng.choice(candidates)
                if _can_add(current, cand):
                    new_current = current + [cand]
                    new_alpha = _independence_number_fast(new_current)
                    delta = new_alpha - current_alpha
                    if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                        current = new_current
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = list(current)
            elif move < 0.7 and len(current) > 2:
                # Remove a random triple
                idx = rng.randrange(len(current))
                new_current = current[:idx] + current[idx+1:]
                new_alpha = _independence_number_fast(new_current)
                delta = new_alpha - current_alpha
                if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                    current = new_current
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = list(current)
            else:
                # Swap: remove one, add another
                if len(current) > 0:
                    idx = rng.randrange(len(current))
                    removed = current[idx]
                    new_current = current[:idx] + current[idx+1:]
                    cand = rng.choice(candidates)
                    if cand != tuple(removed) and _can_add(new_current, cand):
                        new_current = new_current + [cand]
                        new_alpha = _independence_number_fast(new_current)
                        delta = new_alpha - current_alpha
                        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                            current = new_current
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = list(current)
            
            T *= cooling
        
        # Final local search: try to add more triples
        for cand in candidates:
            if time.time() - start_time > DEADLINE - 2:
                break
            if _can_add(best_triples, cand):
                new_t = best_triples + [cand]
                a = _independence_number_fast(new_t)
                if a < best_alpha:
                    best_alpha = a
                    best_triples = new_t
    
    # Verify linearity of best_triples
    if not _is_linear(best_triples):
        # Fallback to seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Convert to list of lists
    return [list(t) for t in best_triples]
