"""Improved search for linear triple systems on 16 points with small independence number."""
import time
import random
import itertools
from copy import deepcopy

DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _independence_number(triples):
    """Compute the independence number: largest set of points containing no triple."""
    n = 16
    # Build a set of frozensets for fast lookup
    triple_set = set(frozenset(t) for t in triples)
    
    # Check all subsets by size, starting from largest
    # A subset S is independent if no triple is entirely contained in S
    # Equivalently, for each triple, not all three points are in S
    
    best = 0
    # We'll check from size 16 down, but that's too many subsets
    # Better: iterate by size from 16 down to 1, stop when we find one
    # But 2^16 = 65536, which is manageable
    
    # For each subset (represented as bitmask), check if it's independent
    # A subset is independent if for every triple, the triple is NOT a subset of the subset
    
    # Precompute triple bitmasks
    triple_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Check subsets from largest size down
    for size in range(16, 0, -1):
        found = False
        for subset_mask in itertools.combinations(range(16), size):
            sm = 0
            for p in subset_mask:
                sm |= (1 << p)
            # Check if any triple is fully contained in sm
            independent = True
            for tm in triple_masks:
                if (sm & tm) == tm:
                    independent = False
                    break
            if independent:
                best = size
                found = True
                break
        if found:
            break
    
    return best

def _independence_number_fast(triples):
    """Faster independence number computation using bitmask iteration."""
    n = 16
    triple_set = set(frozenset(t) for t in triples)
    
    # Precompute triple bitmasks
    triple_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Iterate over all 2^16 = 65536 subsets
    # For each subset, check if it contains no triple
    # Track the maximum size independent set
    
    best = 0
    for mask in range(1 << 16):
        # Count bits
        bit_count = bin(mask).count('1')
        if bit_count <= best:
            continue
        # Check if this subset is independent (contains no triple)
        independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                independent = False
                break
        if independent:
            best = bit_count
            if best == 16:
                break
    
    return best

def _is_valid(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            overlap = len(set(triples[i]) & set(triples[j]))
            if overlap > 1:
                return False
    return True

def _random_triples():
    """Generate a random valid system of triples."""
    points = list(range(16))
    triples = []
    random.shuffle(points)
    
    # Greedily build triples
    used_pairs = set()
    available = set(range(16))
    
    while len(available) >= 3:
        candidates = list(available)
        random.shuffle(candidates)
        placed = False
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                for k in range(j + 1, len(candidates)):
                    a, b, c = candidates[i], candidates[j], candidates[k]
                    # Check pair constraints
                    pairs = [(a, b), (a, c), (b, c)]
                    valid = True
                    for p in pairs:
                        if p in used_pairs:
                            valid = False
                            break
                    if valid:
                        triples.append([a, b, c])
                        used_pairs.add((a, b))
                        used_pairs.add((a, c))
                        used_pairs.add((b, c))
                        available.discard(a)
                        available.discard(b)
                        available.discard(c)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            break
    
    return triples

def _swap_move(triples, start_time, deadline):
    """Try swap moves to improve independence number."""
    best_triples = deepcopy(triples)
    best_indep = _independence_number_fast(best_triples)
    
    # Try random swaps: pick two triples and swap elements
    points_in_use = set()
    for t in triples:
        for p in t:
            points_in_use.add(p)
    
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        attempts += 1
        # Pick two random triples
        if len(triples) < 2:
            break
        i, j = random.sample(range(len(triples)), 2)
        
        # Pick a random element from each to swap
        a_idx = random.choice([0, 1, 2])
        b_idx = random.choice([0, 1, 2])
        
        orig_i = triples[i][:]
        orig_j = triples[j][:]
        
        triples[i][a_idx], triples[j][b_idx] = triples[j][b_idx], triples[i][a_idx]
        
        # Check validity
        if _is_valid(triples):
            indep = _independence_number_fast(triples)
            if indep < best_indep:
                best_indep = indep
                best_triples = deepcopy(triples)
                if best_indep <= 5:
                    break
        else:
            triples[i] = orig_i
            triples[j] = orig_j
    
    return best_triples, best_indep

def _remove_and_replace(triples, start_time, deadline):
    """Try removing a triple and adding a new one to improve independence number."""
    best_triples = deepcopy(triples)
    best_indep = _independence_number_fast(best_triples)
    
    attempts = 0
    while time.time() < deadline and attempts < 30000:
        attempts += 1
        # Remove a random triple
        if not triples:
            break
        idx = random.randrange(len(triples))
        removed = triples.pop(idx)
        
        # Try to add a new triple using available points
        used_points = set()
        for t in triples:
            for p in t:
                used_points.add(p)
        
        available = [p for p in range(16) if p not in used_points]
        if len(available) < 3:
            # Can't add a full triple, put it back
            triples.insert(idx, removed)
            continue
        
        # Try random triples from available points
        random.shuffle(available)
        new_triple = None
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                for k in range(j + 1, len(available)):
                    a, b, c = available[i], available[j], available[k]
                    # Check pair constraints with existing triples
                    valid = True
                    for t in triples:
                        overlap = len({a, b, c} & set(t))
                        if overlap > 1:
                            valid = False
                            break
                    if valid:
                        new_triple = [a, b, c]
                        break
                if new_triple:
                    break
            if new_triple:
                break
        
        if new_triple:
            triples.append(new_triple)
            if _is_valid(triples):
                indep = _independence_number_fast(triples)
                if indep < best_indep:
                    best_indep = indep
                    best_triples = deepcopy(triples)
                    if best_indep <= 5:
                        break
            else:
                triples.pop()
        else:
            triples.insert(idx, removed)
    
    return best_triples, best_indep

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with a random valid system
    best_triples = _random_triples()
    best_indep = _independence_number_fast(best_triples)
    
    # If the random system is invalid or has few triples, try to build a better one
    if len(best_triples) < 5:
        # Try multiple random restarts
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            t = _random_triples()
            if _is_valid(t):
                indep = _independence_number_fast(t)
                if indep < best_indep or len(t) > len(best_triples):
                    best_indep = indep
                    best_triples = t
    
    # Local search with swaps
    best_triples, best_indep = _swap_move(best_triples, start_time, deadline)
    
    # Local search with remove-and-replace
    if time.time() < deadline - 10:
        best_triples, best_indep = _remove_and_replace(best_triples, start_time, deadline)
    
    # More swap iterations if time remains
    if time.time() < deadline - 5:
        best_triples, best_indep = _swap_move(best_triples, start_time, deadline)
    
    return best_triples
