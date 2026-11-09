"""Triple systems on 16 points without K(2,2,2) - optimized construction."""
import time
import random
from itertools import combinations

# Precompute all triples
ALL_TRIPLES = [tuple(sorted(c)) for c in combinations(range(16), 3)]
NUM_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute pairs for each triple
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    TRIPLE_PAIRS.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])

# Map triple to index
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}

def _check_validity_fast(triple_idx, present, pair_to_thirds):
    """Check if adding triple triple_idx creates K(2,2,2).
    
    A K(2,2,2) consists of 3 disjoint pairs {a,a'}, {b,b'}, {c,c'} 
    where all 8 transversals are present.
    
    For the new triple (a,b,c), we need to check if there exist a',b',c'
    (all distinct from a,b,c and each other) such that all 8 transversals
    are in present ∪ {new triple}.
    
    The 8 transversals are:
    (a,b,c), (a,b,c'), (a,b',c), (a,b',c'),
    (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    Since (a,b,c) is the new one, the other 7 must be in present.
    """
    a, b, c = ALL_TRIPLES[triple_idx]
    
    # For each possible a' (not in {a,b,c}), check if there exist b',c'
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        
        # Check pairs involving a_prime
        # We need (a_prime, b, c) and (a_prime, b', c) and (a_prime, b', c')
        # and (a, b_prime, c) and (a, b_prime, c') and (a_prime, b_prime, c) and (a_prime, b_prime, c')
        
        # First, (a_prime, b, c) must be present
        ab_c = (min(a_prime, b), max(a_prime, b))
        if c not in pair_to_thirds.get(ab_c, set()):
            continue
        
        ab_prime_c = (min(a_prime, b_prime) if False else min(a_prime, b), max(a_prime, b))
        # Actually let me redo this more carefully
        
        # Pairs we need to check for the 7 existing transversals:
        # 1. (a', b, c): pair (a',b) contains c, pair (a',c) contains b, pair (b,c) contains a'
        # 2. (a, b', c): pair (a,b') contains c, pair (a,c) contains b', pair (b',c) contains a
        # 3. (a, b, c'): pair (a,b) contains c', pair (a,c') contains b, pair (b,c') contains a
        # 4. (a', b', c): pair (a',b') contains c, pair (a',c) contains b', pair (b',c) contains a'
        # 5. (a', b, c'): pair (a',b) contains c', pair (a',c') contains b, pair (b,c') contains a'
        # 6. (a, b', c'): pair (a,b') contains c', pair (a,c') contains b', pair (b',c') contains a
        # 7. (a', b', c'): pair (a',b') contains c', pair (a',c') contains b', pair (b',c') contains a'
        
        # Let's use a different approach: for each candidate a', find all b' such that
        # (a', b', c) is present, then check if c' exists
        
        # For fixed a', we need to find b' and c' such that all 7 transversals exist.
        # Key insight: (a', b, c) must exist, so b' and c' are constrained.
        
        # Let's iterate over possible b' values
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            
            # Check (a', b', c) exists
            ap_bp_c = (min(a_prime, b_prime), max(a_prime, b_prime))
            if c not in pair_to_thirds.get(ap_bp_c, set()):
                continue
            
            # Check (a, b', c) exists  
            a_bp_c = (min(a, b_prime), max(a, b_prime))
            if c not in pair_to_thirds.get(a_bp_c, set()):
                continue
            
            # Check (a', b, c) exists
            ap_b_c = (min(a_prime, b), max(a_prime, b))
            if c not in pair_to_thirds.get(ap_b_c, set()):
                continue
            
            # Now we need to find c' such that:
            # (a, b, c'), (a', b', c'), (a, b', c'), (a', b, c') all exist
            # i.e., c' must be in the intersection of:
            #   thirds of (a,b), (a',b'), (a,b'), (a',b)
            
            ab_thirds = pair_to_thirds.get((min(a, b), max(a, b)), set())
            ap_bp_thirds = pair_to_thirds.get((min(a_prime, b_prime), max(a_prime, b_prime)), set())
            a_bp_thirds = pair_to_thirds.get((min(a, b_prime), max(a, b_prime)), set())
            ap_b_thirds = pair_to_thirds.get((min(a_prime, b), max(a_prime, b)), set())
            
            # c' must be in all four sets, and not in {a, b, c, a_prime, b_prime}
            candidate_cp = ab_thirds & ap_bp_thirds & a_bp_thirds & ap_b_thirds
            candidate_cp -= {a, b, c, a_prime, b_prime}
            
            if candidate_cp:
                return False
    
    return True

def _greedy_construct(deadline, seed):
    """Greedy construction with randomization."""
    random.seed(seed)
    indices = list(range(NUM_TRIPLES))
    random.shuffle(indices)
    
    present = [False] * NUM_TRIPLES
    pair_to_thirds = {}  # (x,y) -> set of z such that {x,y,z} is present
    
    count = 0
    for idx in indices:
        if time.time() > deadline:
            break
        
        if _check_validity_fast(idx, present, pair_to_thirds):
            present[idx] = True
            count += 1
            a, b, c = ALL_TRIPLES[idx]
            for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                key = (min(x, y), max(x, y))
                if key not in pair_to_thirds:
                    pair_to_thirds[key] = set()
                pair_to_thirds[key].add(z)
    
    return [ALL_TRIPLES[i] for i in range(NUM_TRIPLES) if present[i]]

def _local_search(triples, deadline):
    """Local search: try to add more triples, and try swaps."""
    present = [False] * NUM_TRIPLES
    for t in triples:
        present[TRIPLE_INDEX[t]] = True
    
    pair_to_thirds = {}
    for i in range(NUM_TRIPLES):
        if present[i]:
            a, b, c = ALL_TRIPLES[i]
            for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                key = (min(x, y), max(x, y))
                if key not in pair_to_thirds:
                    pair_to_thirds[key] = set()
                pair_to_thirds[key].add(z)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any valid triple
        for idx in range(NUM_TRIPLES):
            if present[idx]:
                continue
            if _check_validity_fast(idx, present, pair_to_thirds):
                present[idx] = True
                a, b, c = ALL_TRIPLES[idx]
                for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                    key = (min(x, y), max(x, y))
                    if key not in pair_to_thirds:
                        pair_to_thirds[key] = set()
                    pair_to_thirds[key].add(z)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try swaps - remove one triple, try to add two
        if not improved:
            present_indices = [i for i in range(NUM_TRIPLES) if present[i]]
            random.shuffle(present_indices)
            
            for idx_remove in present_indices:
                if time.time() > deadline:
                    break
                
                present[idx_remove] = False
                a, b, c = ALL_TRIPLES[idx_remove]
                for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                    key = (min(x, y), max(x, y))
                    if key in pair_to_thirds:
                        pair_to_thirds[key].discard(z)
                        if not pair_to_thirds[key]:
                            del pair_to_thirds[key]
                
                added_count = 0
                added_indices = []
                for idx_add in range(NUM_TRIPLES):
                    if present[idx_add]:
                        continue
                    if _check_validity_fast(idx_add, present, pair_to_thirds):
                        present[idx_add] = True
                        a2, b2, c2 = ALL_TRIPLES[idx_add]
                        for x, y, z in [(a2, b2, c2), (a2, c2, b2), (b2, c2, a2)]:
                            key = (min(x, y), max(x, y))
                            if key not in pair_to_thirds:
                                pair_to_thirds[key] = set()
                            pair_to_thirds[key].add(z)
                        added_count += 1
                        added_indices.append(idx_add)
                
                if added_count >= 2:
                    improved = True
                    break
                else:
                    for idx in added_indices:
                        present[idx] = False
                        a2, b2, c2 = ALL_TRIPLES[idx]
                        for x, y, z in [(a2, b2, c2), (a2, c2, b2), (b2, c2, a2)]:
                            key = (min(x, y), max(x, y))
                            if key in pair_to_thirds:
                                pair_to_thirds[key].discard(z)
                                if not pair_to_thirds[key]:
                                    del pair_to_thirds[key]
                    present[idx_remove] = True
                    a, b, c = ALL_TRIPLES[idx_remove]
                    for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                        key = (min(x, y), max(x, y))
                        if key not in pair_to_thirds:
                            pair_to_thirds[key] = set()
                        pair_to_thirds[key].add(z)
    
    return [ALL_TRIPLES[i] for i in range(NUM_TRIPLES) if present[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_result = []
    best_count = 0
    num_attempts = 20
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        result = _greedy_construct(deadline, seed=i * 7919 + 104729)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    if time.time() < deadline - 5:
        best_result = _local_search(best_result, deadline)
    
    return best_result
