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
    """Check if adding triple triple_idx creates K(2,2,2)."""
    a, b, c = ALL_TRIPLES[triple_idx]
    
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        
        # (a', b, c) must be present
        ap_b_c = (min(a_prime, b), max(a_prime, b))
        if c not in pair_to_thirds.get(ap_b_c, set()):
            continue
            
        # (a', b, c) is present.
        # We need b' and c'.
        # b' must be such that (a, b', c) and (a', b', c) are present.
        # Let's iterate b'
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            
            # Check (a, b', c)
            a_bp_c = (min(a, b_prime), max(a, b_prime))
            if c not in pair_to_thirds.get(a_bp_c, set()):
                continue
            
            # Check (a', b', c)
            ap_bp_c = (min(a_prime, b_prime), max(a_prime, b_prime))
            if c not in pair_to_thirds.get(ap_bp_c, set()):
                continue
            
            # Now we have a', b' candidates.
            # We need c' such that:
            # (a, b, c'), (a', b', c'), (a, b', c'), (a', b, c') are present.
            # c' is in:
            # S_ab, S_a'b', S_ab', S_a'b
            
            ab_thirds = pair_to_thirds.get((min(a, b), max(a, b)), set())
            ap_bp_thirds = pair_to_thirds.get((min(a_prime, b_prime), max(a_prime, b_prime)), set())
            a_bp_thirds = pair_to_thirds.get((min(a, b_prime), max(a, b_prime)), set())
            ap_b_thirds = pair_to_thirds.get((min(a_prime, b), max(a_prime, b)), set())
            
            # Intersection
            # Optimization: Start with smallest set
            sets = [ab_thirds, ap_bp_thirds, a_bp_thirds, ap_b_thirds]
            # Filter out empty
            valid_sets = [s for s in sets if s]
            if not valid_sets:
                continue
                
            # Sort by size
            valid_sets.sort(key=len)
            candidate_cp = set(valid_sets[0])
            for s in valid_sets[1:]:
                candidate_cp &= s
                if not candidate_cp:
                    break
            
            if candidate_cp:
                # Ensure c' is distinct
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
    pair_to_thirds = {}
    
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
    """Local search with 1-for-2 and 2-for-3 swaps."""
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
        
        # Phase 2: Try swaps
        if not improved:
            present_indices = [i for i in range(NUM_TRIPLES) if present[i]]
            random.shuffle(present_indices)
            
            # Try 1-for-2
            swap_success = False
            for idx_remove in present_indices:
                if time.time() > deadline:
                    break
                
                # Save state
                present[idx_remove] = False
                a, b, c = ALL_TRIPLES[idx_remove]
                removed_keys = []
                for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                    key = (min(x, y), max(x, y))
                    if key in pair_to_thirds:
                        pair_to_thirds[key].discard(z)
                        if not pair_to_thirds[key]:
                            del pair_to_thirds[key]
                        removed_keys.append((key, z))
                
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
                            break
                
                if added_count >= 2:
                    improved = True
                    swap_success = True
                    break
                else:
                    # Revert
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
                    for key, z in removed_keys:
                        if key not in pair_to_thirds:
                            pair_to_thirds[key] = set()
                        pair_to_thirds[key].add(z)
            
            # Try 2-for-3 if 1-for-2 failed
            if not swap_success and time.time() < deadline:
                # Pick 2 random triples to remove
                if len(present_indices) >= 2:
                    idx1, idx2 = random.sample(present_indices, 2)
                    
                    # Remove
                    removed_data = []
                    for idx in [idx1, idx2]:
                        present[idx] = False
                        a, b, c = ALL_TRIPLES[idx]
                        keys = []
                        for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                            key = (min(x, y), max(x, y))
                            if key in pair_to_thirds:
                                pair_to_thirds[key].discard(z)
                                if not pair_to_thirds[key]:
                                    del pair_to_thirds[key]
                                keys.append((key, z))
                        removed_data.append((idx, keys))
                    
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
                            if added_count >= 3:
                                break
                    
                    if added_count >= 3:
                        improved = True
                        # Keep changes
                    else:
                        # Revert
                        for idx in added_indices:
                            present[idx] = False
                            a2, b2, c2 = ALL_TRIPLES[idx]
                            for x, y, z in [(a2, b2, c2), (a2, c2, b2), (b2, c2, a2)]:
                                key = (min(x, y), max(x, y))
                                if key in pair_to_thirds:
                                    pair_to_thirds[key].discard(z)
                                    if not pair_to_thirds[key]:
                                        del pair_to_thirds[key]
                        for idx, keys in removed_data:
                            present[idx] = True
                            for key, z in keys:
                                if key not in pair_to_thirds:
                                    pair_to_thirds[key] = set()
                                pair_to_thirds[key].add(z)
    
    return [ALL_TRIPLES[i] for i in range(NUM_TRIPLES) if present[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
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
