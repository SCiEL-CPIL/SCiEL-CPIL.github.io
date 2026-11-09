"""Improved program using bitmask representation and local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1,16] as bitmasks
_ALL_SETS = []
_SET_TO_IDX = {}
for _combo in combinations(range(16), 4):
    _mask = 0
    for _v in _combo:
        _mask |= (1 << _v)
    _ALL_SETS.append(_mask)
    _SET_TO_IDX[_mask] = len(_ALL_SETS) - 1
_N_SETS = len(_ALL_SETS)

# Precompute popcount for all possible masks (0 to 2^16-1)
_POPCOUNT = [0] * (1 << 16)
for _i in range(1, 1 << 16):
    _POPCOUNT[_i] = _POPCOUNT[_i >> 1] + (_i & 1)

def _is_sunflower_triple_bm(a, b, c):
    """Check if three bitmask sets form a sunflower using bit operations."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(family_bitmasks, new_bm):
    """Check if adding new_bm to family_bitmasks creates no sunflower triple."""
    for i in range(len(family_bitmasks)):
        for j in range(i + 1, len(family_bitmasks)):
            if _is_sunflower_triple_bm(family_bitmasks[i], family_bitmasks[j], new_bm):
                return False
    return True

def _greedy_construct_bm(order, deadline):
    """Greedy construction using bitmask representation."""
    chosen = []
    for bm in order:
        if time.time() > deadline:
            break
        if _can_add(chosen, bm):
            chosen.append(bm)
    return chosen

def _local_search(family, deadline, rng):
    """Local search: try to swap out one set and in another to improve family size.
    Actually, we can only add if possible. So local search tries:
    1. For each set in family, try removing it and then greedily adding more.
    2. Try swapping a set with a non-member.
    """
    family = list(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(family)
        family_set = set(family)
        
        # Try removing each element and re-adding greedily
        for i in range(n):
            if time.time() > deadline:
                break
            candidate = family[:i] + family[i+1:]
            candidate_set = set(candidate)
            
            # Try to add back from remaining sets in a smart order
            # Sort remaining by how many conflicts they have with candidate
            remaining = [bm for bm in _ALL_SETS if bm not in candidate_set]
            
            # Score each remaining set by number of sunflower conflicts with candidate pairs
            def conflict_count(bm):
                count = 0
                for a in candidate:
                    for b in candidate:
                        if a != b and _is_sunflower_triple_bm(a, b, bm):
                            count += 1
                return count
            
            # Sort by conflict count (fewer conflicts = better)
            remaining.sort(key=conflict_count)
            
            new_family = list(candidate)
            for bm in remaining:
                if time.time() > deadline:
                    break
                if _can_add(new_family, bm):
                    new_family.append(bm)
            
            if len(new_family) > len(family):
                family = new_family
                improved = True
                break
    
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 90s
    
    best_family = []
    best_size = 0
    
    rng = random.Random(12345)
    
    # Multiple random greedy attempts with local search
    attempt = 0
    while time.time() < deadline - 10:  # Leave 10s for final local search
        attempt += 1
        order = _ALL_SETS[:]
        rng.shuffle(order)
        
        family = _greedy_construct_bm(order, deadline - 5)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
        
        # Apply local search if we have time
        if time.time() < deadline - 5:
            improved_family = _local_search(family, deadline - 5, rng)
            if len(improved_family) > best_size:
                best_size = len(improved_family)
                best_family = list(improved_family)
        
        if best_size >= 250:
            break
    
    # Final local search on best family
    if time.time() < deadline:
        final_family = _local_search(best_family, deadline, rng)
        if len(final_family) > best_size:
            best_family = final_family
    
    # Convert bitmasks back to sorted lists of 1-based elements
    result = []
    for bm in best_family:
        elems = []
        for bit in range(16):
            if bm & (1 << bit):
                elems.append(bit + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
