"""Rewritten program for sunflower-free 4-uniform family on [1,16]."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _sunflower_conflict_bitmask(new_bits, existing_bits_list, pair_intersections):
    """
    Check if new_bits forms a sunflower with any pair in existing.
    Returns True if conflict exists.
    
    existing_bits_list: list of bitmasks for existing sets
    pair_intersections: dict mapping (i,j) -> intersection bitmask (precomputed)
    """
    for i in range(len(existing_bits_list)):
        a = existing_bits_list[i]
        for j in range(i + 1, len(existing_bits_list)):
            b = existing_bits_list[j]
            # Compute intersections
            ab = a & b
            an = a & new_bits
            bn = b & new_bits
            if ab == an == bn:
                return True
    return False

def _sunflower_conflict_fast(new_bits, existing_bits_list, n_existing):
    """
    Faster conflict check using precomputed pair intersections stored in a flat structure.
    """
    # For each pair (i,j), we need to check if (a&b) == (a&new) == (b&new)
    # a&new and b&new can be computed quickly
    # But we still need to iterate over all pairs
    
    # Alternative: for the new set S, group existing sets by their intersection with S.
    # If two existing sets A, B both have A∩S = I and B∩S = I, and A∩B = I, then sunflower.
    # Actually we need A∩B = A∩S = B∩S.
    
    # Let's use the approach: for each element pattern of intersection with new_bits
    # Actually, let's just do the O(n^2) check but optimized
    
    for i in range(n_existing):
        a = existing_bits_list[i]
        an = a & new_bits
        for j in range(i + 1, n_existing):
            b = existing_bits_list[j]
            bn = b & new_bits
            if an == bn:
                # Check if a & b == an
                if (a & b) == an:
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Represent sets as bitmasks for speed
    # Element i (1-indexed) -> bit (i-1)
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_total = len(all_sets)  # C(16,4) = 1820
    
    best_family = []
    best_size = 0
    
    # Strategy: multiple random greedy passes with different orderings
    # Also try a smarter ordering: sort by element frequency or use local search
    
    # First, do a few random greedy passes
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Lexicographic (natural) order
            order = list(range(n_total))
        elif attempt == 1:
            # Reverse lexicographic
            order = list(range(n_total - 1, -1, -1))
        else:
            order = list(range(n_total))
            random.shuffle(order)
        
        family_bits = []
        n_fam = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            s = all_sets[idx]
            
            # Quick check: does it conflict with any pair in family?
            conflict = False
            for i in range(n_fam):
                a = family_bits[i]
                an = a & s
                for j in range(i + 1, n_fam):
                    b = family_bits[j]
                    bn = b & s
                    if an == bn and (a & b) == an:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                family_bits.append(s)
                n_fam += 1
        
        if n_fam > best_size:
            best_size = n_fam
            best_family = family_bits[:]
        
        if best_size >= 250:
            break
    
    # Local search / hill climbing on the best family found
    if best_size > 0 and time.time() < deadline - 5:
        # Try to improve by swapping or adding/removing
        # Represent family as a set of indices for fast membership
        family_set = set(best_family)
        
        # Try adding sets that weren't included
        candidates = [all_sets[i] for i in range(n_total) if all_sets[i] not in family_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for cand in candidates:
                if time.time() > deadline - 3:
                    break
                
                # Check if we can add cand
                conflict = False
                fam_list = list(family_set)
                n_fam = len(fam_list)
                for i in range(n_fam):
                    a = fam_list[i]
                    an = a & cand
                    for j in range(i + 1, n_fam):
                        b = fam_list[j]
                        bn = b & cand
                        if an == bn and (a & b) == an:
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    family_set.add(cand)
                    improved = True
                    break
        
        if len(family_set) > best_size:
            best_size = len(family_set)
            best_family = list(family_set)
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
