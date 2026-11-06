"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_subset_sums_recip(elements):
    """
    Compute all subset sums of 1/x for x in elements.
    Returns a set of floats (or use fractions for exactness, but floats with tolerance
    might be risky. Let's use integers scaled by LCM? No, LCM of 1..22 is huge.
    Let's use Python's `fractions.Fraction` for exact arithmetic to avoid floating point issues.
    """
    from fractions import Fraction
    sums = {Fraction(0, 1)}
    for x in elements:
        inv = Fraction(1, x)
        new_sums = set()
        for s in sums:
            new_sums.add(s + inv)
        sums.update(new_sums)
    return sums

def _is_dissociated(elements):
    """
    Check if all subset sums of reciprocals are distinct.
    This is equivalent to: the set of elements is dissociated in the additive group of rationals.
    For a set S, all 2^|S| subset sums are distinct if and only if no non-empty subset has sum 0
    AND no two distinct subsets have the same sum.
    Actually, the condition "all subset sums are distinct" is exactly that the map from subsets to sums is injective.
    This is equivalent to: for any two distinct subsets A, B, sum(1/x for x in A) != sum(1/x for x in B).
    This is equivalent to: sum(1/x for x in A \ B) != sum(1/x for x in B \ A).
    Let C = A \ B and D = B \ A. C and D are disjoint.
    So we need to ensure that for any disjoint non-empty subsets C, D of S, sum_{c in C} 1/c != sum_{d in D} 1/d.
    
    A simpler check: generate all subset sums. If the number of unique sums is 2^k where k is len(elements), it's dissociated.
    """
    from fractions import Fraction
    if not elements:
        return True
    sums = {Fraction(0, 1)}
    for x in elements:
        inv = Fraction(1, x)
        # Optimization: if at any point the number of sums is less than 2 * previous count, we have a collision
        prev_count = len(sums)
        new_sums = set()
        for s in sums:
            new_sums.add(s + inv)
        if len(new_sums) < prev_count:
            return False
        sums.update(new_sums)
        # If we have 2^k sums, it's dissociated so far.
        # Actually, we just need to check if the total count at the end is 2^k.
        # But we can early exit if we detect a collision.
        # A collision happens if s1 + inv == s2 + inv for s1 != s2 (impossible)
        # OR s1 + inv == s2 for s1 != s2.
        # So we just need to check if any new sum was already in `sums`.
        # Wait, `new_sums` is a set, so duplicates within new_sums are removed.
        # But we need to check if `new_sums` intersects `sums`.
        # If `len(sums | new_sums) < len(sums) + len(new_sums)`, there's a collision.
        # Since `new_sums` size is at most `len(sums)`, if `len(new_sums) < len(sums)`, collision.
        # If `len(new_sums) == len(sums)`, we still need to check intersection.
        
    return len(sums) == (1 << len(elements))

def _is_dissociated_fast(elements):
    """
    Faster check using bitsets or similar?
    With k up to ~12, 2^k is 4096. Fraction arithmetic is slow.
    Let's stick to Fraction but optimize the loop.
    """
    from fractions import Fraction
    sums = {Fraction(0, 1)}
    for x in elements:
        inv = Fraction(1, x)
        new_sums = {s + inv for s in sums}
        # If any element in new_sums is in sums, we have a collision.
        # Because sums are distinct so far, and new_sums are sums+inv.
        # If (s1 + inv) == s2, then s1 - s2 = -inv.
        # This is a linear relation.
        # Checking intersection is O(n).
        if sums & new_sums:
            return False
        sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s
    
    # Known dissociated sets:
    # Powers of 2: {1, 2, 4, 8, 16} -> size 5
    # {1, 2, 3, 5, 7, 11, 13, 17, 19} -> size 9 (from seed)
    # Let's try to find a larger one.
    
    # Strategy: Greedy construction with backtracking or local search.
    # Since the space is small (22 elements), we can try to build a large set.
    # A set S is dissociated if and only if no non-trivial linear combination with coefficients in {-1, 0, 1} sums to 0.
    
    # Let's use a randomized local search / hill climbing approach.
    # Start with a known valid set, try to add elements, or swap elements.
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Verify best_set
    if not _is_dissociated_fast(best_set):
        # Fallback to smaller known valid set
        best_set = [1, 2, 4, 8, 16]
    
    # Try to extend best_set by adding elements from [1, 22] \ best_set
    # We can also try to replace elements to allow adding more.
    
    # Let's do a randomized search.
    # 1. Start with empty set.
    # 2. Greedily add elements in random order, if valid.
    # 3. Try to add more elements by removing one and adding two, etc.
    
    all_elements = list(range(1, 23))
    
    # Pre-compute reciprocals as Fractions
    recips = {x: __import__('fractions').Fraction(1, x) for x in all_elements}
    
    def check_validity(set_list):
        # Use the fast check
        return _is_dissociated_fast(set_list)
    
    # Local search
    current_set = set(best_set)
    best_size = len(current_set)
    
    # Try to add elements
    while time.time() < deadline:
        # Randomize order of candidates
        candidates = [x for x in all_elements if x not in current_set]
        random.shuffle(candidates)
        
        added = False
        for x in candidates:
            if time.time() > deadline:
                break
            new_set = current_set | {x}
            if check_validity(list(new_set)):
                current_set = new_set
                best_size = len(current_set)
                added = True
                break
        
        if not added:
            # Try to swap: remove one, add one
            # Or remove one, add two
            removed = False
            for x in list(current_set):
                if time.time() > deadline:
                    break
                temp_set = current_set - {x}
                
                # Try adding one
                candidates = [y for y in all_elements if y not in temp_set]
                random.shuffle(candidates)
                for y in candidates:
                    if time.time() > deadline:
                        break
                    new_set = temp_set | {y}
                    if check_validity(list(new_set)):
                        current_set = new_set
                        removed = True
                        break
                if removed:
                    break
                
                # Try adding two
                if not removed:
                    candidates = [y for y in all_elements if y not in temp_set]
                    random.shuffle(candidates)
                    # Only try a few pairs to save time
                    for i in range(min(10, len(candidates))):
                        for j in range(i+1, min(10, len(candidates))):
                            if time.time() > deadline:
                                break
                            y1, y2 = candidates[i], candidates[j]
                            new_set = temp_set | {y1, y2}
                            if check_validity(list(new_set)):
                                current_set = new_set
                                removed = True
                                break
                        if removed:
                            break
                if removed:
                    break
            
            if not removed:
                # Stuck in local optimum, try a random restart with a different initial set
                # or just break if we've been stuck for a while?
                # For now, let's just try a few random restarts.
                # Pick a random subset of size best_size
                # Actually, let's just try to build from scratch greedily with random order.
                # If we find a larger set, update.
                # This might be slow, so limit it.
                pass # We are already in a loop, so we will keep trying swaps.
        
        # If we reached max possible size (which is unlikely to be > 12 or so), we can stop.
        # Max dissociated set in [1, N] is related to the number of distinct subset sums.
        # For N=22, the max size is likely around 10-12.
        # If we find size 12, we can probably stop.
        if best_size >= 12:
            break

    return sorted(current_set)

# EVOLVE-BLOCK-END
