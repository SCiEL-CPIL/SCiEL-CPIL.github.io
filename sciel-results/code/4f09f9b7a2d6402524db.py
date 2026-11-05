import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Universe
    universe = list(range(1, 23))
    
    def is_dissociated(S):
        """Check if all subset sums of reciprocals of S are distinct."""
        if not S:
            return True
        # Generate all subset sums
        sums = set()
        sums.add(Fraction(0))
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, x))
            sums.update(new_sums)
            # Early termination if collision found? 
            # If len(sums) < 2**len(current_subset), collision.
            # But we build incrementally.
            # Actually, just checking uniqueness at the end is O(2^k).
            # For k=10, 1024 sums. Fast.
        # Check uniqueness
        # sums is a set, so if we added 2^k items, len must be 2^k
        expected = 1 << len(S)
        return len(sums) == expected

    def try_add(S, x):
        """Check if S union {x} is dissociated."""
        new_S = list(S)
        new_S.append(x)
        return is_dissociated(new_S)

    # Strategy 1: Greedy Superincreasing
    # Sort by value descending (reciprocal ascending)
    candidates = sorted(universe, reverse=True)
    greedy_set = []
    for x in candidates:
        if try_add(greedy_set, x):
            greedy_set.append(x)
            
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Strategy 2: Local Search / Greedy with different orders
    # Try to add elements from the remaining ones to the greedy set
    remaining = [x for x in universe if x not in greedy_set]
    
    # Try adding any remaining element that maintains validity
    added = True
    while added:
        added = False
        for x in remaining[:]:
            if try_add(best_set, x):
                best_set.append(x)
                remaining.remove(x)
                added = True
                best_size = len(best_set)
                if time.time() - start_time > timeout:
                    return best_set
    
    # Strategy 3: Randomized Restart
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() - start_time > timeout:
            break
        current_set = []
        # Random order
        random.shuffle(candidates)
        for x in candidates:
            if try_add(current_set, x):
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
            # Try to expand
            remaining = [x for x in universe if x not in current_set]
            added = True
            while added:
                added = False
                for x in remaining[:]:
                    if try_add(best_set, x):
                        best_set.append(x)
                        remaining.remove(x)
                        added = True
                        best_size = len(best_set)
                        if time.time() - start_time > timeout:
                            return best_set

    return best_set
