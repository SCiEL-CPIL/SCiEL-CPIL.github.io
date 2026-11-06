"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    # We need a nonempty subset of [1,24]
    # Constraint: for any selected element a, 1/a != sum of 1/b for b in subset of OTHER selected elements
    # Maximize cardinality
    
    # The seed returns list(range(13, 25)) which is [13, 14, ..., 24] (12 elements)
    # Let's verify this is valid and try to improve
    
    # First, let's check if the seed solution is valid
    def is_valid(subset):
        s = set(subset)
        for a in subset:
            target = Fraction(1, a)
            others = [b for b in subset if b != a]
            # Check all subsets of others
            # For small sets, we can enumerate all subsets
            # But for larger sets, we need to be smarter
            # Let's use a different approach: check if target can be formed
            # by summing reciprocals of a subset of others
            
            # We can use dynamic programming or just enumerate if the set is small
            # For a set of size up to ~20, enumerating all 2^n subsets is too slow
            # Let's use a smarter check
            
            # Actually, for the constraint to be violated, we need:
            # 1/a = sum of 1/b_i for some subset of others
            # This is equivalent to: a * sum(1/b_i) = 1
            # Or: sum(a/b_i) = 1
            
            # Let's check if any subset of others sums to 1/a
            # We can use meet-in-the-middle or just check small subsets
            
            # For efficiency, let's check subsets of size 1, 2, 3, ... up to some limit
            # Actually, let's just enumerate all subsets if the set is small enough
            
            n = len(others)
            if n <= 20:
                # Enumerate all non-empty subsets
                for r in range(1, n + 1):
                    for combo in itertools.combinations(others, r):
                        s_val = sum(Fraction(1, b) for b in combo)
                        if s_val == target:
                            return False
            else:
                # For larger sets, use a heuristic check
                # Check subsets of size up to 4
                for r in range(1, 5):
                    for combo in itertools.combinations(others, r):
                        s_val = sum(Fraction(1, b) for b in combo)
                        if s_val == target:
                            return False
                        # Early termination: if sum already exceeds target
                        if s_val > target:
                            break
            # Continue checking other elements
        return True
    
    # Start with the seed solution
    best = list(range(13, 25))
    best_size = len(best)
    
    # Try to add elements from 1-12
    # We want to maximize cardinality, so try to add more elements
    
    # Let's try a greedy approach: start with empty set and add elements
    # that don't violate the constraint, prioritizing larger elements first
    # (larger elements have smaller reciprocals, making it harder to form sums)
    
    # Actually, let's try to find the maximum valid subset
    # Start with all 24 elements and try to remove the minimum number
    
    # First, let's check if [13, 14, ..., 24] is valid
    if is_valid(best):
        # Try to add elements from 1-12
        for candidate in range(1, 13):
            if time.time() > deadline - 5:
                break
            new_set = best + [candidate]
            if is_valid(new_set):
                best = new_set
                best_size = len(best)
    
    # Try other strategies
    # Strategy 2: Start with larger elements and try to add smaller ones
    # Let's try all subsets of a certain size and find the largest valid one
    
    # For efficiency, let's try sizes from 24 down to best_size + 1
    # But that's too many combinations
    
    # Let's try a randomized approach with local search
    import random
    
    best_set = set(best)
    
    # Try to improve by swapping elements
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        if len(best_set) >= 24:
            break
        
        # Try adding a random element from 1-12
        candidates = [x for x in range(1, 13) if x not in best_set]
        if not candidates:
            break
        candidate = random.choice(candidates)
        new_set = list(best_set) + [candidate]
        if is_valid(new_set):
            best_set.add(candidate)
    
    best = list(best_set)
    
    # Final validation
    if not is_valid(best):
        # Fall back to seed
        best = list(range(13, 25))
    
    return best
# EVOLVE-BLOCK-END
