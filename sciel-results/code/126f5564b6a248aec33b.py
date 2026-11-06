"""Program to find max-cardinality dissociated reciprocal set from [1,22]."""
import random
import itertools

def construct():
    """Return a nonempty subset of [1,22] maximizing cardinality such that
    all reciprocal subset sums (including empty sum = 0) are distinct."""
    
    def is_dissociated(S):
        """Check if all subset sums of {1/x for x in S} are distinct."""
        if not S:
            return True
        n = len(S)
        if n > 20:
            # Too many to check exhaustively, use probabilistic check
            return False
        # Compute all 2^n subset sums
        sums = [0.0]
        for x in S:
            inv = 1.0 / x
            sums = sums + [s + inv for s in sums]
        # Check for duplicates using sorted and comparing
        # Use a set with careful float handling - use rounding to avoid FP issues
        # Actually, for small n, exact comparison via fractions would be better
        # But for performance, let's use a tolerance-free approach:
        # Convert to tuples of integers by using common denominator
        # For sets up to size ~12, this is feasible
        from math import gcd
        from functools import reduce
        # Use Fraction for exact arithmetic
        from fractions import Fraction
        frac_sums = [Fraction(0)]
        for x in S:
            inv = Fraction(1, x)
            frac_sums = frac_sums + [f + inv for f in frac_sums]
        if len(frac_sums) != len(set(frac_sums)):
            return False
        return True
    
    def is_dissociated_fast(S):
        """Faster check using floating point with careful comparison."""
        if not S:
            return True
        n = len(S)
        if n > 18:
            return False  # Too many to check
        # Use Fraction for correctness
        from fractions import Fraction
        frac_sums = [Fraction(0)]
        for x in S:
            inv = Fraction(1, x)
            frac_sums = frac_sums + [f + inv for f in frac_sums]
        return len(frac_sums) == len(set(frac_sums))
    
    # Greedy construction with randomization
    best_set = None
    best_size = 0
    
    candidates = list(range(1, 23))
    
    for trial in range(500):
        random.shuffle(candidates)
        current = []
        for x in candidates:
            new_set = current + [x]
            if is_dissociated_fast(new_set):
                current = new_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 12:  # Good enough, but keep trying for more
                pass
    
    # Local search: try to add elements not in best_set
    if best_set:
        changed = True
        while changed:
            changed = False
            remaining = [x for x in range(1, 23) if x not in best_set]
            random.shuffle(remaining)
            for x in remaining:
                new_set = best_set + [x]
                if is_dissociated_fast(new_set):
                    best_set = new_set
                    changed = True
                    break
    
    # Try swapping: replace one element with another to potentially allow more additions
    if best_set:
        for _ in range(200):
            trial_set = best_set[:]
            # Try to add an element
            remaining = [x for x in range(1, 23) if x not in trial_set]
            random.shuffle(remaining)
            added = False
            for x in remaining:
                new_set = trial_set + [x]
                if is_dissociated_fast(new_set):
                    trial_set = new_set
                    added = True
                    break
            
            if not added and len(trial_set) < 22:
                # Try swapping: remove one, add another
                if trial_set:
                    idx = random.randint(0, len(trial_set) - 1)
                    removed = trial_set[idx]
                    trial_without = trial_set[:idx] + trial_set[idx+1:]
                    for x in remaining:
                        new_set = trial_without + [x]
                        if is_dissociated_fast(new_set):
                            trial_set = new_set
                            # Now try to add more
                            changed2 = True
                            while changed2:
                                changed2 = False
                                rem2 = [y for y in range(1, 23) if y not in trial_set]
                                random.shuffle(rem2)
                                for y in rem2:
                                    new_set2 = trial_set + [y]
                                    if is_dissociated_fast(new_set2):
                                        trial_set = new_set2
                                        changed2 = True
                                        break
                            break
            
            if len(trial_set) > len(best_set):
                best_set = trial_set
                if len(best_set) >= 13:
                    break
    
    if not best_set:
        best_set = [1]
    
    return sorted(best_set)
