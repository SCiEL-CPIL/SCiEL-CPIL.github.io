"""Find maximum dissociated reciprocal subset of [1,22]."""
import time
from fractions import Fraction
import random

def construct():
    deadline = time.time() + 80
    
    def is_valid(elements):
        """Check if all subset sums of reciprocals are distinct."""
        if len(elements) == 0:
            return True
        seen = {Fraction(0, 1)}
        sums = [Fraction(0, 1)]
        for elem in elements:
            recip = Fraction(1, elem)
            new_sums = []
            for s in sums:
                new_s = s + recip
                if new_s in seen:
                    return False
                seen.add(new_s)
                new_sums.append(new_s)
            sums.extend(new_sums)
        return True
    
    candidates = list(range(1, 23))
    best_set = [1]
    best_size = 1
    
    random.seed(42)
    
    # Greedy with multiple random orderings
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        order = candidates[:]
        if trial == 0:
            pass  # increasing
        elif trial == 1:
            order = order[::-1]  # decreasing
        else:
            random.shuffle(order)
        
        current = []
        for elem in order:
            trial_set = current + [elem]
            if is_valid(trial_set):
                current = trial_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Try to extend best_set
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        extended = best_set[:]
        for elem in candidates:
            if elem not in extended:
                trial = extended + [elem]
                if is_valid(trial):
                    extended = trial
        if len(extended) > best_size:
            best_size = len(extended)
            best_set = extended
    
    return best_set
