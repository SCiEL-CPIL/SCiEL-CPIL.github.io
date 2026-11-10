import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute reciprocals
    recips = [Fraction(0, 1)] + [Fraction(1, i) for i in range(1, 25)]
    
    best_set = []
    best_size = 0
    
    # Candidates sorted descending to find large sets early?
    # Actually, order doesn't strictly matter for correctness, but for pruning,
    # finding a good solution early helps.
    # Descending is a common heuristic for packing.
    candidates = list(range(24, 0, -1))
    
    def backtrack(idx, current_set, achievable_sums, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        # Pruning: even if we take all remaining, we can't beat best_size
        remaining = len(candidates) - idx
        if current_size + remaining <= best_size:
            return
            
        if idx == len(candidates):
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        x = candidates[idx]
        target = recips[x]
        
        # Option 1: Include x
        # Check if 1/x is in achievable_sums
        if target not in achievable_sums:
            # Add x
            new_set = current_set + [x]
            # Update achievable sums
            # We create a new set to avoid modifying the parent's set
            new_sums = set(achievable_sums)
            for s in achievable_sums:
                new_sums.add(s + target)
                
            backtrack(idx + 1, new_set, new_sums, current_size + 1)
            
        # Option 2: Exclude x
        backtrack(idx + 1, current_set, achievable_sums, current_size)
        
    backtrack(0, [], {Fraction(0, 1)}, 0)
    
    if not best_set:
        return [24]
        
    return best_set
