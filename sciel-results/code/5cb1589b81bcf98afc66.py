import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s
    
    def run_greedy(order):
        """Try to build a valid set by adding elements in the given order."""
        selected = []
        current_sums = set()
        
        for x in order:
            if x in selected:
                continue
            # Check if adding x creates any collision
            # New triples involving x:
            # (x,x,x), (x,x,y) for y in selected, (x,y,z) for y<=z in selected
            
            # Precompute: for each y in selected, check x+x+y and x+y+z
            # We need to check all new sums against current_sums AND among themselves
            
            new_sums = []
            
            # (x, x, x)
            s = 3 * x
            new_sums.append(s)
            
            # (x, x, y) for each y in selected
            for y in selected:
                s = 2 * x + y
                new_sums.append(s)
            
            # (x, y, z) for y <= z in selected
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    s = x + selected[i] + selected[j]
                    new_sums.append(s)
            
            # Check for duplicates within new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
            
            # Check collision with existing sums
            new_sum_set = set(new_sums)
            if new_sum_set & current_sums:
                continue
            
            # All good, add x
            selected.append(x)
            current_sums.update(new_sum_set)
            
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
        
        return selected
    
    best_set = []
    best_size = 0
    
    # First: deterministic sequential greedy
    seq_order = list(range(1, 4097))
    result = run_greedy(seq_order)
    if len(result) > best_size:
        best_set = result
        best_size = len(result)
    
    if time.time() - start_time > time_limit - 10:
        return best_set
    
    # Now try randomized approaches
    # Strategy: random permutations, but biased toward certain ranges
    
    trials = 0
    while time.time() - start_time < time_limit - 2:
        trials += 1
        
        # Different strategies for ordering
        strategy = trials % 5
        
        if strategy == 0:
            # Fully random permutation
            order = list(range(1, 4097))
            random.shuffle(order)
        elif strategy == 1:
            # Random but start with small numbers, then random
            small = list(range(1, 100))
            rest = list(range(100, 4097))
            random.shuffle(rest)
            order = small + rest
        elif strategy == 2:
            # Exponential spacing bias
            # Generate candidates with exponential gaps
            order = []
            x = 1
            while x <= 4096:
                order.append(x)
                x += random.randint(1, 50)
            random.shuffle(order)
        elif strategy == 3:
            # Random subset, sorted
            subset = random.sample(range(1, 4097), min(200, 4096))
            order = sorted(subset)
        else:
            # Mixed: some sequential, some random
            base = list(range(1, 200))
            extra = random.sample(range(200, 4097), 100)
            order = base + extra
            random.shuffle(order[200:])  # Keep first part somewhat ordered
        
        result = run_greedy(order)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    return best_set
