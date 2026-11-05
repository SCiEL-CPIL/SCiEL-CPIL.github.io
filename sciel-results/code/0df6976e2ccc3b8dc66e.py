import time
import random

N = 256

def construct():
    best_set = []
    best_size = float('inf')
    deadline = time.time() + 55.0
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        for x in candidates:
            # Fast check
            # Generate new sums
            # Using a list comprehension might be faster than set comprehension for small lists?
            # Or just a loop?
            # Set comprehension is fine.
            
            # Optimization: Instead of creating a new set every time,
            # we can check each sum individually.
            # But we need to add them all if valid.
            
            valid = True
            new_sums = []
            for a in current_set:
                s = x + a
                if s in current_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                two_x = 2 * x
                if two_x in current_sums:
                    valid = False
                else:
                    # Check if two_x collides with any new_sum?
                    # two_x == x + a => x == a. Impossible.
                    # So just add.
                    current_set.append(x)
                    current_sums.update(new_sums)
                    current_sums.add(two_x)
                    
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return best_set
