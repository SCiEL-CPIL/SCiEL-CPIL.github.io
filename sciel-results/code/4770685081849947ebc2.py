import itertools
import math
import random
import time

def construct():
    # Randomized local search for better sets
    # Score: 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
    
    def calc_score(u):
        if not u:
            return 0
        max_u = max(u)
        if max_u == 0:
            return 0
        
        # Use sets for simplicity and speed for small n
        sums = set()
        diffs = set()
        for i in u:
            for j in u:
                sums.add(i + j)
                diffs.add(i - j)
        
        num_sums = len(sums)
        num_diffs = len(diffs)
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    best_score = 1.0281 # Current seed score
    best_set = [0, 1, 3, 7, 15] # Fallback
    
    start_time = time.time()
    time_limit = 10.0 # Run for 10 seconds to be safe
    
    # Parameters for search
    max_M = 50
    min_n = 4
    max_n = 10
    
    # Try different sizes
    for n in range(min_n, max_n + 1):
        # Initialize with a random set
        # Ensure 0 is in the set
        current_set = set([0])
        while len(current_set) < n:
            current_set.add(random.randint(1, max_M))
        
        current_score = calc_score(list(current_set))
        
        # Local search
        stuck_count = 0
        while time.time() - start_time < time_limit:
            # Pick a random move
            move_type = random.choice(['change', 'add', 'remove', 'swap'])
            
            new_set = set(current_set)
            
            if move_type == 'change':
                if len(new_set) == 0: continue
                elem = random.choice(list(new_set))
                if elem == 0:
                    # Don't change 0, or if we do, ensure 0 is re-added? 
                    # Task says "containing 0". So 0 must be in final set.
                    # We can change other elements.
                    pass
                else:
                    new_val = random.randint(0, max_M)
                    new_set.remove(elem)
                    new_set.add(new_val)
                    if 0 not in new_set:
                        # If we removed 0 (shouldn't happen if we skip 0), re-add
                        pass
            
            elif move_type == 'add':
                if len(new_set) < max_n:
                    new_set.add(random.randint(0, max_M))
            
            elif move_type == 'remove':
                if len(new_set) > min_n:
                    # Cannot remove 0
                    candidates = [x for x in new_set if x != 0]
                    if candidates:
                        new_set.remove(random.choice(candidates))
            
            elif move_type == 'swap':
                if len(new_set) >= 2:
                    candidates = [x for x in new_set if x != 0]
                    if len(candidates) >= 2:
                        e1, e2 = random.sample(candidates, 2)
                        new_set.remove(e1)
                        new_set.remove(e2)
                        new_set.add(random.randint(0, max_M))
                        new_set.add(random.randint(0, max_M))
                        if 0 not in new_set:
                             # Ensure 0 is present
                             # This move is tricky, maybe just change one
                             pass
            
            # Validate 0 is in set
            if 0 not in new_set:
                new_set.add(0)
                # If size exceeded, remove a random non-zero
                if len(new_set) > n:
                    candidates = [x for x in new_set if x != 0]
                    if candidates:
                        new_set.remove(random.choice(candidates))
            
            # Check for duplicates? Set handles it.
            # Check if set changed
            if new_set == current_set:
                continue
            
            new_score = calc_score(list(new_set))
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                stuck_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = sorted(list(current_set))
            else:
                stuck_count += 1
                # If stuck for too long, restart with new random set
                if stuck_count > 100:
                    break
        
        # Restart for next n or new random start if time allows
        if time.time() - start_time > time_limit:
            break
    
    return best_set
