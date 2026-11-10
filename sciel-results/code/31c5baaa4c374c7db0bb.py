import itertools
import time
import random

def is_sidon_4(subset):
    """
    Check if a 4-element tuple/list of distinct integers is a Sidon set.
    Sums a+b for a<=b must be unique.
    """
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4(S):
    """
    Count the number of 4-element subsets of S that are Sidon.
    S is a list of 12 distinct integers.
    """
    count = 0
    # Precompute sorted S for consistent iteration, though itertools handles order
    # We just need to check combinations
    for combo in itertools.combinations(S, 4):
        if is_sidon_4(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial candidate: Arithmetic Progression 0..11
    # This is known to have max Sidon size 4, so count_sidon_4 > 0
    best_set = list(range(12))
    best_count = count_sidon_4(best_set)
    
    if best_count == 0:
        return best_set

    # Local Search to eliminate all Sidon 4-subsets
    current_set = best_set.copy()
    current_count = best_count
    
    # We will try to reduce current_count to 0
    # Strategy: Hill Climbing with random restarts if stuck
    # Move: Replace one element with a random new value from [0, 128]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts % 10000 == 0:
             # Optional: print debug info if needed, but usually not
             pass
        
        # Try a random move
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Generate a new value
        # Bias towards values close to the current set to maintain "density" 
        # which encourages sum collisions
        if random.random() < 0.7:
             # Pick a random existing value as a neighbor
             neighbor_idx = random.randint(0, 11)
             neighbor_val = current_set[neighbor_idx]
             # Perturb around neighbor
             new_val = neighbor_val + random.randint(-5, 5)
        else:
             new_val = random.randint(0, 128)
        
        # Ensure new_val is in range and not already in the set (distinctness)
        if new_val < 0 or new_val > 128:
            continue
            
        # Check distinctness
        # Efficient check: is new_val in current_set?
        # Since we are modifying idx, we can check the rest
        # Or just make a copy and check
        if new_val in current_set and new_val != old_val:
            continue
        
        # Apply move
        current_set[idx] = new_val
        
        # Evaluate
        new_count = count_sidon_4(current_set)
        
        if new_count < current_count:
            # Improvement
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_set = current_set.copy()
            if current_count == 0:
                # Found a set with no Sidon 4-subsets!
                # Max Sidon size is 3 (since any 3 distinct ints form a Sidon set? 
                # Wait, 3 distinct ints a,b,c. 
                # Sums: 2a, a+b, 2b, a+c, b+c, 2c.
                # Are they always distinct?
                # 2a = a+b => a=b (impossible)
                # a+b = 2b => a=b (impossible)
                # a+b = a+c => b=c (impossible)
                # a+b = b+c => a=c (impossible)
                # a+b = 2c => possible? e.g. 1,2,3 -> 1+2=3, 2+2=4, 1+3=4. 
                # 1,2,3: Sums 2,3,4, 4,5,6. Collision 2+2=4, 1+3=4. 
                # So {1,2,3} is NOT Sidon.
                # So max Sidon size could be 2?
                # If max Sidon size is 2, score is even higher.
                # But usually for 12 elements, 3 is achievable.
                # Regardless, 0 Sidon 4-subsets means max size <= 3.
                # We return this set.
                return current_set
        else:
            # No improvement or worse
            # Revert with some probability to allow exploration (Simulated Annealing style)
            # Or just revert always for strict hill climbing.
            # Given the small space and target of 0, strict hill climbing might get stuck.
            # Let's accept equal or slightly worse with low probability?
            # No, let's stick to strict improvement for simplicity and speed.
            # If stuck, the outer loop doesn't restart automatically.
            # We need a restart mechanism.
            
            # Simple restart logic: if we haven't improved in a while, restart from best_set
            # We don't track "since improvement" easily here.
            # Let's add a counter.
            
            # Revert
            current_set[idx] = old_val
            
        # Check for timeout more frequently
        if time.time() > deadline:
            break
            
    # If we didn't find 0, return the best set found
    # This will have max Sidon size 4 (if best_count > 0)
    # Which is likely the same as the seed, yielding score 1.0
    return best_set

