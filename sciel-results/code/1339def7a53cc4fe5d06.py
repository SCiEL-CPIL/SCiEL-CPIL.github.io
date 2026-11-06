"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_set(S):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in S for some a.
    """
    S_set = set(S)
    count = 0
    # Max possible d: a=1, a+2d <= 512 => 2d <= 511 => d <= 255
    for d in range(1, 256):
        # Check if there exists a in S such that a+d and a+2d are in S
        # Optimization: Iterate over S. If S is small, this is fast.
        found = False
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                found = True
                break
        if found:
            count += 1
    return count

def _local_search(initial_set, deadline):
    """
    Perform hill climbing local search on the set.
    Returns the best set found.
    """
    current_set = initial_set[:]
    current_score = _score_set(current_set)
    
    no_improvement_count = 0
    max_no_improvement = 1000 # Restart if stuck for this many moves
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break # Signal to restart
            
        improved = False
        # Try a few random moves
        # We alternate between Swap and Shift to explore different neighborhoods
        move_type = random.choice(['swap', 'shift'])
        
        for _ in range(10): # Try 10 moves per iteration
            if time.time() > deadline:
                return current_set, current_score, no_improvement_count
            
            new_set = current_set[:]
            
            if move_type == 'swap':
                i = random.randint(0, 31)
                old_val = new_set[i]
                new_val = random.randint(1, 512)
                # Ensure new_val is not already in the set
                # Small optimization: if collision, just pick another
                while new_val in new_set:
                    new_val = random.randint(1, 512)
                new_set[i] = new_val
            else: # shift
                i = random.randint(0, 31)
                old_val = new_set[i]
                # Try shifting by -1 or +1
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                if 1 <= new_val <= 512 and new_val not in new_set:
                    new_set[i] = new_val
                else:
                    continue # Skip this move if invalid
            
            new_score = _score_set(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                no_improvement_count = 0
                break # Accept and restart the move generation loop
            else:
                no_improvement_count += 1
                
    return current_set, current_score, no_improvement_count

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_set = list(range(1, 33))
    best_score = _score_set(best_set)
    
    # List of initial seeds to explore
    # 1. The seed
    # 2. Random uniform
    # 3. Spread out (every 16th number)
    # 4. Clustered in middle
    seeds = [
        list(range(1, 33)),
        random.sample(range(1, 513), 32),
        [i * 16 + 1 for i in range(32)], # 1, 17, 33...
        [i * 10 + 200 for i in range(32)] # Clustered
    ]
    
    # Ensure seeds are valid (distinct, in range)
    for i, seed in enumerate(seeds):
        if len(set(seed)) != 32 or any(x < 1 or x > 512 for x in seed):
            seeds[i] = random.sample(range(1, 513), 32)

    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Run local search from this seed
        # We can run multiple local searches from the same seed if time permits
        # but usually one good run is enough if we have multiple seeds.
        # To be robust, we'll just run one pass per seed, and if time is left, 
        # we'll do random restarts.
        
        res_set, res_score, _ = _local_search(seed, deadline)
        
        if res_score > best_score:
            best_score = res_score
            best_set = res_set
    
    # If we have time left, do random restarts to explore more
    while time.time() < deadline - 5.0: # Keep 5s for final check
        # Generate a new random seed
        seed = random.sample(range(1, 513), 32)
        
        # Short local search
        # Use a tighter deadline for this specific search to allow more restarts
        local_deadline = min(deadline, time.time() + 10.0)
        
        res_set, res_score, _ = _local_search(seed, local_deadline)
        
        if res_score > best_score:
            best_score = res_score
            best_set = res_set

    return best_set
# EVOLVE-BLOCK-END
