import time
import random
import numpy as np

def _count_progression_diffs(S):
    """
    Count distinct d > 0 such that exists a with a, a+d, a+2d in S.
    S is a set of integers.
    """
    S_set = set(S)
    diffs = set()
    S_list = list(S)
    # For each pair (a, b) with a < b, d = b - a.
    # Check if a + 2d = b + (b-a) = 2b - a is in S.
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            d = b - a
            # We need a, a+d, a+2d. 
            # We have a and a+d=b.
            # a+2d = a + 2(b-a) = 2b - a.
            c = 2 * b - a
            if c in S_set:
                diffs.add(d)
    return len(diffs)

def _local_search(current, deadline, rng):
    """
    Perform local search / simulated annealing from current set.
    current: set of ints
    deadline: float time
    rng: random.Random instance
    """
    N = 32
    MAXV = 512
    current_list = list(current)
    current_score = _count_progression_diffs(current)
    best = set(current)
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute a bitmask for fast membership? 
    # Since MAXV is 512, a Python int can act as a 512-bit mask.
    # But set() lookup is O(1) and very fast in CPython. 
    # Let's stick to set for clarity and speed, unless we do many checks.
    
    while time.time() < deadline:
        # Generate a move
        # Strategy: 80% random swap, 20% "progression completion" swap
        move_type = rng.random()
        
        if move_type < 0.8:
            # Random swap
            to_remove_idx = rng.randrange(N)
            to_remove = current_list[to_remove_idx]
            
            # Find a random element to add
            # We can sample from 1..512 until we find one not in current
            while True:
                to_add = rng.randint(1, MAXV)
                if to_add not in current:
                    break
            
            if to_remove == to_add:
                continue
            
            new_set = set(current)
            new_set.remove(to_remove)
            new_set.add(to_add)
            
            # Update current_list for next iteration? 
            # It's safer to keep current as a set and convert to list when needed for indexing, 
            # or maintain the list in sync. 
            # Maintaining list in sync is faster.
            current_list[to_remove_idx] = to_add
            current = new_set
            
        else:
            # Heuristic move: Try to complete a progression
            # Find a pair (a, b) in current such that c = 2b-a is NOT in current, 
            # and swap some element x with c.
            # Or find (a, c) such that b = (a+c)/2 is missing.
            
            # Pick a random pair from current
            i, j = rng.sample(range(N), 2)
            a = current_list[i]
            b = current_list[j]
            
            if a > b:
                a, b = b, a
                
            d = b - a
            c = b + d # a + 2d
            
            if c <= MAXV and c not in current:
                # We have a, b. Missing c.
                # Swap some random element x with c.
                # Prefer to swap an element that is "useless" or part of a sparse region.
                # Just random swap for simplicity, but bias towards removing large d's?
                # Let's just pick a random element to remove.
                to_remove_idx = rng.randrange(N)
                to_remove = current_list[to_remove_idx]
                
                if to_remove == c:
                    continue
                
                new_set = set(current)
                new_set.remove(to_remove)
                new_set.add(c)
                
                current_list[to_remove_idx] = c
                current = new_set
            else:
                # If c is already in current, maybe try to fix a different gap?
                # Or just do a random swap.
                to_remove_idx = rng.randrange(N)
                to_remove = current_list[to_remove_idx]
                while True:
                    to_add = rng.randint(1, MAXV)
                    if to_add not in current:
                        break
                if to_remove == to_add:
                    continue
                new_set = set(current)
                new_set.remove(to_remove)
                new_set.add(to_add)
                current_list[to_remove_idx] = to_add
                current = new_set

        new_score = _count_progression_diffs(current)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0:
                # Accept with probability exp(delta / T)
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                else:
                    # Revert move
                    # This is tricky because we mutated current_list and current.
                    # We need to store the previous state or revert explicitly.
                    # To avoid complex revert logic, we can just keep a history stack or 
                    # re-evaluate from a copy. 
                    # Given the low cost of copying a set of 32 ints, let's just copy before move.
                    # Wait, I mutated in place. Let's refactor to avoid in-place mutation for safety.
                    pass
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best, best_score

def _local_search_safe(current, deadline, rng):
    """
    Safe local search that doesn't mutate inputs in a way that breaks revert.
    """
    N = 32
    MAXV = 512
    current = set(current)
    current_score = _count_progression_diffs(current)
    best = set(current)
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        # Generate candidate move
        move_type = rng.random()
        new_set = set(current)
        
        if move_type < 0.8:
            # Random swap
            to_remove = rng.choice(list(current))
            while True:
                to_add = rng.randint(1, MAXV)
                if to_add not in current:
                    break
            
            if to_remove == to_add:
                continue
                
            new_set.remove(to_remove)
            new_set.add(to_add)
            
        else:
            # Heuristic: Complete progression
            # Pick random a, b from current
            a, b = rng.sample(list(current), 2)
            if a > b:
                a, b = b, a
            d = b - a
            c = b + d
            
            if c <= MAXV and c not in current:
                # Swap random element with c
                to_remove = rng.choice(list(current))
                if to_remove == c:
                    continue
                new_set.remove(to_remove)
                new_set.add(c)
            else:
                # Fallback to random swap
                to_remove = rng.choice(list(current))
                while True:
                    to_add = rng.randint(1, MAXV)
                    if to_add not in current:
                        break
                if to_remove == to_add:
                    continue
                new_set.remove(to_remove)
                new_set.add(to_add)

        new_score = _count_progression_diffs(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0:
                if rng.random() < np.exp(delta / T):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best, best_score

def construct():
    start_time = time.time()
    # Leave 5 seconds for safety
    deadline = start_time + 145.0
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    best_overall = set(range(1, N + 1))
    best_overall_score = _count_progression_diffs(best_overall)
    
    # Try a few different initializations
    # 1. Random
    # 2. Geometric-ish (1, 2, 4, 8...) - might be too sparse
    # 3. Multiples of small k
    
    # Let's run multiple restarts
    # Estimate time per restart: 
    # Local search loop runs until deadline. 
    # It's better to have a fixed number of steps per restart and many restarts, 
    # OR one long run. 
    # Given the non-convexity, multiple restarts are good.
    
    # Let's do 5 restarts, each with a portion of the time.
    # Or just one long run with random restarts if stuck?
    # The current program does restarts. Let's keep that.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random initial set
        # Bias towards smaller numbers? 
        # Smaller numbers allow more room for larger d? 
        # Actually, if numbers are small, d is small. 
        # If numbers are spread out, d is large.
        # We want distinct d. 
        # A set like {1, 2, 4, 8, 16...} has d's that are powers of 2.
        # A set like {1, 2, 3, 4...} has many small d's.
        # Let's mix.
        
        current = set(rng.sample(range(1, MAXV + 1), N))
        current_score = _count_progression_diffs(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search for a short duration
        # 2 seconds per restart?
        # Total time 145s. ~70 restarts? 
        # Local search might be too slow if it does full scoring every step.
        # _count_progression_diffs is ~1000 ops. 1000 steps * 1000 ops = 1M ops. ~0.1s.
        # So we can do many steps.
        
        # Let's allocate 2s per restart
        restart_deadline = time.time() + 2.0
        if restart_deadline > deadline:
            restart_deadline = deadline
            
        best, best_score = _local_search_safe(current, restart_deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
            
    return sorted(list(best_overall))
