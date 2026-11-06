"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_size = 4
    best_set = [1, 2, 5, 11]
    
    # Helper to check if a set is valid (pairwise sums distinct)
    def is_valid(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                total = s[i] + s[j]
                if total in sums:
                    return False
                sums.add(total)
        return True
    
    # Helper to compute sum set
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    # 1. Greedy initialization (fast, gives a decent starting point)
    def greedy_init():
        a = [1, 2, 5, 11]
        sums = get_sums(a)
        for x in range(12, 285):
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & sums:
                a.append(x)
                sums |= new
        return a
    
    current = greedy_init()
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # 2. Local Search with Simulated Annealing
    # We will try to improve the current set by swapping elements.
    # State: sorted list of integers
    # Move: remove one element, add another (not in set)
    # Cost: number of duplicate sums (0 is valid)
    
    def count_conflicts(s):
        sums = {}
        conflicts = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                total = s[i] + s[j]
                if total in sums:
                    conflicts += 1
                else:
                    sums[total] = True
        return conflicts
    
    def local_search(initial_set, time_limit):
        s = sorted(initial_set)
        current_conflicts = count_conflicts(s)
        best_s = s[:]
        best_conflicts = current_conflicts
        best_size = len(s)
        
        # Temperature parameters
        T = 10.0
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            if current_conflicts == 0:
                # Valid solution, try to increase size? 
                # For now, just return if we found a valid one of this size.
                # Actually, we want to maximize size. 
                # If we have a valid set, we can try to add more elements greedily.
                break
            
            # Random move: pick a random element to remove, pick a random candidate to add
            # To keep size constant, we swap.
            # Or we can vary size? Let's stick to fixed size for SA, then try to expand.
            
            idx_remove = random.randint(0, len(s) - 1)
            removed = s[idx_remove]
            # Pick a random candidate from 1..284 not in s
            # To be efficient, pick from a pool of "interesting" numbers or random
            # Random from 1..284
            candidate = random.randint(1, 284)
            
            if candidate == removed:
                continue
            if candidate in s:
                continue
                
            new_s = s[:idx_remove] + s[idx_remove+1:] + [candidate]
            new_s.sort()
            
            new_conflicts = count_conflicts(new_s)
            
            delta = new_conflicts - current_conflicts
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                s = new_s
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_s = s[:]
                    if best_conflicts == 0:
                        # Found a valid set of this size
                        # Try to expand it
                        expanded = try_expand(best_s)
                        if len(expanded) > best_size:
                            best_size = len(expanded)
                            best_s = expanded[:]
                            best_conflicts = 0
            T *= cooling_rate
            
            # If T is too low, restart with higher T or just break
            if T < 0.1:
                break
                
        return best_s, best_size
    
    def try_expand(s):
        # Greedily add elements to a valid set s
        a = s[:]
        sums = get_sums(a)
        # Try adding from high to low or low to high? 
        # Let's try random order for diversity, or just sequential.
        # Sequential is faster for greedy.
        candidates = [x for x in range(1, 285) if x not in a]
        random.shuffle(candidates)
        for x in candidates:
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & sums:
                a.append(x)
                sums |= new
        return a
    
    # Main loop: multiple restarts
    # We start with the greedy result.
    # Then we perform SA to potentially find a better arrangement of the same size,
    # or we can try to fix conflicts if we start with a slightly invalid set?
    # Actually, the greedy result is valid.
    # The goal is to INCREASE size.
    # Local search on fixed size doesn't increase size.
    # We need a strategy to increase size.
    # Strategy:
    # 1. Take current best valid set.
    # 2. Try to add one more element. If stuck, use SA to "make room" for a new element?
    # This is complex.
    
    # Alternative:
    # Start with a set of size K.
    # Run SA to minimize conflicts.
    # If conflicts == 0, we have a valid set of size K.
    # Try to increase K by 1 and run SA again.
    
    # Let's try this incremental approach.
    
    # Start with size 4 (prescribed)
    # Actually, greedy already gave us a valid set of size ~12-15.
    # Let's try to improve the greedy result using SA to find a "better" valid set
    # that allows for more extensions?
    # Or just use SA to find a valid set of size N+1 directly?
    
    # Let's try to find a valid set of size S where S > len(greedy).
    # We can initialize with the greedy set plus a random element (causing conflicts),
    # and then use SA to resolve conflicts.
    
    base_set = greedy_init()
    base_size = len(base_set)
    
    # Try to extend to base_size + 1, base_size + 2, etc.
    target_size = base_size + 1
    
    while time.time() < deadline - 5 and target_size <= 20: # Max possible is 20
        
        # Initialize with base_set + 1 random element
        candidates = [x for x in range(1, 285) if x not in base_set]
        if not candidates:
            break
            
        init_s = base_set + [random.choice(candidates)]
        random.shuffle(init_s)
        
        # Run SA to resolve conflicts for this size
        found_valid = False
        best_s_for_size = None
        
        # Run SA multiple times or for a short time
        sa_deadline = time.time() + 5 # 5 seconds per target size attempt?
        
        s, size = local_search(init_s, sa_deadline)
        
        if size == target_size and count_conflicts(s) == 0:
            # Found a valid set of target_size
            best_set = s
            best_size = target_size
            base_set = s # Update base for next iteration
            target_size += 1
            found_valid = True
        else:
            # Failed to extend
            # Maybe try a different random start?
            # For now, break or try a few more times
            # Let's try 3 times for this size
            success = False
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                init_s = base_set + [random.choice([x for x in range(1, 285) if x not in base_set])]
                random.shuffle(init_s)
                s, size = local_search(init_s, time.time() + 3)
                if size == target_size and count_conflicts(s) == 0:
                    best_set = s
                    best_size = target_size
                    base_set = s
                    target_size += 1
                    success = True
                    break
            
            if not success:
                break # Couldn't extend further with current method
                
    # Final check: try greedy expansion on the best_set found
    final_set = try_expand(best_set)
    if len(final_set) > best_size:
        best_set = final_set
        best_size = len(final_set)
        
    return best_set
# EVOLVE-BLOCK-END
