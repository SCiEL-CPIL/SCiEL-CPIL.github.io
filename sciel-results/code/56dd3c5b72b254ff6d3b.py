"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # Safety margin
    N = 4096
    TARGET_SIZE = 40  # Theoretical max based on sum range 3..12288
    
    def can_add(x, selected, triple_sums):
        """
        Check if x can be added to selected without creating duplicate triple sums.
        Returns the set of new triple sums if valid, else None.
        """
        new_sums = set()
        
        # 1. x + x + x
        s = 3 * x
        if s in triple_sums or s in new_sums:
            return None
        new_sums.add(s)
        
        # 2. x + x + y for y in selected
        for y in selected:
            s = 2 * x + y
            if s in triple_sums or s in new_sums:
                return None
            new_sums.add(s)
            
        # 3. x + y + z for y, z in selected (y <= z)
        k = len(selected)
        for i in range(k):
            y = selected[i]
            s = x + 2 * y
            if s in triple_sums or s in new_sums:
                return None
            new_sums.add(s)
            
            for j in range(i + 1, k):
                z = selected[j]
                s = x + y + z
                if s in triple_sums or s in new_sums:
                    return None
                new_sums.add(s)
                
        return new_sums

    def greedy_build(start_range, target_size, randomize=False):
        """
        Build a set using a greedy approach.
        If randomize is False, picks the smallest valid x.
        If randomize is True, picks a random valid x from a window or shuffled list.
        """
        selected = []
        triple_sums = set()
        
        if not randomize:
            candidates = list(range(start_range, N + 1))
        else:
            candidates = list(range(start_range, N + 1))
            random.shuffle(candidates)
            
        for x in candidates:
            if len(selected) >= target_size:
                break
            if time.time() > DEADLINE:
                break
                
            new_sums = can_add(x, selected, triple_sums)
            if new_sums is not None:
                selected.append(x)
                triple_sums.update(new_sums)
                
        return selected

    best_set = []
    
    # 1. Try deterministic greedy (smallest first)
    # This is the fastest and often works well for Sidon sets if range is large enough.
    current = greedy_build(1, TARGET_SIZE, randomize=False)
    if len(current) > len(best_set):
        best_set = current
        
    # 2. If we didn't reach target, try some randomized restarts
    # Randomizing helps escape local minima where small numbers block larger ones.
    if len(best_set) < TARGET_SIZE:
        attempts = 0
        while time.time() < DEADLINE - 5.0 and attempts < 20:
            # Try starting from different ranges or just shuffling
            # A full shuffle is expensive to generate, but we can just pick random x's
            # Let's do a "random greedy": at each step, pick a random valid x from [1, N]
            # This is slower per step but might find a better global structure.
            
            def random_greedy(target_size):
                selected = []
                triple_sums = set()
                # To avoid O(N) scan every time, we can just try random numbers
                # But checking a random number is O(k^2). 
                # If we fail, we try another.
                # With N=4096 and k~40, probability of a random x being valid is decent.
                
                for _ in range(target_size):
                    added = False
                    # Try a few random candidates
                    for _ in range(100): # Max 100 tries per slot
                        if time.time() > DEADLINE:
                            return selected
                        x = random.randint(1, N)
                        if x in selected:
                            continue
                        new_sums = can_add(x, selected, triple_sums)
                        if new_sums is not None:
                            selected.append(x)
                            triple_sums.update(new_sums)
                            added = True
                            break
                    if not added:
                        break
                return selected

            current = random_greedy(TARGET_SIZE)
            if len(current) > len(best_set):
                best_set = current
                if len(best_set) == TARGET_SIZE:
                    break
            attempts += 1
            
    return best_set

# EVOLVE-BLOCK-END
