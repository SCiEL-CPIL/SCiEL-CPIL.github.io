import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    MAX_SUM = 2 * N
    
    best_size = 0
    best_chosen = []
    
    rng = random.Random(42)
    
    def run_one_search(seed_offset=0):
        nonlocal best_size, best_chosen
        
        chosen_bits = 0
        chosen = set()
        sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        blocked_bits = 0
        
        # List of elements in chosen for fast iteration/removal
        # We'll just use the set, but for random choice we need a list.
        # We can maintain a list and a set.
        chosen_list = []
        
        def can_add(x):
            # Check if x can be added
            if (chosen_bits << x) & blocked_bits:
                return False
            if (blocked_bits >> (2*x)) & 1:
                return False
            return True

        def add_element(x):
            nonlocal chosen_bits, blocked_bits
            for y in chosen_list:
                s = x + y
                sum_counts[s] += 1
                if sum_counts[s] == 2:
                    blocked_bits |= (1 << s)
            s2 = 2 * x
            sum_counts[s2] += 1
            if sum_counts[s2] == 2:
                blocked_bits |= (1 << s2)
            chosen_bits |= (1 << x)
            chosen.add(x)
            chosen_list.append(x)

        def remove_element(x):
            nonlocal chosen_bits, blocked_bits
            # Remove x from chosen_list
            # Find index? O(N). 
            # For N=1000, O(N) is fine.
            try:
                idx = chosen_list.index(x)
                chosen_list[idx] = chosen_list[-1]
                chosen_list.pop()
            except ValueError:
                pass
            
            for y in chosen_list:
                if y == x:
                    continue
                s = x + y
                sum_counts[s] -= 1
                if sum_counts[s] == 1:
                    blocked_bits &= ~(1 << s)
            s2 = 2 * x
            sum_counts[s2] -= 1
            if sum_counts[s2] == 1:
                blocked_bits &= ~(1 << s2)
            chosen_bits &= ~(1 << x)
            chosen.discard(x)

        # Phase 1: Greedy
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline:
                return
            if can_add(x):
                add_element(x)
        
        current_size = len(chosen)
        
        # Phase 2: Local Search
        stagnant_count = 0
        max_stagnant = 500 # Increased
        
        while time.time() < deadline - 5 and stagnant_count < max_stagnant:
            improved = False
            
            # 1. Try to add
            # Pick random few from unchosen
            # To avoid O(N) unchosen list construction, we can just pick random 1..N and check if in chosen
            # But we need to ensure we try enough.
            # Let's try 50 random candidates.
            for _ in range(50):
                x = rng.randint(1, N)
                if x not in chosen:
                    if can_add(x):
                        add_element(x)
                        improved = True
                        stagnant_count = 0
                        break
            
            if improved:
                continue
            
            # 2. Try Swap: Remove y, Add x1, x2
            if not chosen_list:
                break
                
            y = rng.choice(chosen_list)
            remove_element(y)
            
            # Track sums that became free (count dropped from 2 to 1)
            # We don't strictly need this list, but it helps target adds.
            # For now, just try to add random elements.
            
            added_count = 0
            # Try to add up to 3 elements to ensure net gain or at least break tie
            # If we add 1, size is same. If we add 2, size +1.
            # We should accept size same swaps to explore.
            
            # To make swaps more effective, we can try to add elements that are "compatible"
            # with the freed sums.
            # But simple random might work with enough restarts.
            
            unchosen_candidates = []
            # Generate a pool of candidates
            # We can just pick randoms
            
            for _ in range(100):
                x = rng.randint(1, N)
                if x in chosen:
                    continue
                if can_add(x):
                    add_element(x)
                    added_count += 1
                    if added_count >= 2: # Net gain +1
                        break
            
            if added_count > 0:
                # We removed 1, added k.
                # If k=1, size same. If k>=2, size increased.
                # Accept.
                improved = True
                stagnant_count = 0
            else:
                # Revert
                add_element(y)
                stagnant_count += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
            
        return len(chosen)

    restarts = 0
    while time.time() < deadline - 10:
        run_one_search(restarts)
        restarts += 1

    return best_chosen
