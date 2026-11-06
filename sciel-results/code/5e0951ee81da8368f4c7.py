import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    best_result = []
    best_len = 0
    
    def is_valid_addition(S, x, sums_set):
        """Check if adding x to S keeps all triple sums unique."""
        n = len(S)
        
        # (x, x, x)
        if 3 * x in sums_set:
            return False
        
        # (x, x, y) for y in S
        for y in S:
            if 2 * x + y in sums_set:
                return False
        
        # (x, y, z) for y <= z in S
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                if x + y + z in sums_set:
                    return False
        
        return True
    
    def add_element(S, sums_set, x):
        """Add x to S and update sums_set with all new triple sums."""
        n = len(S)
        S.append(x)
        sums_set.add(3 * x)
        for i in range(n):
            y = S[i]
            sums_set.add(2 * x + y)
            for j in range(i, n):
                z = S[j]
                sums_set.add(x + y + z)

    def greedy_smallest():
        """Greedy algorithm picking the smallest valid number."""
        S = []
        sums_set = set()
        used = set()
        
        for x in range(1, 4097):
            if len(S) >= 16:
                break
            if x in used:
                continue
            if is_valid_addition(S, x, sums_set):
                add_element(S, sums_set, x)
                used.add(x)
        
        return S

    def greedy_randomized(order):
        """Greedy algorithm with a given order of candidates."""
        S = []
        sums_set = set()
        
        for x in order:
            if len(S) >= 16:
                break
            if time.time() > deadline:
                break
            if is_valid_addition(S, x, sums_set):
                add_element(S, sums_set, x)
        
        return S

    # Strategy 1: Deterministic smallest-first
    result = greedy_smallest()
    if len(result) > best_len:
        best_len = len(result)
        best_result = result
    
    # Strategy 2: Randomized restarts
    # We try to find a set of size 16.
    # If we already have 16, we can stop early or try to verify/improve (though 16 is max).
    if best_len < 16:
        rng = random.Random(42)
        for _ in range(5000): # Do many restarts
            if time.time() > deadline:
                break
            
            # Generate a random order of candidates
            # Instead of shuffling 4096 numbers, we can just pick random numbers
            # and check if they fit. This is equivalent to a random permutation.
            # But we want to bias towards smaller numbers to fit in range?
            # Actually, random numbers in [1, 4096] is fine.
            
            S = []
            sums_set = set()
            
            # Try to add 16 elements
            for _ in range(16):
                # Pick a random candidate
                # To increase success rate, we can try a few random candidates
                # before giving up on this restart
                found = False
                for _ in range(10): # Try 10 random candidates for each slot
                    x = rng.randint(1, 4096)
                    if is_valid_addition(S, x, sums_set):
                        add_element(S, sums_set, x)
                        found = True
                        break
                
                if not found:
                    break
            
            if len(S) > best_len:
                best_len = len(S)
                best_result = S
                
                if best_len == 16:
                    # We found a maximal set. We can break or continue to see if we can find another (not needed for score)
                    # But maybe we can try to polish it? No, 16 is the bound.
                    pass

    # Strategy 3: Local Search to improve if we are stuck at 15
    if best_len < 16 and best_result:
        # Try to swap elements to increase size?
        # If we have 15 elements, we can try to add a 16th.
        # If we can't add, maybe swapping one element allows adding two?
        # This is complex. Given the time, the randomized restarts should find 16 if it exists.
        pass

    return best_result if best_result else [1, 2, 4]
