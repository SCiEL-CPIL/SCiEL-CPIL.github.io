import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to check if a set is valid and return counts
    def get_counts(S):
        counts = {}
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def is_valid(S):
        # Quick check: max count <= 2
        # We can do this incrementally in the search, but for initial check:
        counts = {}
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True

    def greedy_fill(S, counts, candidates):
        """
        Try to add candidates to S, updating counts.
        Returns new S and counts.
        """
        S = set(S)
        counts = dict(counts)
        # We want to add elements that fit.
        # To be efficient, we iterate through candidates and check fit.
        # The 'candidates' list should be those not in S.
        for y in candidates:
            if y in S:
                continue
            # Check if adding y is valid
            # Sums to check: y + x for x in S, and 2y
            valid = True
            for x in S:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            
            if valid:
                S.add(y)
                for x in S:
                    if x == y:
                        continue
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2*y] = counts.get(2*y, 0) + 1
        return S, counts

    # 1. Initial Greedy Construction
    S = set()
    counts = {}
    for x in range(1, 1001):
        # Check if x can be added
        valid = True
        for y in S:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            s = 2 * x
            if counts.get(s, 0) + 1 > 2:
                valid = False
        
        if valid:
            S.add(x)
            for y in S:
                if y == x:
                    continue
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2*x] = counts.get(2*x, 0) + 1
            
    best_S = S
    best_size = len(S)
    
    # 2. Local Search: Remove one, Refill
    # We repeat this until no improvement or time runs out
    while time.time() < deadline:
        improved = False
        S_list = list(S)
        random.shuffle(S_list) # Randomize order to escape symmetry
        
        for x in S_list:
            if time.time() > deadline:
                break
            # Remove x
            S_temp = S - {x}
            counts_temp = dict(counts)
            
            # Decrement counts for x
            for y in S_temp:
                s = x + y
                counts_temp[s] -= 1
            s = 2 * x
            counts_temp[s] -= 1
            # Clean up zeros if necessary, or just keep them (get(s,0) handles it)
            # It's safer to keep them to avoid KeyError, but they are 0 now.
            
            # Refill
            # Candidates are 1..1000 not in S_temp
            # We can just iterate 1..1000 and skip if in S_temp
            S_new, counts_new = greedy_fill(S_temp, counts_temp, range(1, 1001))
            
            if len(S_new) > best_size:
                best_S = S_new
                best_size = len(S_new)
                improved = True
                S = S_new
                counts = counts_new
                # Restart the loop with the new S? 
                # Or just continue with the improved S for the rest of the pass?
                # Continuing is faster.
        
        if not improved:
            # Try a more aggressive move: Remove 2, Refill
            # This allows escaping deeper local minima
            S_list = list(S)
            random.shuffle(S_list)
            found_improvement = False
            for i in range(len(S_list)):
                if time.time() > deadline:
                    break
                x1 = S_list[i]
                for j in range(i+1, len(S_list)):
                    if time.time() > deadline:
                        break
                    x2 = S_list[j]
                    S_temp = S - {x1, x2}
                    counts_temp = dict(counts)
                    
                    # Decrement for x1
                    for y in S_temp:
                        s = x1 + y
                        counts_temp[s] -= 1
                    s = 2 * x1
                    counts_temp[s] -= 1
                    
                    # Decrement for x2
                    for y in S_temp:
                        s = x2 + y
                        counts_temp[s] -= 1
                    s = 2 * x2
                    counts_temp[s] -= 1
                    
                    # Refill
                    S_new, counts_new = greedy_fill(S_temp, counts_temp, range(1, 1001))
                    
                    if len(S_new) > best_size:
                        best_S = S_new
                        best_size = len(S_new)
                        found_improvement = True
                        S = S_new
                        counts = counts_new
                        break
                if found_improvement:
                    break
            
            if not found_improvement:
                # If stuck, try a random restart with a slightly different greedy seed
                # e.g., skip the first K numbers
                if random.random() < 0.5:
                    # Try starting with a sparse set
                    S = set()
                    counts = {}
                    # Randomly pick 5 starting points
                    starters = random.sample(range(1, 1001), 5)
                    for x in starters:
                        valid = True
                        for y in S:
                            s = x + y
                            if counts.get(s, 0) + 1 > 2:
                                valid = False
                                break
                        if valid:
                            s = 2 * x
                            if counts.get(s, 0) + 1 > 2:
                                valid = False
                        if valid:
                            S.add(x)
                            for y in S:
                                if y == x:
                                    continue
                                s = x + y
                                counts[s] = counts.get(s, 0) + 1
                            counts[2*x] = counts.get(2*x, 0) + 1
                    
                    # Fill the rest greedily
                    S, counts = greedy_fill(S, counts, range(1, 1001))
                    if len(S) > best_size:
                        best_S = S
                        best_size = len(S)
                    # Then run local search on this new S
                    # The outer while loop will handle this if we don't break, 
                    # but we are inside the 'if not improved' block.
                    # We should probably just let the outer loop continue.
                    # To ensure we search this new basin, we can set S to best_S 
                    # and continue the while loop.
                    S = best_S
                    counts = get_counts(best_S) # Recalculate to be safe
                    # The while loop will repeat.

    return list(best_S)

