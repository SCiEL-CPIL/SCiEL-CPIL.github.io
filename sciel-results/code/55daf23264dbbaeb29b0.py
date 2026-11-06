"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a large B2[2] subset of {1..1000} using greedy + simulated annealing
    with fast sum-count maintenance via a 1D array.
    
    A set A is B2[2] if every integer s has at most 2 representations s = a + b
    with a, b in A and a <= b.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N  # max possible sum is 2000
    
    def is_valid_add(counts, x, chosen_set):
        """Check if adding x keeps all sum counts <= 2. Returns (valid, max_new_count)."""
        valid = True
        # Sums involving x: x + y for y in chosen_set (y <= x to avoid double count, but we need a<=b)
        # Actually, we need all pairs (a,b) with a<=b from A∪{x}.
        # New pairs involving x: (x,x) and (y,x) for y in chosen_set with y <= x.
        # Wait, the representation is a+b with a<=b.
        # If we add x, new pairs are:
        #   (x, x) -> sum 2x
        #   (y, x) for y in chosen_set where y <= x -> sum y+x
        #   (x, y) for y in chosen_set where y > x -> but we require a<=b, so this is (y,x) if y<x? No.
        #   The pairs are unordered {a,b} with a<=b. So new pairs are {x,y} for y in chosen_set, and {x,x}.
        #   For y in chosen_set: if y < x, pair is (y,x), sum y+x. If y > x, pair is (x,y), sum x+y.
        #   So for every y in chosen_set, we add one representation of sum x+y.
        #   And we add one representation of sum 2x.
        
        # We need to check if any of these sums would exceed count 2.
        # sums to check: [2*x] + [x + y for y in chosen_set]
        # But we can do this efficiently.
        
        # Check 2x
        if counts[2*x] >= 2:
            return False, 3
        
        # Check x + y for all y in chosen_set
        # We can iterate, but let's see if we can optimize.
        # For now, iterate.
        for y in chosen_set:
            s = x + y
            if counts[s] >= 2:
                return False, 3
        
        return True, 0
    
    def greedy_forward():
        """Standard greedy: add smallest possible element."""
        chosen = []
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        for x in range(1, N + 1):
            if is_valid_add(counts, x, chosen_set):
                # Add x
                chosen.append(x)
                chosen_set.add(x)
                # Update counts
                counts[2*x] += 1
                for y in chosen_set:
                    if y != x:  # y < x since we're going forward and x is new max? No, x is current, chosen has smaller
                        counts[x + y] += 1
                # Wait, the loop above adds x+y for all y in chosen_set. But when y=x, we already did 2x.
                # So we should exclude y=x or handle carefully.
                # Let's redo: for y in chosen_set (which doesn't include x yet), add count for x+y.
                # Then add 2x.
        
        # Let me rewrite greedy_forward more carefully.
        return chosen
    
    def greedy_forward_v2():
        """Standard greedy with proper count updates."""
        chosen = []
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        for x in range(1, N + 1):
            # Check validity
            valid = True
            if counts[2*x] >= 2:
                valid = False
            else:
                for y in chosen_set:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                # Update counts: add 1 to count[2x] and count[x+y] for all y in chosen_set (y != x, but x not in set yet)
                counts[2*x] += 1
                for y in chosen_set:
                    counts[x + y] += 1
        
        return chosen
    
    # First, get a baseline with greedy
    baseline = greedy_forward_v2()
    best_set = set(baseline)
    best_size = len(baseline)
    
    # Now try to improve with local search / simulated annealing
    # We'll try to remove some elements and add others.
    
    def get_count_array(chosen_set):
        """Compute the sum count array for a given set."""
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        lst = list(chosen_set)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            counts[2*a] += 1
            for j in range(i + 1, n):
                b = lst[j]
                counts[a + b] += 1
        return counts
    
    def is_valid_set(chosen_set):
        """Check if the set is a valid B2[2] set."""
        counts = get_count_array(chosen_set)
        return np.max(counts) <= 2
    
    # Try a few random restarts with greedy using different random orderings
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random ordering greedy
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        chosen = []
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        for x in order:
            valid = True
            if counts[2*x] >= 2:
                valid = False
            else:
                for y in chosen_set:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                counts[2*x] += 1
                for y in chosen_set:
                    counts[x + y] += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
    
    # Try local search: remove one, add one
    if time.time() < deadline:
        current_set = set(best_set)
        current_size = best_size
        
        # Precompute counts
        counts = get_count_array(current_set)
        
        max_iters = 20000
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to remove a random element and add a different one
            if len(current_set) == 0:
                break
            
            rem = rng.choice(list(current_set))
            current_set.remove(rem)
            
            # Find a candidate to add
            # Try random candidates not in set
            added = False
            for _ in range(20):
                cand = int(rng.integers(1, N + 1))
                if cand in current_set:
                    continue
                
                # Check if adding cand is valid
                valid = True
                if counts[2*cand] >= 2:
                    valid = False
                else:
                    for y in current_set:
                        if counts[cand + y] >= 2:
                            valid = False
                            break
                
                if valid:
                    current_set.add(cand)
                    counts[2*cand] += 1
                    for y in current_set:
                        if y != cand:
                            counts[cand + y] += 1
                    added = True
                    break
            
            if not added:
                # Revert
                current_set.add(rem)
                counts[2*rem] += 1
                for y in current_set:
                    if y != rem:
                        counts[rem + y] += 1
            
            if len(current_set) > current_size:
                current_size = len(current_set)
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
    
    return list(best_set)

# EVOLVE-BLOCK-END
