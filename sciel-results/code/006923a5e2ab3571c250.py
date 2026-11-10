import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # 1. Generate a valid initial set.
    # Powers of 2 are "sum-free" in the sense that no element is the mean 
    # of any subset of others? 
    # Actually, we just need a valid starting point. 
    # Let's use a greedy construction to get a valid set of size ~30-40 quickly.
    
    def greedy_construct(rng):
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        selected = []
        # sums_by_size[k] = set of sums of k elements from selected
        sums_by_size = {0: {0}}
        
        for x in candidates:
            valid = True
            # Check if x is mean of subset of selected
            for k in range(2, len(selected) + 1):
                if k * x in sums_by_size.get(k, set()):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if any a in selected is mean of subset containing x
            # subset S of selected U {x} containing x, size k
            # sum(S) = k*a => sum(S\{x}) + x = k*a => sum(S\{x}) = k*a - x
            # S\{x} is subset of selected of size k-1
            for a in selected:
                for k in range(2, len(selected) + 2):
                    target = k * a - x
                    if k - 1 <= len(selected):
                        if target in sums_by_size.get(k - 1, set()):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                selected.append(x)
                # Update sums
                old_sums = sums_by_size
                new_sums = {k: set() for k in range(len(selected) + 1)}
                new_sums[0] = {0}
                for k in range(1, len(selected) + 1):
                    if k in old_sums:
                        new_sums[k].update(old_sums[k])
                    if k - 1 in old_sums:
                        new_sums[k].update(s + x for s in old_sums[k - 1])
                sums_by_size = new_sums
                
                if len(selected) >= 30: # Just need a decent start
                    break
                    
        return selected, sums_by_size

    # Initialize with a few different seeds to pick the best starting point
    best_start_set = []
    best_start_sums = {}
    
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        s, sums = greedy_construct(rng)
        if len(s) > len(best_start_set):
            best_start_set = s
            best_start_sums = sums
            
    if not best_start_set:
        # Fallback to powers of 2 if greedy failed badly (unlikely)
        best_start_set = [2**i for i in range(9) if 2**i <= 512]
        # Rebuild sums
        best_start_sums = {0: {0}}
        for x in best_start_set:
            old = best_start_sums
            new = {k: set() for k in range(len(best_start_set)+1)}
            new[0] = {0}
            for k in range(1, len(best_start_set)+1):
                if k in old: new[k].update(old[k])
                if k-1 in old: new[k].update(s+x for s in old[k-1])
            best_start_sums = new

    current_set = best_start_set[:]
    current_sums = best_start_sums
    current_len = len(current_set)
    best_set = current_set[:]
    best_len = current_len
    
    # Helper to check validity of a swap (remove x, add y)
    # We maintain sums_by_size for the CURRENT set.
    # To check if we can remove x and add y:
    # 1. Check if y is mean of any subset of (current_set \ {x})
    # 2. Check if any a in (current_set \ {x}) is mean of any subset of ((current_set \ {x}) U {y}) containing y
    
    def can_swap_remove_x_add_y(x, y, set_list, sums_by_size):
        if y in set_list:
            return False
            
        # Create temporary sums for set_list without x
        # Instead of rebuilding full sums, we can compute sums for S\{x} on the fly or cache?
        # Rebuilding sums for 48 elements is O(N^2 * S) which is fast enough for 1000s of swaps?
        # N=48, S=48. 48*48*48 ~ 100k ops. 1000 swaps = 100M ops. Might be slow in Python.
        
        # Optimization: 
        # We can pre-compute the "impact" of removing x?
        # Or simply rebuild sums_by_size for S\{x} each time. 
        # Let's try rebuilding sums for S\{x} in the inner loop.
        
        # Actually, let's just rebuild the sums for the candidate set S' = (S \ {x}) U {y}
        # and check validity of the whole set? No, that's expensive.
        
        # Better approach for check:
        # 1. Compute sums_by_size for S_minus_x = S \ {x}
        # 2. Check y against S_minus_x
        # 3. Check a in S_minus_x against subsets of S_minus_x U {y} containing y
        
        S_minus_x = [v for v in set_list if v != x]
        n_minus = len(S_minus_x)
        
        # Build sums for S_minus_x
        sums_minus = {0: {0}}
        for v in S_minus_x:
            old = sums_minus
            new = {k: set() for k in range(n_minus + 1)}
            new[0] = {0}
            for k in range(1, n_minus + 1):
                if k in old:
                    new[k].update(old[k])
                if k - 1 in old:
                    new[k].update(s + v for s in old[k - 1])
            sums_minus = new
            
        # Check 1: y is not mean of subset of S_minus_x
        for k in range(2, n_minus + 1):
            if k * y in sums_minus.get(k, set()):
                return False
                
        # Check 2: No a in S_minus_x is mean of subset of (S_minus_x U {y}) containing y
        # Subset size k, contains y.
        # Sum of subset = k * a
        # Sum of (subset \ {y}) + y = k * a
        # Sum of (subset \ {y}) = k * a - y
        # subset \ {y} is a subset of S_minus_x of size k-1
        for a in S_minus_x:
            for k in range(2, n_minus + 2):
                target = k * a - y
                if k - 1 <= n_minus:
                    if target in sums_minus.get(k - 1, set()):
                        return False
                        
        return True

    # Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    
    # Precompute all possible swaps? No, too many.
    # Randomly pick x to remove, y to add.
    
    while time.time() < deadline - 2:
        if T < 0.01:
            T = 1.0 # Restart if stuck too low? Or just break.
            # Let's just break if T is too low and we haven't improved in a while?
            # For simplicity, just continue with low T (greedy descent)
            
        # Pick random x to remove
        if not current_set:
            break
        x_idx = random.randint(0, len(current_set) - 1)
        x = current_set[x_idx]
        
        # Pick random y to add from [1, 512] not in current_set
        # To make it faster, we can restrict y to a range?
        # Or just pick random y in 1..512 and check if in set.
        y = random.randint(1, 512)
        
        if y in current_set:
            continue
            
        if can_swap_remove_x_add_y(x, y, current_set, current_sums):
            # Calculate "energy" or just accept if valid?
            # The objective is to maximize size. 
            # If we swap 1 for 1, size stays same. 
            # We only gain size if we add without removing? No, we must maintain distinctness and validity.
            # Wait, the task says "Maximize cardinality".
            # If we swap x for y, cardinality is unchanged.
            # How do we increase cardinality?
            # We need to ADD an element. 
            # But if the set is "maximal" (no element can be added), we must swap to reach a different maximal set.
            
            # So, we accept the swap if it leads to a state where we might be able to add more later?
            # Or just random walk among valid sets of same size?
            # Yes, random walk among valid sets of size N.
            # Then, periodically, try to ADD an element.
            
            # Perform swap
            current_set[x_idx] = y
            # Rebuild sums for current_set
            n = len(current_set)
            new_sums = {0: {0}}
            for v in current_set:
                old = new_sums
                new = {k: set() for k in range(n + 1)}
                new[0] = {0}
                for k in range(1, n + 1):
                    if k in old:
                        new[k].update(old[k])
                    if k - 1 in old:
                        new[k].update(s + v for s in old[k - 1])
                new_sums = new
            current_sums = new_sums
            
            # Try to expand: Pick random y_add not in set, check if can add
            y_add = random.randint(1, 512)
            if y_add not in current_set:
                # Check if y_add can be added to current_set
                # Check 1: y_add not mean of subset of current_set
                valid_add = True
                for k in range(2, len(current_set) + 1):
                    if k * y_add in current_sums.get(k, set()):
                        valid_add = False
                        break
                
                if valid_add:
                    # Check 2: No a in current_set is mean of subset containing y_add
                    for a in current_set:
                        for k in range(2, len(current_set) + 2):
                            target = k * a - y_add
                            if k - 1 <= len(current_set):
                                if target in current_sums.get(k - 1, set()):
                                    valid_add = False
                                    break
                        if not valid_add:
                            break
                
                if valid_add:
                    # Accept expansion!
                    current_set.append(y_add)
                    # Rebuild sums
                    n = len(current_set)
                    new_sums = {0: {0}}
                    for v in current_set:
                        old = new_sums
                        new = {k: set() for k in range(n + 1)}
                        new[0] = {0}
                        for k in range(1, n + 1):
                            if k in old:
                                new[k].update(old[k])
                            if k - 1 in old:
                                new[k].update(s + v for s in old[k - 1])
                        new_sums = new
                    current_sums = new_sums
                    current_len = len(current_set)
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_set = current_set[:]
                        if best_len >= 48:
                            return best_set
        else:
            # Invalid swap, reject
            pass
            
        T *= cooling

    return best_set

# Final verification is handled by the evaluator, but we return the best valid set found.
# Note: The "can_swap" and "add" checks are exact.
# The initial greedy set is valid.
# Swaps preserve validity if the check passes.
# Additions preserve validity if the check passes.
# So best_set is always valid.
