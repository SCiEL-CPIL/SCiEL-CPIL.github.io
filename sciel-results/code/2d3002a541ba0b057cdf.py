"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # First, run the greedy to get a baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    greedy_len = len(values)
    
    if time.time() > deadline - 5:
        return values
    
    # Try to improve using local search / hill climbing
    # We'll try to insert elements at various positions or swap to increase length
    
    # Actually, the greedy is quite good. Let's try a different approach:
    # Use backtracking with pruning to find a longer sequence.
    # But the search space is huge. Instead, let's try multiple random restarts
    # with different starting points or ordering strategies.
    
    best = values[:]
    best_len = greedy_len
    
    # Strategy: try different orderings of candidates
    # 1. Ascending (done)
    # 2. Descending
    # 3. Various random shuffles
    
    import random
    
    # Try descending
    values, sums = [], set()
    for x in range(256, 0, -1):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    # Sort to make it increasing
    values.sort()
    if len(values) > best_len:
        best = values
        best_len = len(values)
    
    if time.time() > deadline - 10:
        return best
    
    # Try some random restarts with greedy
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random permutation of [1, 256]
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        values, sums = [], set()
        for x in candidates:
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        values.sort()
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    # Try a more sophisticated approach: start with greedy, then try to insert
    # additional elements at random positions
    
    for trial in range(100):
        if time.time() > deadline - 3:
            break
        
        # Start from the best found so far
        current = best[:]
        current_sums = set()
        
        # Compute all consecutive block sums for current
        n = len(current)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += current[j]
                current_sums.add(total)
        
        # Try to insert each unused number at each position
        used = set(current)
        unused = [x for x in range(1, 257) if x not in used]
        rng.shuffle(unused)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in unused[:20]:  # Try a subset for speed
                for pos in range(len(current) + 1):
                    # Check if inserting x at pos keeps all block sums distinct
                    new_sums = set(current_sums)
                    valid = True
                    
                    # Add new blocks involving x
                    # Blocks ending at x (starting from some i to pos-1)
                    total = x
                    for i in range(pos - 1, -1, -1):
                        total += current[i]
                        if total in new_sums:
                            valid = False
                            break
                        new_sums.add(total)
                    if not valid:
                        continue
                    
                    # Blocks starting at x (ending at some j to pos)
                    total = x
                    for j in range(pos, len(current)):
                        total += current[j]
                        if total in new_sums:
                            valid = False
                            break
                        new_sums.add(total)
                    if not valid:
                        continue
                    
                    # Blocks that now span across x (start before pos, end after pos)
                    for i in range(pos - 1, -1, -1):
                        total = current[i]
                        # Check if this prefix sum already conflicts
                        if total in new_sums:
                            valid = False
                            break
                        # Now extend with x and suffix
                        total += x
                        for j in range(pos, len(current)):
                            total += current[j]
                            if total in new_sums:
                                valid = False
                                break
                        if not valid:
                            break
                        new_sums.add(total)
                    if not valid:
                        continue
                    
                    # If valid, insert x
                    new_current = current[:pos] + [x] + current[pos:]
                    new_current.sort()
                    
                    # Verify the entire sequence is valid
                    verify_sums = set()
                    nn = len(new_current)
                    for i in range(nn):
                        total = 0
                        for j in range(i, nn):
                            total += new_current[j]
                            if total in verify_sums:
                                valid = False
                                break
                            verify_sums.add(total)
                        if not valid:
                            break
                    
                    if valid:
                        current = new_current
                        current_sums = verify_sums
                        unused.remove(x)
                        improved = True
                        if len(current) > best_len:
                            best = current[:]
                            best_len = len(current)
                        break
                if improved:
                    break
    
    return best
# EVOLVE-BLOCK-END
