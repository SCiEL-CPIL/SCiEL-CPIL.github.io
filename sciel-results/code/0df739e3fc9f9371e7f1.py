import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Use simple random or direct sum if applicable
        if n <= 4:
            # Perfect code exists for n=4? Tetracode.
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        # For safety, return a large valid code
        # Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Brute force for small n
             return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    
    # Map word tuple to index
    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx
        
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    # Precompute neighbors for all 2187 words
    # neighbors[i] is a list of indices within distance 1 of idx_to_tuple(i)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w = idx_to_tuple(i)
        # Self
        neighbors[i].append(i)
        # Mutations
        for pos in range(n):
            for val in range(3):
                if val != w[pos]:
                    new_w = w[:]
                    new_w[pos] = val
                    neighbors[i].append(tuple_to_idx(new_w))
    
    # Initial Solution: Direct Sum (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_code.append(tuple_to_idx(h + list(tail)))
            
    # Ensure distinctness (they should be distinct)
    init_code = list(set(init_code))
    
    # Coverage array
    coverage = [0] * N
    code = list(init_code)
    
    # Update coverage
    for c in code:
        for w in neighbors[c]:
            coverage[w] += 1
            
    # Helper to check if code is valid (all coverage >= 1)
    # Not strictly needed if we maintain coverage correctly, but good for sanity
    # def is_valid():
    #     return all(cov >= 1 for cov in coverage)

    # Local Search: Removal
    def try_remove():
        nonlocal code, coverage
        removed = False
        # Iterate through a copy to allow removal during iteration
        for c in list(code):
            # Check if c is redundant
            # c is redundant if for all w in neighbors[c], coverage[w] > 1
            redundant = True
            for w in neighbors[c]:
                if coverage[w] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                code.remove(c)
                for w in neighbors[c]:
                    coverage[w] -= 1
                removed = True
        return removed

    # Local Search: Kick (Remove one, Greedy Repair)
    def greedy_repair(uncovered_set):
        # uncovered_set is a set of indices
        added = []
        temp_uncovered = set(uncovered_set)
        while temp_uncovered:
            # Find candidate that covers most of temp_uncovered
            best_c = -1
            best_count = 0
            # We can restrict candidates to those that intersect temp_uncovered
            # For efficiency, we can just iterate all N, but N=2187 is small
            # However, doing this many times is slow.
            # Optimization: Only consider candidates that are in the current code 
            # or maybe random ones? 
            # Actually, we should consider ALL potential codewords to find the best repair.
            # But checking all 2187 every step is 2187 * |uncovered| ops.
            # |uncovered| is small (size of sphere of removed c, max 15).
            # So 2187 * 15 ~ 30k ops per repair step.
            # Number of steps in repair ~ size of gap.
            
            # Let's just iterate all candidates
            for c in range(N):
                # Count how many of temp_uncovered are in neighbors[c]
                # Optimization: Pre-computed neighbors are lists.
                # Intersection size
                cnt = 0
                for w in neighbors[c]:
                    if w in temp_uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            
            if best_c == -1:
                break # Should not happen if space is connected
                
            added.append(best_c)
            for w in neighbors[best_c]:
                temp_uncovered.discard(w)
                
        return added

    def kick():
        nonlocal code, coverage
        if not code:
            return False
            
        old_size = len(code)
        
        # Pick a random codeword to remove
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]
        
        # Simulate removal
        temp_coverage = coverage[:]
        for w in neighbors[c_remove]:
            temp_coverage[w] -= 1
            
        # Find uncovered words
        uncovered = set()
        for w in range(N):
            if temp_coverage[w] == 0:
                uncovered.add(w)
                
        if not uncovered:
            # We can just remove it? This is handled by try_remove usually.
            # But if try_remove missed it? Unlikely.
            return False

        # Greedy repair
        added = greedy_repair(uncovered)
        
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            # Accept
            # Apply changes
            code.remove(c_remove)
            for w in neighbors[c_remove]:
                coverage[w] -= 1
            for c in added:
                code.append(c)
                for w in neighbors[c]:
                    coverage[w] += 1
            return True
        else:
            return False

    # Main Loop
    last_improvement = time.time()
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove()
            
        # 2. Kick
        if time.time() > deadline:
            break
        
        # Try a few kicks
        kicks_tried = 0
        success = False
        while kicks_tried < 10:
            if kick():
                success = True
                # Reset kicks counter if we found an improvement? 
                # Or just keep going.
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break
        if not success and kicks_tried >= 10:
            # Stuck. Maybe try a different strategy or restart?
            # For now, just continue trying removals (in case state changed subtly? No, state is static if no kick)
            # So we are in a local minimum.
            # We can try to randomize the order of removals? 
            # Or just wait for time to pass? No, that's bad.
            # Let's try a "Random Restart" of the local search? 
            # No, that's expensive.
            # Let's try more aggressive kicks: Remove 2, repair?
            # Or just increase kick attempts.
            pass

    # Convert indices back to lists of ints
    result = [idx_to_tuple(c) for c in code]
    return result
