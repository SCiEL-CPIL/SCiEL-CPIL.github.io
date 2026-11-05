import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave margin for 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists (balls of radius 1)
    # neighbors[i] is a numpy array of indices
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # Initial Solution: Direct Sum (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_indices.append(tuple_to_idx(h + list(tail)))
    
    init_indices = list(set(init_indices))
    
    # State
    code = init_indices
    code_set = set(code)
    coverage = np.zeros(N, dtype=np.int32)
    
    # Update coverage
    for c in code:
        coverage[neighbors_list[c]] += 1

    # Helper: Check if a candidate c is redundant given current coverage
    # c is redundant if min(coverage[neighbors[c]]) > 1
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    # Helper: Greedy Repair
    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Candidate pool: neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_list[w].tolist())
            
            # Filter out current code members (we are adding new ones)
            # Note: If a candidate is already in code, it's already active.
            # But wait, we removed some words. The "code" set has been updated in the caller?
            # In this function, we assume 'code' is the current state BEFORE the removal?
            # No, the caller passes 'uncovered_set' which is derived from a simulation.
            # The 'code' global variable still contains the removed words?
            # Let's check the caller.
            # In kick(): 
            #   temp_coverage = coverage.copy()
            #   temp_coverage[...] -= 1
            #   uncovered = ...
            #   added = greedy_repair(uncovered_set)
            #   # Then it updates code and coverage.
            # So during greedy_repair, 'code' still contains the removed words.
            # We should NOT add words that are already in 'code' (except the removed ones? No, we are repairing the gap).
            # Actually, if a word is in 'code', it is already covering its neighbors.
            # If it's in the candidate pool, it's because it's a neighbor of an uncovered word.
            # But if it's in 'code', it's already active.
            # So we should only consider candidates NOT in 'code'.
            
            for c in candidate_pool:
                if c in code_set:
                    continue
                
                # Count intersection with temp_uncovered
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        if best_count == len(temp_uncovered):
                            break
                
                if best_count == len(temp_uncovered):
                    break

            if best_c == -1:
                # Should not happen if coverage was valid before removal
                break

            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)

        return added

    def try_remove():
        nonlocal code, code_set, coverage
        removed = False
        # Iterate through a copy
        for c in list(code):
            if is_redundant(c):
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
                removed = True
        return removed

    def kick():
        nonlocal code, code_set, coverage
        if not code:
            return False
            
        old_size = len(code)
        
        # Pick a random codeword to remove
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]
        
        # Simulate removal
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        # Find uncovered words (where coverage becomes 0)
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # It was redundant, but try_remove should have caught it.
            # Just remove it.
            code.remove(c_remove)
            code_set.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        
        # Greedy repair
        added = greedy_repair(uncovered_set)
        
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            # Accept
            code.remove(c_remove)
            code_set.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                if c not in code_set: # Ensure distinct and not already added
                    code.append(c)
                    code_set.add(c)
                    coverage[neighbors_list[c]] += 1
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
        # Increase kick attempts if we are fast
        max_kicks = 20
        while kicks_tried < max_kicks:
            if kick():
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break
        
        if not success:
            # Stuck in local minimum.
            # Try a "2-kick" or "swap"?
            # For now, just continue. The random nature of kick selection might help eventually.
            # Or we can try to force a swap: Remove 2, Repair.
            # Let's implement a simple 2-removal kick.
            
            if len(code) > 2:
                # Pick 2 random distinct
                i1 = random.randint(0, len(code) - 1)
                i2 = random.randint(0, len(code) - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % len(code)
                
                c1 = code[i1]
                c2 = code[i2]
                
                # Simulate removal
                temp_coverage = coverage.copy()
                temp_coverage[neighbors_list[c1]] -= 1
                temp_coverage[neighbors_list[c2]] -= 1
                
                uncovered = np.where(temp_coverage == 0)[0]
                
                if len(uncovered) == 0:
                    # Both redundant? Unlikely if try_remove passed.
                    # Just remove one?
                    pass
                else:
                    uncovered_set = set(uncovered.tolist())
                    added = greedy_repair(uncovered_set)
                    
                    new_size = len(code) - 2 + len(added)
                    
                    if new_size < len(code):
                        # Accept
                        code.remove(c1)
                        code_set.remove(c1)
                        code.remove(c2)
                        code_set.remove(c2)
                        coverage[neighbors_list[c1]] -= 1
                        coverage[neighbors_list[c2]] -= 1
                        for c in added:
                            if c not in code_set: # Ensure distinct
                                code.append(c)
                                code_set.add(c)
                                coverage[neighbors_list[c]] += 1
                        # Success!
                        # Reset kicks counter?
                        pass

    # Convert indices back to lists of ints
    result = [idx_to_tuple(c) for c in code]
    return result
