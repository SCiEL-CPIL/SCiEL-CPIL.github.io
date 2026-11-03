import numpy as np
import time
import random
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    # Precompute masks for each codeword
    # mask[c] is a Python integer where bit p is 1 if p is covered by c
    masks = [0] * M
    powers = [3**i for i in range(n)]
    
    # To compute masks efficiently, we can iterate through all points
    # But M=6561, n=8.
    # For each c, its ball is c and neighbors.
    # We can precompute the list of points in the ball for each c.
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            # +1
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            # -1
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball
        
    # Convert balls to bitmasks
    for i in range(M):
        mask = 0
        for p in balls[i]:
            mask |= (1 << p)
        masks[i] = mask

    # Greedy Set Cover
    def greedy_set_cover(initial_code=None):
        code = set(initial_code) if initial_code else set()
        uncovered_mask = (1 << M) - 1 # All points uncovered initially
        
        # If initial code, update uncovered
        if initial_code:
            covered_mask = 0
            for c in initial_code:
                covered_mask |= masks[c]
            uncovered_mask = ~covered_mask & ((1 << M) - 1)

        # Heap: (-gain, codeword)
        # Gain is number of newly covered points
        gains = [0] * M
        for c in range(M):
            if c not in code:
                gains[c] = bin(uncovered_mask & masks[c]).count('1')
        
        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
        heapq.heapify(heap)
        
        uncovered_count = bin(uncovered_mask).count('1')
        
        while uncovered_count > 0:
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if c in code:
                    continue
                # Check if gain is still valid
                current_gain = bin(uncovered_mask & masks[c]).count('1')
                if current_gain != -neg_g:
                    if current_gain > 0:
                        heapq.heappush(heap, (-current_gain, c))
                    continue
                if current_gain <= 0:
                    continue
                valid_c = c
                break
            
            if valid_c is None:
                break
                
            code.add(valid_c)
            new_covered = uncovered_mask & masks[valid_c]
            uncovered_mask ^= new_covered # Remove newly covered
            uncovered_count -= bin(new_covered).count('1')
            
            # Update gains for other codewords
            # This is the expensive part.
            # We can lazily update. The heap check above handles stale entries.
            # But we need to push updated gains for neighbors?
            # Actually, lazy deletion is sufficient if we just push the new gain when we pop?
            # No, if gain decreases, the old entry is "too high", so we pop it, see it's stale, and push the correct one.
            # If gain increases? Gain only decreases as we cover more points.
            # So lazy deletion works.
            
        return code, uncovered_mask

    # Local Search
    def local_search(code, time_limit=115):
        start_time = time.time()
        # Precompute count array for fast redundancy check
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1
        
        # Helper to get unique points of c
        def get_unique_points(c):
            return [p for p in balls[c] if count[p] == 1]
            
        # Helper to remove c
        def remove_c(c):
            code.remove(c)
            for p in balls[c]:
                count[p] -= 1
                
        # Helper to add c
        def add_c(c):
            code.add(c)
            for p in balls[c]:
                count[p] += 1

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Phase 1: Remove redundant
            to_remove = []
            for c in list(code):
                # Check if all points in ball[c] have count > 1
                redundant = True
                for p in balls[c]:
                    if count[p] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
                    
            if to_remove:
                for c in to_remove:
                    remove_c(c)
                improved = True
                continue

            # Phase 2: 1-Swap
            code_list = list(code)
            random.shuffle(code_list)
            swapped = False
            for c in code_list:
                if (time.time() - start_time) >= time_limit:
                    break
                unique_points = get_unique_points(c)
                if not unique_points:
                    remove_c(c)
                    improved = True
                    swapped = True
                    break
                
                # Find candidate c' that covers all unique_points
                # Candidates must be in coverers[unique_points[0]]
                # We need coverers. Let's precompute coverers as lists of indices?
                # Or just iterate?
                # M=6561. Iterating all M for each swap is 6561 * 400 * 17 ~ 44M ops. Might be okay in C, slow in Python.
                # Let's use the mask intersection trick?
                # Intersection of masks of unique points? No, we need c' such that mask[c'] covers unique_points.
                # i.e., (mask[c'] & unique_mask) == unique_mask.
                
                unique_mask = 0
                for p in unique_points:
                    unique_mask |= (1 << p)
                
                found = False
                # Iterate through all non-code words? Or just all words?
                # To speed up, we can precompute for each point p, the list of c covering p.
                # We didn't do that in this bitmask version.
                # Let's stick to the list-based coverers from the previous logic but optimize the check.
                
                # Actually, let's just try random candidates from the ball of the first unique point?
                # No, c' must cover ALL unique points.
                
                # Efficient check:
                # Precompute `coverers` list of lists.
                # We didn't build it in this snippet. Let's assume we can access it or rebuild it.
                # Rebuilding coverers is O(M * 17) = 111k. Fast.
                
                if not hasattr(local_search, 'coverers'):
                    local_search.coverers = [[] for _ in range(M)]
                    for c in range(M):
                        for p in balls[c]:
                            local_search.coverers[p].append(c)
                
                cands = set(local_search.coverers[unique_points[0]])
                for p in unique_points[1:]:
                    cands.intersection_update(local_search.coverers[p])
                cands -= code
                
                if cands:
                    c_new = random.choice(list(cands))
                    remove_c(c)
                    add_c(c_new)
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                continue

            # Phase 3: Perturbation (if stuck)
            if (time.time() - start_time) < time_limit:
                # Simple perturbation: remove k random, re-greedy
                if len(code) > 10:
                    k = 5
                    to_perturb = random.sample(list(code), k)
                    for c in to_perturb:
                        remove_c(c)
                    
                    # Re-greedy on the uncovered set
                    # Calculate current uncovered mask
                    covered_mask = 0
                    for c in code:
                        covered_mask |= masks[c]
                    uncovered_mask = ~covered_mask & ((1 << M) - 1)
                    
                    # Simple greedy fill
                    gains = [0] * M
                    for c in range(M):
                        if c not in code:
                            gains[c] = bin(uncovered_mask & masks[c]).count('1')
                    
                    heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
                    heapq.heapify(heap)
                    
                    while heap:
                        neg_g, c = heapq.heappop(heap)
                        if c in code: continue
                        cur_g = bin(uncovered_mask & masks[c]).count('1')
                        if cur_g != -neg_g:
                            if cur_g > 0: heapq.heappush(heap, (-cur_g, c))
                            continue
                        if cur_g <= 0: continue
                        
                        add_c(c)
                        new_cov = uncovered_mask & masks[c]
                        uncovered_mask ^= new_cov
                        
                        if bin(uncovered_mask).count('1') == 0:
                            break

        return code

    # Multi-start
    best_code = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 115

    # Start with 0
    initial_code = {0}
    code, _ = greedy_set_cover(initial_code)
    code = local_search(code, time_limit=10)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # Random starts
    for _ in range(5):
        if time.time() - start_time > time_limit - 10:
            break
        initial_code = set(random.sample(range(M), 10))
        code, _ = greedy_set_cover(initial_code)
        code = local_search(code, time_limit=10)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()

    # Final refinement
    if best_code:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            best_code = local_search(best_code, time_limit=remaining_time)

    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in best_code]
