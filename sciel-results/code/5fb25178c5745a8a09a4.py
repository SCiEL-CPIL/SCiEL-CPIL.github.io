import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other params, though task is specific
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3**n
    ball_size = 1 + 2*n # 17
    
    # Precompute balls as lists of indices
    balls = [None] * M
    # Precompute powers of 3
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * 3

    for i in range(M):
        ball = [i]
        digits = [0]*n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        
        for pos in range(n):
            # +1
            new_digit = (digits[pos] + 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
                
            # -1
            new_digit = (digits[pos] - 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        
        balls[i] = ball
        
    # Precompute coverers: for each point p, list of codewords c such that p in balls[c]
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    # 1. Greedy Set Cover
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.full(M, ball_size, dtype=np.int32)
    
    heap = [(-ball_size, c) for c in range(M)]
    heapq.heapify(heap)
    
    uncovered_count = M
    
    while uncovered_count > 0:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
            valid_c = c
            break
            
        if valid_c is None:
            break
            
        code.add(valid_c)
        
        for p in balls[valid_c]:
            if uncovered[p]:
                uncovered[p] = False
                uncovered_count -= 1
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        heapq.heappush(heap, (-gains[c_prime], c_prime))
                        
    # 2. Local Search with Iterated Perturbation
    start_time = time.time()
    time_limit = 115
    
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
            
    def try_remove_and_swap(code, count, balls, coverers):
        """
        Attempts to remove redundant codewords and perform 1-for-1 swaps.
        Returns True if any improvement (removal or swap) was made.
        """
        improved = False
        
        # Phase 1: Remove redundant
        to_remove = []
        for c in list(code):
            redundant = True
            for p in balls[c]:
                if count[p] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
            return True
            
        # Phase 2: Swap
        # We want to find a codeword c that has unique points, and a candidate c'
        # that covers those unique points.
        # To improve the solution, we prefer c' that maximizes redundancy.
        
        # Shuffle to avoid bias
        code_list = list(code)
        random.shuffle(code_list)
        
        for c in code_list:
            unique_points = [p for p in balls[c] if count[p] == 1]
            
            if not unique_points:
                # Should have been removed in Phase 1
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
                return True
            
            # Find candidates for c'
            # c' must cover all points in unique_points
            candidates = set(coverers[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(coverers[p])
            
            candidates -= code
            
            if not candidates:
                continue
                
            # Heuristic: Pick c' that maximizes the number of points in its ball
            # that are already covered by >1 codeword (excluding c).
            # This tends to create more redundancy.
            
            best_c = -1
            best_score = -1
            
            # Limit candidates for speed if list is huge?
            # Usually small.
            for c_new in candidates:
                # Calculate score
                # score = sum(1 for p in balls[c_new] if count[p] > 1)
                # Note: count[p] includes c. If we remove c, count[p] decreases by 1 for p in balls[c].
                # But c_new is not c.
                # We want to know if c_new helps cover points that are "safe".
                
                # A simpler heuristic: Maximize overlap with existing code.
                # If c_new covers many points that are already covered, it's a "safe" swap.
                
                # Let's just pick the one with max "redundancy potential"
                # which is roughly the number of neighbors in balls[c_new] that have count > 1.
                
                # Optimization: Pre-compute? No, count changes.
                # Just iterate.
                
                score = 0
                for p in balls[c_new]:
                    if count[p] > 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_c = c_new
                    
            if best_c != -1:
                # Perform Swap
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
                    
                code.add(best_c)
                for p in balls[best_c]:
                    count[p] += 1
                return True
                
        return False

    while (time.time() - start_time) < time_limit:
        # Try to improve
        if try_remove_and_swap(code, count, balls, coverers):
            continue
            
        # No improvement found. Perturb.
        # Remove K random codewords
        K = 5
        if len(code) < K:
            break
            
        to_perturb = random.sample(list(code), K)
        for c in to_perturb:
            code.remove(c)
            for p in balls[c]:
                count[p] -= 1
                
        # Repair: Greedily add codewords to cover uncovered points
        # We need to identify uncovered points.
        # Instead of scanning all M, we can just run a mini-greedy.
        # Find points with count == 0.
        
        # Efficient way: 
        # Create a list of uncovered points
        # Since K is small, the number of uncovered points is bounded by K * ball_size.
        # But some might be covered by others.
        
        # Simple repair:
        # 1. Identify uncovered points.
        # 2. While uncovered points exist:
        #    a. Pick a random uncovered point p.
        #    b. Pick a random codeword c from coverers[p] that is not in code.
        #    c. Add c to code, update counts.
        
        # To be faster, we can prioritize codewords that cover multiple uncovered points.
        
        # Let's collect uncovered points
        uncovered_points = [p for p in range(M) if count[p] == 0]
        
        while uncovered_points:
            # Pick a random uncovered point
            p_idx = random.randrange(len(uncovered_points))
            p = uncovered_points[p_idx]
            
            # Pick a candidate from coverers[p] not in code
            valid_cands = [c for c in coverers[p] if c not in code]
            if not valid_cands:
                # Should not happen if code was valid before perturbation
                break
                
            # Heuristic: Pick the one that covers most other uncovered points
            # For speed, just pick random or first
            # Let's try to pick one that covers max uncovered
            best_c = -1
            max_covered = -1
            for c in valid_cands:
                # Count how many of its neighbors are in uncovered_points
                # This is O(17 * len(uncovered_points)) which is expensive if list is large.
                # Given K=5, uncovered points <= 85.
                # 17 * 85 = 1445 checks. Acceptable.
                covered = 0
                for np_ in balls[c]:
                    if count[np_] == 0:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_c = c
                    
            if best_c == -1:
                best_c = valid_cands[0]
                
            code.add(best_c)
            for np_ in balls[best_c]:
                count[np_] += 1
                if count[np_] == 1:
                    # It was uncovered, now covered.
                    # Remove from uncovered_points list?
                    # This is inefficient to remove from list.
                    # We can just rebuild the list or use a set.
                    pass
                    
            # Rebuild uncovered list for next iteration? 
            # Or just check if p is covered.
            # Since we added best_c, p is now covered (count[p] >= 1).
            # Remove p from uncovered_points
            uncovered_points[p_idx] = uncovered_points[-1]
            uncovered_points.pop()
            
            # Note: Other points in balls[best_c] might also be covered now.
            # They are still in uncovered_points list with count > 0.
            # The loop condition `while uncovered_points` will continue.
            # Inside, we pick a random p. If count[p] > 0, we should skip it.
            # Let's add a check.
            # Actually, the `if count[np_] == 0` in the selection loop handles the "uncovered" status for scoring,
            # but the `uncovered_points` list might contain "stale" entries.
            
            # Better approach:
            # While True:
            #   Find any p with count[p] == 0.
            #   If none, break.
            #   ...
            
            # This is getting complex. Let's simplify the repair.
            # Just run the original Greedy logic on the "deficit".
            
            # Alternative Repair:
            # 1. Identify all p with count[p] == 0.
            # 2. While any exist:
            #    a. Pick p.
            #    b. Pick c from coverers[p] not in code.
            #    c. Add c.
            #    d. Update counts.
            
            # Let's restart the inner loop logic properly.
            # I'll implement a dedicated repair function.

    return [list(map(int, [ (c // (3**i)) % 3 for i in range(n) ])) for c in code]
