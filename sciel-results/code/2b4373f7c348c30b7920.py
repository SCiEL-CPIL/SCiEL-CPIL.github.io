import time
import numpy as np
import random
import heapq
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task specifies 3,7,1
        import itertools
        if n >= 4:
            # Simple construction for fallback
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
            
    N = 3 ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15
    
    # Precompute balls and cover_map
    # balls[i] is list of indices of words within distance 1 of i
    # cover_map[w] is list of indices of codewords that cover w
    balls = [[] for _ in range(N)]
    cover_map = [[] for _ in range(N)]
    
    # Helper to convert index to base-3 digits
    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # Precomputation
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.append(base3_to_int(new_digits))
        balls[i] = neighbors
        for w in neighbors:
            cover_map[w].append(i)
            
    # Convert to sets for fast intersection in local search
    ball_sets = [set(b) for b in balls]
    # cover_map is already lists, good for intersection
    
    def greedy_set_cover(seed=None):
        rng = random.Random(seed) if seed is not None else None
        covered = [False] * N
        uncovered_count = N
        selected = []
        
        # Gain: number of uncovered words in ball
        # We can just compute gain on the fly or use heap
        # Simple greedy: pick word with max gain
        # To speed up, we can use the fact that balls are small.
        
        # Initial gain is ball_size for all
        gains = [ball_size] * N
        
        # Use a heap for max-gain selection
        # heapq is min-heap, so use -gain
        heap = [(-ball_size, i) for i in range(N)]
        heapq.heapify(heap)
        in_heap = [True] * N
        # We need to handle stale entries. 
        # Better: just recompute or use a simpler loop if N is small enough?
        # N=2187. 2187^2 is ~4.7M. Greedy takes ~180 steps. 180 * 2187 * 15 is manageable.
        
        # Let's use the heap approach from inspiration for speed
        # But updating gains is O(cover_map[w]) for each newly covered w.
        
        cnt = [ball_size] * N # cnt[i] is current gain for codeword i
        
        while uncovered_count > 0:
            # Get max gain
            best_gain = -1
            best_idx = -1
            # Pop from heap
            while heap:
                neg_val, i = heapq.heappop(heap)
                if cnt[i] == -neg_val:
                    best_gain = cnt[i]
                    best_idx = i
                    break
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            
            # Update coverage
            for w in balls[best_idx]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    # Update gains for all codewords that cover w
                    for c2 in cover_map[w]:
                        if not covered[c2]: # Optimization: only if c2 is not selected? 
                            # Actually, gain of c2 decreases by 1 if w was uncovered and c2 covers w.
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))
                            # If cnt[c2] becomes 0, it's still in heap but will be skipped if stale
                            
        return selected

    def local_search(selected, seed):
        rng = random.Random(seed)
        code = list(selected)
        in_code = [False] * N
        for c in code:
            in_code[c] = True
            
        # cover_count[w] = number of codewords in code that cover w
        cover_count = [0] * N
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > time_limit - 5:
                    break
                
                c_remove = code[i]
                
                # 1. Try Deletion
                # Check if any word in balls[c_remove] has cover_count == 1
                can_remove = True
                for w in balls[c_remove]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove c_remove
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                    code.pop(i)
                    in_code[c_remove] = False
                    improved = True
                    break # Restart loop with new code
                
                # 2. Try 1-Swap (Remove c_remove, add c_add)
                # Find uncovered words if c_remove is removed
                newly_uncovered = []
                for w in balls[c_remove]:
                    if cover_count[w] == 1:
                        newly_uncovered.append(w)
                        
                if newly_uncovered:
                    # Find candidates that cover all newly_uncovered
                    # Intersection of cover_map[w] for w in newly_uncovered
                    cand_pool = None
                    for w in newly_uncovered:
                        candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                        if cand_pool is None:
                            cand_pool = set(candidates_w)
                        else:
                            cand_pool.intersection_update(candidates_w)
                        if not cand_pool:
                            break
                            
                    if cand_pool:
                        # Pick a random candidate or best one?
                        # Any candidate that covers the gap is valid for size preservation.
                        # We want to improve, so we just need to find a valid swap.
                        # To escape local minima, random is good.
                        c_add = rng.choice(list(cand_pool))
                        
                        # Perform swap
                        # Remove c_remove
                        for w in balls[c_remove]:
                            cover_count[w] -= 1
                        # Add c_add
                        for w in balls[c_add]:
                            cover_count[w] += 1
                        
                        code[i] = c_add
                        in_code[c_remove] = False
                        in_code[c_add] = True
                        improved = True
                        break # Restart loop
                        
                # 3. Try 2-Swap (Remove c_remove and c_other, add c_add)
                # This is more expensive. Only try if 1-swap failed.
                # We need to find c_other such that removing both leaves a set U
                # that can be covered by a single c_add.
                # This is O(N^2) per step if done naively.
                # Optimization: Only try a few random c_other.
                
                # Pick a few random other codewords to try removing with c_remove
                # If code is large, pick 5-10 random partners.
                partners = rng.sample(code, min(10, len(code)-1)) if len(code) > 1 else []
                
                for c_other in partners:
                    if c_other == c_remove:
                        continue
                    
                    # Temporarily remove c_remove and c_other
                    # Calculate U (uncovered words)
                    # Words with cover_count == 1 in balls[c_remove] or balls[c_other]
                    # And cover_count == 2 if in both?
                    
                    # Efficient check:
                    # Decrement counts for c_remove and c_other
                    # But we don't want to actually modify if it fails.
                    
                    # Let's simulate
                    # U = { w | cover_count[w] - (1 if w in ball(c_remove) else 0) - (1 if w in ball(c_other) else 0) == 0 }
                    
                    # To make this fast, we can iterate over the union of balls
                    # Union size is at most 30.
                    # Let's just compute the new cover counts for the affected words.
                    
                    affected = set(balls[c_remove]) | set(balls[c_other])
                    new_counts = {}
                    for w in affected:
                        cnt_w = cover_count[w]
                        if w in ball_sets[c_remove]:
                            cnt_w -= 1
                        if w in ball_sets[c_other]:
                            cnt_w -= 1
                        if cnt_w == 0:
                            new_counts[w] = 0 # Uncovered
                    # Wait, if cnt_w < 0, it's an error, but shouldn't happen if distinct.
                    
                    if not new_counts:
                        # Can remove both? That's a deletion of 2, which is handled by repeated deletions.
                        # If U is empty, we can just remove both.
                        # But local search usually does 1 move at a time.
                        # If U is empty, we can remove c_remove (deletion) and then c_other might become deletable.
                        # So this case is covered by deletion.
                        pass
                    else:
                        # U is non-empty. Find c_add that covers all w in U.
                        U = list(new_counts.keys())
                        cand_pool = None
                        for w in U:
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove and c2 != c_other]
                            if cand_pool is None:
                                cand_pool = set(candidates_w)
                            else:
                                cand_pool.intersection_update(candidates_w)
                            if not cand_pool:
                                break
                                
                        if cand_pool:
                            c_add = rng.choice(list(cand_pool))
                            
                            # Perform 2-swap: remove c_remove, c_other; add c_add
                            # Update cover_count
                            for w in balls[c_remove]:
                                cover_count[w] -= 1
                            for w in balls[c_other]:
                                cover_count[w] -= 1
                            for w in balls[c_add]:
                                cover_count[w] += 1
                                
                            # Update code list
                            # Remove c_remove and c_other, add c_add
                            # Find indices
                            idx_rm1 = code.index(c_remove)
                            idx_rm2 = code.index(c_other)
                            # Remove one, then the other (indices shift)
                            # Easier: create new list
                            new_code = []
                            for c in code:
                                if c != c_remove and c != c_other:
                                    new_code.append(c)
                            new_code.append(c_add)
                            
                            in_code[c_remove] = False
                            in_code[c_other] = False
                            in_code[c_add] = True
                            
                            code = new_code
                            improved = True
                            break # Restart loop
                            
                if improved:
                    break
                    
        return code

    best_result = None
    best_size = float('inf')
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        if restart == 0:
            selected = greedy_set_cover()
        else:
            selected = greedy_set_cover(seed=restart)
            
        result = local_search(selected, seed=restart)
        
        if len(result) < best_size:
            best_size = len(result)
            best_result = result
            if best_size <= 186:
                # Try to polish further?
                # If we hit the known best, we might stop or try to beat it.
                pass
        
        # If we have time, try to do more local search on the best result found so far?
        # The local_search function already loops until no improvement or time limit.
        
    if best_result is None:
        best_result = greedy_set_cover()
        
    # Final polish: try to remove redundant words again
    # local_search does this, but maybe one more pass?
    # The local_search loop breaks on 'improved', so it restarts.
    # It continues until no improvement in a full pass.
    # So it should be minimal.
    
    return [list(int_to_base3(i, n)) for i in best_result]
