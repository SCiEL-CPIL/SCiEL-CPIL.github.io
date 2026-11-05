import time
import random
import sys
import itertools

def construct(q, n, radius):
    # We assume q=3, n=6, radius=1 based on the task description.
    # Fallback for other cases to ensure validity if the evaluator passes different params.
    if not (q == 3 and n == 6 and radius == 1):
        # Simple greedy fallback for non-target cases
        all_words = list(itertools.product(range(q), repeat=n))
        code = []
        covered = set()
        for w in all_words:
            if w not in covered:
                code.append(list(w))
                for i in range(n):
                    for v in range(q):
                        if v == w[i]: continue
                        new_w = list(w)
                        new_w[i] = v
                        covered.add(tuple(new_w))
                covered.add(w)
        return code

    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin

    # Precomputation for q=3, n=6
    total_words = q ** n
    
    # Precompute neighbors (indices within distance 1) for all words
    # neighbors[i] is a set of indices j such that dist(i, j) <= 1
    neighbors = [set() for _ in range(total_words)]
    
    # Helper to convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Helper to convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    for i in range(total_words):
        w = idx_to_word(i)
        neighbors[i].add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neighbors[i].add(word_to_idx(tuple(new_w)))
        
    # Initial code: Direct Sum Construction (Tetracode x Tails)
    # Tetracode is a (9, 4, 3) code covering radius 1 for q=3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code_indices = set()
    counts = [0] * total_words
    
    for w in seed_words:
        idx = word_to_idx(tuple(w))
        code_indices.add(idx)
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    best_size = len(code_indices)
    best_code = list(code_indices)
    
    # Local Search Loop
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        # Iterate over a copy to allow modification
        for c in list(code_indices):
            # Check if c can be removed: all neighbors must have count > 1
            # If any neighbor has count 1, removing c makes it 0 (uncovered)
            can_remove = True
            for nb in neighbors[c]:
                if counts[nb] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in neighbors[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                removed_any = True
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                break # Restart removal loop
        
        if removed_any:
            continue
        
        # Phase 2: Local Search (Swaps)
        # If we can't remove, try to swap to create a removable word or improve slack.
        
        # Heuristic: Find a swap that maximizes the number of "fixed" critical words.
        # A critical word is one with count 1.
        # A swap (c_out, c_in) fixes a critical word u if:
        # 1. u is covered by c_in (u in neighbors[c_in])
        # 2. u is NOT covered by c_out (u not in neighbors[c_out]) -> so count increases
        # 3. u was critical (counts[u] == 1)
        
        # Also, the swap must be valid:
        # For all u in neighbors[c_out] \ neighbors[c_in], counts[u] must be >= 2.
        # (Because removing c_out decreases count by 1, and c_in doesn't cover it).
        
        best_swap = None
        best_score = -1
        
        # To speed up, we can limit the search or use randomization if full search is too slow.
        # With ~80 codewords and 729 total, full search is 80 * 650 = 52k checks.
        # Each check is O(13) for validity + O(13) for score. ~1000 ops per check.
        # 52k * 1000 = 52 million ops. Might be slow in Python (seconds).
        # Let's optimize:
        # 1. Precompute critical_words = {u for u in range(total_words) if counts[u] == 1}
        # 2. For validity, we only need to check u in neighbors[c_out] that are NOT in neighbors[c_in].
        
        critical_words = set(u for u in range(total_words) if counts[u] == 1)
        
        # Sample some c_out to speed up if needed, but let's try full first with optimizations.
        # Optimization: Only consider c_out that are "bad" (blocking removal).
        # Actually all c_out block removal if no removal is possible.
        
        # We can iterate c_out and sample c_in if time is tight, but let's try to find a good swap.
        
        # To make it faster, we can just look for ANY valid swap that increases the count of some critical word.
        # If we find one, we apply it.
        
        found_swap = False
        # Shuffle c_out order to avoid bias
        c_out_list = list(code_indices)
        random.shuffle(c_out_list)
        
        for c_out in c_out_list:
            if time.time() > deadline:
                break
            
            # Identify neighbors of c_out that are critical and NOT covered by c_in?
            # We don't know c_in yet.
            
            # Let's iterate c_in candidates.
            # To save time, we can prioritize c_in that cover many critical words.
            # But calculating that for all c_in is expensive.
            
            # Alternative: Randomized Local Search
            # Pick a random c_out.
            # Try a few random c_in.
            
            # Let's do a limited search:
            # For this c_out, try up to 100 random c_in.
            # If we find a valid swap that fixes >= 1 critical word, take it.
            
            # To be more effective, let's try to find a swap that maximizes fixed criticals.
            # We'll check a subset of c_in.
            
            # Generate random c_in not in code
            # Or iterate through all? 
            # Let's stick to a heuristic:
            # 1. Pick a random c_out.
            # 2. Pick a random c_in not in code.
            # 3. Check validity.
            # 4. Calculate score (fixed criticals).
            # 5. If score > 0, apply.
            
            # To improve quality, let's check a few c_in for each c_out.
            
            # Optimization: Precompute a list of words not in code? 
            # Or just generate random indices.
            
            # Let's try a small number of random swaps per c_out.
            # If we fail to find a good swap after trying all c_out with random c_in, 
            # we might get stuck. 
            # Let's try to be a bit smarter:
            # For a given c_out, which c_in are likely to be good?
            # c_in should cover critical words that c_out does NOT cover.
            
            # Let's collect critical words not covered by c_out.
            # criticals_not_covered_by_out = {u in critical_words if u not in neighbors[c_out]}
            
            # Then we want a c_in that covers many of these.
            # This is a set cover problem itself.
            
            # Given time constraints, let's use a simple randomized approach with a restart strategy.
            
            # Try 10 random c_in for this c_out
            for _ in range(10):
                c_in = random.randint(0, total_words - 1)
                if c_in in code_indices:
                    continue
                
                # Check validity
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbors[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Calculate score: number of critical words fixed
                # Fixed if u in critical_words AND u in neighbors[c_in] AND u not in neighbors[c_out]
                score = 0
                for u in neighbors[c_in]:
                    if u in critical_words and u not in neighbors[c_out]:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_swap = (c_out, c_in)
                    
                    # If we found a good swap, we can break early?
                    # Max score is small. Let's say if score >= 2, it's good.
                    if score >= 2:
                        break
            
            if best_score >= 2:
                break
        
        if best_swap:
            c_out, c_in = best_swap
            # Apply swap
            for nb in neighbors[c_out]:
                counts[nb] -= 1
            for nb in neighbors[c_in]:
                counts[nb] += 1
            code_indices.remove(c_out)
            code_indices.add(c_in)
            
            # Update critical_words? 
            # It changes, so we recalculate in next iteration or lazily.
            # Since we recalculate at the start of the loop if we don't remove, 
            # we should probably recalculate if we swap.
            # But the loop structure is: 
            # while time:
            #   try remove
            #   if removed: continue
            #   try swap
            #   if swapped: continue (implicit by loop end)
            
            # So we will recalculate critical_words in the next iteration? 
            # No, critical_words is calculated inside the "Phase 2" block.
            # So it will be updated next time we enter Phase 2.
            
        else:
            # No good swap found with random sampling.
            # Try a full search? Or restart?
            # Let's try a full search for c_in for a single c_out if we have time.
            # Or just break and return best if we are stuck for a long time?
            # To avoid infinite loop of "no swap found", we should perturb.
            # Let's apply a random valid swap even if score is 0, to escape local minima.
            
            swapped = False
            for _ in range(20):
                c_out = random.choice(list(code_indices))
                c_in = random.randint(0, total_words - 1)
                if c_in in code_indices:
                    continue
                
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbors[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                            
                if valid:
                    for nb in neighbors[c_out]:
                        counts[nb] -= 1
                    for nb in neighbors[c_in]:
                        counts[nb] += 1
                    code_indices.remove(c_out)
                    code_indices.add(c_in)
                    swapped = True
                    break
            
            if not swapped:
                # Completely stuck?
                # Return best found
                break

    return [list(idx_to_word(idx)) for idx in best_code]
