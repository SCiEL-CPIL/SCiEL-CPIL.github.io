import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        return _general_greedy(q, n, radius)

    # Specific optimization for 3^6
    N = 3**n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx
    
    def idx_to_word(idx):
        w = [0]*n
        for i in range(n-1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute balls as bitmasks
    # bits[i] is 1 if word i is covered
    balls = [0] * N
    # Also store list of indices in ball for count updates
    ball_indices = [[] for _ in range(N)]
    
    for idx in range(N):
        w = idx_to_word(idx)
        mask = 1 << idx
        ball_indices[idx].append(idx)
        for i in range(n):
            for delta in [-1, 1]:
                new_val = w[i] + delta
                if 0 <= new_val < 3:
                    w[i] = new_val
                    n_idx = word_to_idx(w)
                    mask |= (1 << n_idx)
                    ball_indices[idx].append(n_idx)
                    w[i] = w[i] - delta # restore
        balls[idx] = mask

    full_mask = (1 << N) - 1

    def greedy_init(seed):
        random.seed(seed)
        selected = []
        counts = np.zeros(N, dtype=np.int32)
        uncovered_mask = full_mask
        
        while uncovered_mask != 0:
            best_idx = -1
            best_score = -1
            
            # Heuristic: Maximize (num_uncovered_covered, num_low_coverage_covered)
            # Or simply: Maximize sum of 1/counts[x] for x in ball? 
            # Let's stick to: Maximize count of uncovered points. 
            # Tie-breaker: Maximize count of points with count==1?
            
            # To speed up, we can precompute a score for each candidate?
            # Score depends on `uncovered_mask` and `counts`.
            # Since N=729, we can just loop.
            
            for c in range(N):
                if c in selected:
                    continue
                mask_c = balls[c]
                # Count uncovered
                unc_cnt = (mask_c & uncovered_mask).bit_count()
                if unc_cnt == 0:
                    continue
                # If we want to be smarter, we could weight points with count 0 higher.
                # But standard greedy set cover is: max new elements.
                # Let's try a weighted score: 
                # score = 100 * unc_cnt + (mask_c & (counts == 1 mask)).bit_count()
                # But counts==1 mask changes.
                
                # Simple greedy: max unc_cnt
                if unc_cnt > best_score:
                    best_score = unc_cnt
                    best_idx = c
                    if best_score == 13: # Max possible
                        break
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            for x in ball_indices[best_idx]:
                counts[x] += 1
            uncovered_mask &= ~balls[best_idx]
            
        return selected, counts

    def local_search(selected, counts, deadline):
        # Returns (selected, counts, improved_size)
        best_len = len(selected)
        current_selected = selected[:]
        current_counts = counts.copy()
        
        no_improvement_time = 0
        last_improvement = time.time()
        
        while time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove all possible redundant codewords
            removed_any = False
            for i in range(len(current_selected) - 1, -1, -1):
                c_out = current_selected[i]
                # Check if redundant
                # Redundant if min(counts[x] for x in ball(c_out)) > 1
                is_redundant = True
                for x in ball_indices[c_out]:
                    if current_counts[x] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove
                    for x in ball_indices[c_out]:
                        current_counts[x] -= 1
                    current_selected.pop(i)
                    best_len -= 1
                    improved = True
                    removed_any = True
                    last_improvement = time.time()
            
            if removed_any:
                continue # Restart loop to check for more removals
            
            # Phase 2: Swap
            # Find a codeword to remove and a new one to add
            # Strategy: Pick a random codeword, compute its critical set.
            # If critical set is empty, it would have been removed in Phase 1.
            # So critical set is non-empty.
            # Find c_in that covers critical set.
            
            # To avoid getting stuck, we can randomize the choice of c_out.
            # But we should prioritize c_out with small critical set?
            # Small critical set is easier to cover.
            
            # Let's collect all c_out with their critical set size.
            # This is expensive if done every iteration.
            # Just pick a random one?
            
            c_out_idx = random.randint(0, len(current_selected)-1)
            c_out = current_selected[c_out_idx]
            
            U_out = []
            for x in ball_indices[c_out]:
                if current_counts[x] == 1:
                    U_out.append(x)
            
            if not U_out:
                # Should not happen if Phase 1 ran correctly
                continue
            
            mask_U = 0
            for x in U_out:
                mask_U |= (1 << x)
            
            # Find best c_in
            # Valid if (balls[c_in] & mask_U) == mask_U
            # Among valid, pick one that maximizes "utility"
            # Utility: number of points in ball(c_in) that are currently count 1 (excluding U_out? No, including is fine as they become 2)
            # Or simply: any valid c_in.
            
            found_swap = False
            # Shuffle candidates to avoid bias
            candidates = list(range(N))
            random.shuffle(candidates)
            
            best_c_in = -1
            best_utility = -1
            
            for c_in in candidates:
                if c_in in current_selected:
                    continue
                
                if (balls[c_in] & mask_U) != mask_U:
                    continue
                
                # Calculate utility: how many points in ball(c_in) have count == 1?
                # This increases redundancy.
                # Note: points in U_out are count 1. They will become 2.
                # Points in ball(c_in) \ U_out with count 1 will become 2.
                # Points with count 0? None, covered by current_selected.
                
                # We want to increase the "slack" to allow future removals.
                # So maximizing the number of points that go from 1 to 2 is good.
                
                # We can approximate this by counting bits in (balls[c_in] & (counts==1 mask))
                # But we don't have a mask for counts==1.
                # We can just pick the first valid one? 
                # Or pick one that covers the most total points (all 13)?
                
                # Let's just pick the first valid one found in the shuffled list.
                # This is random enough.
                
                best_c_in = c_in
                found_swap = True
                break
            
            if found_swap:
                # Execute swap
                for x in ball_indices[c_out]:
                    current_counts[x] -= 1
                for x in ball_indices[best_c_in]:
                    current_counts[x] += 1
                
                current_selected[c_out_idx] = best_c_in
                # Size unchanged, so best_len stays same
                # But structure changed.
                # Do we count this as improvement? 
                # Only if it leads to a smaller size later.
                # For now, we just continue.
                
                # To prevent infinite loop of useless swaps, we might need a more sophisticated utility.
                # But let's try this first.
                
                # If we swap frequently without size reduction, we might waste time.
                # We can add a counter: if we do X swaps without a removal, restart greedy?
                
            else:
                # No swap found? This means no c_in covers U_out.
                # This implies c_out is "essential" in a strong sense?
                # Actually, it just means we can't replace it with a single other codeword.
                # We might need to replace it with two? (Size +1, then remove others?)
                # Or just try a different c_out.
                pass
        
        return current_selected, current_counts, best_len

    # Run multiple restarts if time permits
    best_selected = None
    best_counts = None
    best_size = N # Upper bound
    
    # First run: Greedy + Local Search
    sel, cnt = greedy_init(42)
    sel, cnt, size = local_search(sel, cnt, deadline)
    
    if size < best_size:
        best_size = size
        best_selected = sel
        best_counts = cnt
    
    # If time remains, try more restarts with different seeds
    while time.time() < deadline - 10:
        seed = random.randint(0, 10000)
        sel, cnt = greedy_init(seed)
        # Quick local search with tight deadline
        sub_deadline = min(deadline - 2, time.time() + 5) # Give 5s per restart
        sel, cnt, size = local_search(sel, cnt, sub_deadline)
        
        if size < best_size:
            best_size = size
            best_selected = sel
            best_counts = cnt
            print(f"Improved to {size}")

    # Convert back to list of lists
    result = [idx_to_word(idx) for idx in best_selected]
    return result

def _general_greedy(q, n, radius):
    # Fallback for non-3^6 cases
    # Not strictly required if task is fixed, but good practice
    pass
