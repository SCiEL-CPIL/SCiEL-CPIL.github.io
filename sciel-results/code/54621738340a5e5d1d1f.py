"""Multi-Start Local Search with Redundancy Removal for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_neighbors(q, n, radius):
    """Precompute neighbors for all words."""
    N = q ** n
    nbrs_list = [None] * N
    
    if q == 3 and n == 7:
        # Unrolled fast path for specific case
        for idx in range(N):
            word = [0] * 7
            tmp = idx
            word[0] = tmp % 3; tmp //= 3
            word[1] = tmp % 3; tmp //= 3
            word[2] = tmp % 3; tmp //= 3
            word[3] = tmp % 3; tmp //= 3
            word[4] = tmp % 3; tmp //= 3
            word[5] = tmp % 3; tmp //= 3
            word[6] = tmp % 3; tmp //= 3
            
            nbrs = [idx]
            for pos in range(7):
                old = word[pos]
                for v in range(3):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = new_word[0] + new_word[1]*3 + new_word[2]*9 + new_word[3]*27 + new_word[4]*81 + new_word[5]*243 + new_word[6]*729
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
    else:
        for idx in range(N):
            word = [0] * n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            nbrs = [idx]
            for pos in range(n):
                old = word[pos]
                for v in range(q):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = 0
                        mul = 1
                        for i in range(n):
                            new_idx += new_word[i] * mul
                            mul *= q
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
            
    return nbrs_list

def _local_search(nbrs_list, N, time_limit, start_time, rng):
    """
    Perform local search starting from a random initial set.
    Strategy:
    1. Initialize with random set of size approx N / (radius+1) or slightly more.
    2. Iteratively:
       a. Remove redundant codewords.
       b. If not fully covered, add the best codeword (max uncovered coverage).
       c. If fully covered, try to reduce size by swapping? (Optional, might be slow)
    """
    # Initial size heuristic: N / 15 is lower bound ~146. Let's start with 180-200.
    # The optimal is 186.
    init_size = int(N / 12) # ~182
    
    # Random initialization
    all_indices = np.arange(N)
    selected_indices = rng.choice(all_indices, size=init_size, replace=False)
    selected_set = set(selected_indices.tolist())
    
    # Coverage count array
    covered_count = np.zeros(N, dtype=np.int32)
    for idx in selected_indices:
        covered_count[nbrs_list[idx]] += 1
        
    best_size = len(selected_set)
    best_selected = list(selected_set)
    
    while time.time() - start_time < time_limit:
        # 1. Remove Redundant Codewords
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            # Iterate through a copy to allow removal
            for idx in list(selected_set):
                # Check if idx is critical: i.e., there is some v in N(idx) 
                # such that covered_count[v] == 1.
                # If no such v, idx is redundant.
                nbrs_idx = nbrs_list[idx]
                # We need to check if any neighbor has count 1.
                # If all neighbors have count > 1, it's redundant.
                # Note: If a neighbor has count 0, it's uncovered, so idx is critical (covers it).
                # Actually, if a neighbor has count 0, idx is the ONLY one covering it? 
                # No, if count is 0, it means NO selected codeword covers it. 
                # Wait, covered_count[v] is the number of selected codewords covering v.
                # If covered_count[v] == 0, v is uncovered.
                # If we remove idx, and v was covered by idx, then covered_count[v] becomes -1? 
                # No, we must ensure we only remove if it doesn't uncover anything.
                
                # A codeword idx is redundant if for all v in N(idx), covered_count[v] >= 2.
                # Why? If covered_count[v] >= 2, then removing idx leaves covered_count[v] >= 1, so v is still covered.
                # If covered_count[v] == 1, removing idx makes it 0 (uncovered).
                # If covered_count[v] == 0, that's impossible if idx covers v and idx is selected.
                
                is_redundant = np.all(covered_count[nbrs_idx] >= 2)
                
                if is_redundant:
                    # Remove idx
                    covered_count[nbrs_idx] -= 1
                    selected_set.remove(idx)
                    changed = True
                    break # Restart check as counts changed
        
        # Update best if improved
        if len(selected_set) < best_size:
            best_size = len(selected_set)
            best_selected = list(selected_set)
            
        # Check if fully covered
        # Uncovered points are those with covered_count == 0
        uncovered_mask = (covered_count == 0)
        num_uncovered = np.sum(uncovered_mask)
        
        if num_uncovered == 0:
            # Fully covered. Can we improve further? 
            # Try to see if we can remove more? The redundancy loop above should have caught all.
            # Try Swaps? 
            # For simplicity and speed, we just break and restart with a different seed.
            break
        
        # 2. Add Best Codeword
        # Gain of adding a codeword w is the number of currently uncovered points in N(w).
        # gain[w] = sum(uncovered_mask[nbrs_list[w]])
        
        # Vectorized calculation
        # We need to sum uncovered_mask over neighbors for each w.
        # This is equivalent to: for each w, count how many of its neighbors are in uncovered set.
        
        # Efficient way:
        # Create a boolean array U where U[v] = 1 if v is uncovered.
        # We want gain[w] = sum_{v in N(w)} U[v].
        
        # We can precompute a matrix of neighbors? 
        # Or just loop? N=2187. 
        # Looping 2187 times, each checking 15 neighbors is ~32k ops. Very fast.
        
        best_gain = -1
        best_idx_to_add = -1
        
        for w in range(N):
            if w in selected_set:
                continue
            nbrs_w = nbrs_list[w]
            gain = np.sum(uncovered_mask[nbrs_w])
            if gain > best_gain:
                best_gain = gain
                best_idx_to_add = w
                
        if best_gain == 0:
            # No codeword can cover any new uncovered point? 
            # This implies we are stuck or something is wrong.
            # Actually, if num_uncovered > 0, there must be some codeword covering them.
            # Unless the space is disconnected? No, radius 1 covers everything.
            break
            
        # Add the best codeword
        selected_set.add(best_idx_to_add)
        covered_count[nbrs_list[best_idx_to_add]] += 1
        
    return best_selected

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for conversion
    
    nbrs_list = _precompute_neighbors(q, n, radius)
    N = q ** n
    
    best_code = None
    best_size = float('inf')
    
    # Multi-start
    # We run multiple iterations. Each iteration has a sub-time-limit.
    # Or we just run until time runs out, keeping the global best.
    
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < time_limit - 5.0:
        # Run one local search instance
        # Give it a generous time slice, e.g., 5 seconds? 
        # Or just let it run until it converges (which is fast for this size).
        # Convergence is usually very fast (< 0.1s).
        
        result = _local_search(nbrs_list, N, time_limit - 5.0, start_time, rng)
        
        if len(result) < best_size:
            best_size = len(result)
            best_code = result.copy()
            
        # If we found a very good solution (close to lower bound 156), we might want to stop or refine.
        # But let's keep searching for better ones.
        
    # Convert indices to words
    res = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        res.append(word)
        
    return res
# EVOLVE-BLOCK-END
