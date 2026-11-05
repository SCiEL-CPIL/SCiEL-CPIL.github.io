import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Handle specific task parameters
    # The task is K_3(7,1), so q=3, n=7, radius=1
    # We will implement a generic solver but optimized for this case.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task, but good for robustness)
        # Simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        def get_neighbors(w):
            nbs = [w]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    nbs.append(tuple(new_w))
            return nbs
        
        neighbors = [get_neighbors(w) for w in all_words]
        neighbors_idx = [[word_to_idx[t] for t in nbs] for nbs in neighbors]
        
        uncovered = set(range(N))
        code = []
        
        while uncovered:
            best_c = -1
            max_cov = 0
            for c in range(N):
                if c in code:
                    continue
                cnt = 0
                for nb in neighbors_idx[c]:
                    if nb in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for nb in neighbors_idx[best_c]:
                uncovered.discard(nb)
        
        return [list(all_words[c]) for c in code]

    # Specific optimization for q=3, n=7, radius=1
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer for 120s limit
    
    N = 3 ** n  # 2187
    
    # 1. Precompute neighbors as a dense NumPy array
    # Shape: (N, 1 + n*(q-1)) = (2187, 15)
    # neighbors_arr[i] contains the indices of word i and all words at Hamming distance 1.
    
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

    neighbors_arr = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    
    for i in range(N):
        w = idx_to_tuple(i)
        neighbors_arr[i, 0] = i
        col = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                neighbors_arr[i, col] = tuple_to_idx(new_w)
                col += 1

    # 2. Greedy Set-Cover Construction
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code_set = set()
    code_list = []
    
    # Initial step: pick word 0
    code_list.append(0)
    code_set.add(0)
    coverage[neighbors_arr[0]] += 1
    uncovered_mask[neighbors_arr[0]] = False
    
    # Greedy loop
    while uncovered_mask.any() and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        # We can vectorize this calculation:
        # count_uncovered[c] = sum(uncovered_mask[neighbors_arr[c]])
        # This is a matrix multiplication or reduction.
        # neighbors_arr is (N, 15). uncovered_mask is (N,).
        # We want sum over axis 1 of uncovered_mask[neighbors_arr].
        # This is effectively neighbors_arr @ uncovered_mask if we treat neighbors as one-hot? No.
        # It's a gather.
        # np.take(uncovered_mask, neighbors_arr).sum(axis=1) is efficient.
        
        counts = np.take(uncovered_mask, neighbors_arr).sum(axis=1)
        
        # Exclude already chosen codewords
        for c in code_set:
            counts[c] = 0
            
        best_c = np.argmax(counts)
        best_count = counts[best_c]
        
        if best_count == 0:
            # Should not happen if uncovered_mask.any() is true, but safety check
            break
            
        code_list.append(int(best_c))
        code_set.add(int(best_c))
        coverage[neighbors_arr[best_c]] += 1
        uncovered_mask[neighbors_arr[best_c]] = False

    best_size = len(code_set)
    best_code_set = set(code_set)
    
    # 3. Local Search / Improvement
    # We will use a loop that tries to reduce the size.
    # Strategies:
    # a) Remove redundant codewords.
    # b) Kick: Remove k random codewords, repair, accept if size decreases.
    
    random.seed(42)
    no_improvement_counter = 0
    max_no_improvement = 50
    
    while time.time() < deadline:
        # Check for redundancy
        removed_any = False
        for c in list(code_set):
            # A codeword is redundant if all its neighbors are covered by at least one other codeword.
            # i.e., min(coverage[neighbors[c]]) > 1
            if np.min(coverage[neighbors_arr[c]]) > 1:
                # Remove it
                code_set.discard(c)
                code_list.remove(c)
                coverage[neighbors_arr[c]] -= 1
                removed_any = True
                best_size = len(code_set)
                best_code_set = set(code_set)
                no_improvement_counter = 0
                # Restart redundancy check as removal might expose more redundancies
                break
        
        if removed_any:
            continue
            
        # Try a kick
        # Pick k random codewords to remove (k=1, 2, or 3)
        k = random.choice([1, 2, 3])
        if len(code_set) <= k:
            break
            
        removed_indices = random.sample(list(code_set), k)
        
        # Temporarily remove
        temp_coverage = coverage.copy()
        temp_code_set = set(code_set)
        for c in removed_indices:
            temp_coverage[neighbors_arr[c]] -= 1
            temp_code_set.discard(c)
        
        # Find uncovered words
        uncovered_idx = np.where(temp_coverage == 0)[0]
        
        if len(uncovered_idx) == 0:
            # Perfect removal
            code_set = temp_code_set
            coverage = temp_coverage
            code_list = list(code_set)
            best_size = len(code_set)
            best_code_set = set(code_set)
            no_improvement_counter = 0
            continue
            
        # Greedy Repair
        # We need to add codewords to cover uncovered_idx.
        # We will maintain a local coverage for the repair process to be fast,
        # but since we start from temp_coverage, we can just update it.
        
        added_indices = []
        current_uncovered = set(uncovered_idx.tolist())
        
        # Candidate pool: neighbors of uncovered words
        # We can just iterate through all N candidates and pick the best one each step.
        # N=2187, steps ~ 10-20. 2187*20 = 43k ops. Fast.
        
        while current_uncovered and time.time() < deadline:
            best_c = -1
            max_gain = 0
            
            # Optimization: Only consider candidates that are neighbors of uncovered words
            # But iterating all N is fine for N=2187.
            for c in range(N):
                if c in temp_code_set:
                    continue
                    
                # Count how many currently uncovered words are covered by c
                # c covers neighbors_arr[c].
                # Intersection of neighbors_arr[c] and current_uncovered.
                # We can check if any neighbor is in current_uncovered.
                # Since current_uncovered is a set, we can iterate neighbors_arr[c]
                # and check membership.
                
                gain = 0
                for nb in neighbors_arr[c]:
                    if nb in current_uncovered:
                        gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_c = c
                            if gain == len(current_uncovered):
                                break
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
                    if max_gain == len(current_uncovered):
                        break
                        
            if best_c == -1 or max_gain == 0:
                # Cannot repair
                break
                
            added_indices.append(best_c)
            temp_code_set.add(best_c)
            temp_coverage[neighbors_arr[best_c]] += 1
            
            # Update current_uncovered
            for nb in neighbors_arr[best_c]:
                if nb in current_uncovered:
                    current_uncovered.discard(nb)
                    
        new_size = len(temp_code_set)
        
        if new_size < len(code_set):
            # Accept the change
            code_set = temp_code_set
            coverage = temp_coverage
            code_list = list(code_set)
            best_size = new_size
            best_code_set = set(code_set)
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        if no_improvement_counter > max_no_improvement:
            # Restart with a stronger kick or break?
            # Let's just break to save time if stuck, or increase max_no_improvement?
            # Given the time limit, breaking is safer if we've been stuck for a while.
            # However, we might just be in a local minimum.
            # Let's try a few more random kicks with larger k.
            if no_improvement_counter > 200:
                break
                
    # Return the best code found
    # Ensure we return the best set found
    if len(best_code_set) < len(code_set):
        final_code_set = best_code_set
    else:
        final_code_set = code_set
        
    # Convert indices back to words
    result = [list(idx_to_tuple(c)) for c in final_code_set]
    
    # Final verification (optional, but good for debugging/safety)
    # For speed, we skip full verification in production if we trust the logic,
    # but let's do a quick check on the size.
    
    return result
