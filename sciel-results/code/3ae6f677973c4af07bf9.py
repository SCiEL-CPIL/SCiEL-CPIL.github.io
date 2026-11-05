import numpy as np
import itertools
import time
import random
import sys
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1) as bitsets AND index lists
    ball_masks = [0] * N
    ball_idx = [[] for _ in range(N)]
    for i, w in enumerate(words):
        mask = 0
        idx_list = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                mask |= (1 << idx)
                idx_list.append(idx)
                w_list[pos] = orig_val
        ball_masks[i] = mask
        ball_idx[i] = idx_list
    
    # Precompute ball_idx as numpy arrays for vectorized ops
    ball_idx_np = [np.array(b, dtype=np.int32) for b in ball_idx]
    
    # Precompute bit masks as a list of big ints for fast intersection
    # ball_masks[i] already exists
    
    UNIVERSE = (1 << N) - 1

    # 3. Greedy Set Cover with bitset-accelerated gain computation
    def greedy_cover(uncovered_mask, current_code, current_counts):
        code = set(current_code)
        counts = np.array(current_counts, dtype=np.int32)
        
        while uncovered_mask:
            if time.time() > deadline - 5:
                break
            
            # Vectorized gain: for each candidate i, gain = popcount(ball_masks[i] & uncovered_mask)
            # Use numpy with uint64 chunks for speed
            # Convert uncovered_mask to numpy uint64 array
            num_chunks = (N + 63) // 64
            unc_chunks = np.zeros(num_chunks, dtype=np.uint64)
            m = uncovered_mask
            for chunk_i in range(num_chunks):
                if m == 0:
                    break
                unc_chunks[chunk_i] = m & ((1 << 64) - 1)
                m >>= 64
            
            # For each candidate, compute gain = popcount(ball_masks[i] & uncovered_mask)
            # This is O(N * num_chunks) which is 2187 * 35 ≈ 76k ops
            gains = np.zeros(N, dtype=np.int32)
            for i in range(N):
                bm = ball_masks[i]
                gain = 0
                for chunk_i in range(num_chunks):
                    if unc_chunks[chunk_i] == 0:
                        continue
                    gain += (bm & unc_chunks[chunk_i]).bit_count()
                gains[i] = gain
            
            best_idx = int(np.argmax(gains))
            max_gain = gains[best_idx]
            
            if max_gain == 0:
                break
                
            code.add(best_idx)
            # Update counts using numpy
            counts[ball_idx_np[best_idx]] += 1
            
            # Update uncovered_mask
            uncovered_mask &= ~ball_masks[best_idx]
                    
        return code, counts.tolist(), uncovered_mask

    # Initial Greedy
    initial_uncovered = UNIVERSE
    initial_counts = [0] * N
    best_code, best_counts, _ = greedy_cover(initial_uncovered, set(), initial_counts)
    
    # Polish (deletion) using precomputed ball_idx lists
    def polish(code_set, counts):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for c in list(code_set):
                # Check if c is redundant: all points in ball_idx[c] have count > 1
                redundant = True
                for j in ball_idx[c]:
                    if counts[j] <= 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for j in ball_idx[c]:
                        counts[j] -= 1
                    changed = True
    
    polish(best_code, best_counts)
    best_size = len(best_code)
    
    # 4. Kick-Repair Local Search with swap-based improvement
    kick_sizes = [1, 2, 3, 4, 5]
    
    # 1-for-1 replacement: remove c1, add c2 that covers all critical points of c1
    def replacement_improve(code_set, counts):
        """Try 1-for-1 replacements that may enable further polish to reduce size."""
        improved = False
        code_list = list(code_set)
        n_code = len(code_list)
        
        for i in range(n_code):
            if time.time() > deadline - 3:
                break
            c1 = code_list[i]
            
            # Points that would become uncovered if we remove c1
            critical_points = []
            for j in ball_idx[c1]:
                if counts[j] == 1:
                    critical_points.append(j)
            
            if not critical_points:
                # c1 is redundant
                code_set.remove(c1)
                for j in ball_idx[c1]:
                    counts[j] -= 1
                improved = True
                return True
            
            # Try random candidates for c2 that cover all critical points
            crit_set = set(critical_points)
            for attempt in range(30):
                if time.time() > deadline - 3:
                    break
                c2 = random.randrange(N)
                if c2 in code_set:
                    continue
                
                # Check if c2 covers all critical points using bitset
                if (ball_masks[c2] & UNIVERSE) == 0:
                    continue
                # Check coverage of critical points
                covers = True
                for j in critical_points:
                    if not (ball_masks[c2] >> j) & 1:
                        covers = False
                        break
                if not covers:
                    continue
                
                # Perform swap
                code_set.remove(c1)
                for j in ball_idx[c1]:
                    counts[j] -= 1
                code_set.add(c2)
                for j in ball_idx[c2]:
                    counts[j] += 1
                
                # Try to polish
                polish(code_set, counts)
                
                # Check if size decreased
                if len(code_set) < n_code:
                    improved = True
                    return True
                else:
                    # Revert swap
                    code_set.remove(c2)
                    for j in ball_idx[c2]:
                        counts[j] -= 1
                    code_set.add(c1)
                    for j in ball_idx[c1]:
                        counts[j] += 1
                    # Re-polish to restore state
                    polish(code_set, counts)
        
        return improved
    
    while time.time() < deadline - 5:
        # Select a kick size
        k = random.choice(kick_sizes)
        
        # Copy current best
        curr_code = set(best_code)
        curr_counts = best_counts[:]
        
        # 1. Kick: Remove k random codewords
        if len(curr_code) <= k:
            continue
            
        code_list = list(curr_code)
        random.shuffle(code_list)
        to_remove = code_list[:k]
        
        for c in to_remove:
            curr_code.remove(c)
            for j in ball_idx[c]:
                curr_counts[j] -= 1
                
        # 2. Repair: Cover the holes
        uncovered = 0
        for j in range(N):
            if curr_counts[j] == 0:
                uncovered |= (1 << j)
                
        # Greedy repair using bitset-accelerated gain
        while uncovered:
            if time.time() > deadline - 2:
                break
            
            # Find best candidate using bitset popcount
            num_chunks = (N + 63) // 64
            unc_chunks = np.zeros(num_chunks, dtype=np.uint64)
            m = uncovered
            for chunk_i in range(num_chunks):
                if m == 0:
                    break
                unc_chunks[chunk_i] = m & ((1 << 64) - 1)
                m >>= 64
            
            best_idx = -1
            max_gain = 0
            for i in range(N):
                bm = ball_masks[i]
                gain = 0
                for chunk_i in range(num_chunks):
                    if unc_chunks[chunk_i] == 0:
                        continue
                    gain += (bm & unc_chunks[chunk_i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if max_gain == 0:
                break
                
            curr_code.add(best_idx)
            for j in ball_idx[best_idx]:
                curr_counts[j] += 1
            
            # Update uncovered
            uncovered &= ~ball_masks[best_idx]
            
        # 3. Polish: Remove redundant codewords
        polish(curr_code, curr_counts)
        
        # 3b. Replacement-based improvement
        replacement_improve(curr_code, curr_counts)
        
        # 4. Accept if better
        if len(curr_code) < best_size:
            best_code = curr_code
            best_counts = curr_counts
            best_size = len(best_code)
            
    return [list(words[i]) for i in best_code]
