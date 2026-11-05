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
    
    # 2. Precompute neighbors (balls of radius 1) as index lists
    # ball_idx[i] is a list of indices covered by word i
    ball_idx = [[] for _ in range(N)]
    # cover_map[j] is a list of indices i such that j is in ball_idx[i]
    cover_map = [[] for _ in range(N)]
    
    for i, w in enumerate(words):
        w_list = list(w)
        # Self
        ball_idx[i].append(i)
        cover_map[i].append(i)
        
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                ball_idx[i].append(idx)
                cover_map[idx].append(i)
                w_list[pos] = orig_val

    # 3. Greedy Set Cover with heap-based gain computation
    def greedy_cover(uncovered_mask, current_code, current_counts):
        code = set(current_code)
        counts = np.array(current_counts, dtype=np.int32)
        
        # We need to track which points are uncovered.
        # Instead of a bitset, we use a boolean array for speed in gain updates.
        uncovered_bool = np.zeros(N, dtype=bool)
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            uncovered_bool[idx] = True
            m ^= lsb
        
        # Initialize gains: for each candidate i, gain[i] = count of uncovered points in ball_idx[i]
        gains = np.zeros(N, dtype=np.int32)
        for i in range(N):
            gains[i] = uncovered_bool[ball_idx[i]].sum()
        
        # Heap: (-gain, index)
        # We only push indices with gain > 0
        heap = [(-gains[i], i) for i in range(N) if gains[i] > 0]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 5:
                break
                
            neg_gain, best_idx = heapq.heappop(heap)
            # Check if gain is still valid (lazy deletion)
            if gains[best_idx] != -neg_gain:
                continue
            
            if gains[best_idx] == 0:
                break
                
            code.add(best_idx)
            
            # Update counts
            for j in ball_idx[best_idx]:
                counts[j] += 1
                if uncovered_bool[j]:
                    uncovered_bool[j] = False
                    # Update gains for all candidates covering j
                    for c in cover_map[j]:
                        if gains[c] > 0:
                            gains[c] -= 1
                            if gains[c] > 0:
                                heapq.heappush(heap, (-gains[c], c))
                    # Update uncovered_mask
                    uncovered_mask &= ~(1 << best_idx) # This is wrong, we need to clear bit j
                    # Actually, let's just return the code and counts, and recompute uncovered_mask if needed
                    # But for the loop condition, we can just check if heap is empty or all gains 0.
                    # Let's keep a counter of uncovered points.
                    
        return code, counts.tolist(), uncovered_mask

    # Let's refine the greedy to properly track uncovered count and mask
    def greedy_cover_fast(uncovered_mask, current_code, current_counts):
        code = set(current_code)
        counts = np.array(current_counts, dtype=np.int32)
        
        uncovered_bool = np.zeros(N, dtype=bool)
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            uncovered_bool[idx] = True
            m ^= lsb
        
        gains = np.zeros(N, dtype=np.int32)
        for i in range(N):
            gains[i] = uncovered_bool[ball_idx[i]].sum()
        
        heap = [(-gains[i], i) for i in range(N) if gains[i] > 0]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 5:
                break
                
            neg_gain, best_idx = heapq.heappop(heap)
            if gains[best_idx] != -neg_gain:
                continue
            
            if gains[best_idx] == 0:
                break
                
            code.add(best_idx)
            
            for j in ball_idx[best_idx]:
                counts[j] += 1
                if uncovered_bool[j]:
                    uncovered_bool[j] = False
                    uncovered_mask &= ~(1 << j)
                    for c in cover_map[j]:
                        if gains[c] > 0:
                            gains[c] -= 1
                            if gains[c] > 0:
                                heapq.heappush(heap, (-gains[c], c))
                    
        return code, counts.tolist(), uncovered_mask

    # Initial Greedy
    initial_uncovered = (1 << N) - 1
    initial_counts = [0] * N
    best_code, best_counts, _ = greedy_cover_fast(initial_uncovered, set(), initial_counts)
    
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
    
    # Swap-based local search: try replacing one codeword with another
    def swap_improve(code_set, counts):
        """Try to find a swap that reduces code size or keeps it same with better structure."""
        improved = False
        code_list = list(code_set)
        n_code = len(code_list)
        
        for i in range(n_code):
            if time.time() > deadline - 3:
                break
            c1 = code_list[i]
            # For each candidate replacement c2 not in code
            # c1 can be removed if all points in ball_idx[c1] are covered by other codewords
            # i.e., counts[j] >= 2 for all j in ball_idx[c1]
            # But we also want to try swaps: remove c1, add c2, see if we can then delete more
            
            # First check if c1 is redundant (can be deleted outright)
            c1_redundant = True
            for j in ball_idx[c1]:
                if counts[j] <= 1:
                    c1_redundant = False
                    break
            
            if c1_redundant:
                code_set.remove(c1)
                for j in ball_idx[c1]:
                    counts[j] -= 1
                improved = True
                # Restart the loop since code changed
                return True
        
        # Try swaps: for each c1 in code, try replacing with c2 not in code
        # such that the resulting code is still valid and we can then polish to get smaller
        for i in range(n_code):
            if time.time() > deadline - 3:
                break
            c1 = code_list[i]
            
            # Points that would become uncovered if we remove c1
            # These are points j in ball_idx[c1] where counts[j] == 1
            critical_points = []
            for j in ball_idx[c1]:
                if counts[j] == 1:
                    critical_points.append(j)
            
            if not critical_points:
                continue  # c1 is redundant, handled above
            
            # Try a few random candidates for c2
            # c2 must cover all critical_points
            # Pre-filter: c2 must have all critical_points in ball_idx[c2]
            # This is expensive, so just try random candidates
            
            for attempt in range(20):
                if time.time() > deadline - 3:
                    break
                c2 = random.randrange(N)
                if c2 in code_set:
                    continue
                
                # Check if c2 covers all critical points
                c2_ball_set = set(ball_idx[c2])
                covers_all = all(j in c2_ball_set for j in critical_points)
                if not covers_all:
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
                
        # Greedy repair using ball_idx lists
        # Use the fast greedy with heap
        curr_code, curr_counts, _ = greedy_cover_fast(uncovered, curr_code, curr_counts)
        
        # 3. Polish: Remove redundant codewords
        polish(curr_code, curr_counts)
        
        # 3b. Swap-based improvement
        swap_improve(curr_code, curr_counts)
        
        # 4. Accept if better
        if len(curr_code) < best_size:
            best_code = curr_code
            best_counts = curr_counts
            best_size = len(best_code)
            
    return [list(words[i]) for i in best_code]
