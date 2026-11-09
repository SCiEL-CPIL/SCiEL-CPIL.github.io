import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*(q-1) = 15
    
    t_start = time.time()
    deadline = 115.0 # Safety margin (limit is 120s)
    
    # Precompute all words as indices 0..2186
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx

    # Precompute balls: balls[i] is a list of indices of words within distance radius of word i
    ball_size = 1 + n * (q - 1)
    balls = np.empty((N, ball_size), dtype=np.int32)
    
    for i, w in enumerate(all_words):
        ball = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball
        
    # Bitset representation for fast coverage tracking
    # N=2187, so we need ceil(2187/64) = 35 uint64s per bitset
    num_chunks = (N + 63) // 64
    chunk_indices = np.arange(N) // 64
    bit_masks = (1 << (np.arange(N) % 64)).astype(np.uint64)
    
    # Precompute ball bitsets: ball_bitsets[i] is a list of (chunk_idx, mask) pairs
    # This allows fast addition/removal via XOR or OR/ANDNOT
    # Actually, for coverage counts, we need to know if count > 0.
    # A simpler approach for greedy: maintain a boolean array 'covered' and a count array 'counts'.
    # But for "gain" calculation, we need to know how many NEW points are covered.
    # Gain(i) = number of points in ball[i] that are NOT in covered.
    
    # Let's use a hybrid: 
    # 1. Greedy construction using vectorized gain calculation.
    #    We maintain a boolean array 'covered' (N,).
    #    To calculate gains for all i at once is expensive if done naively.
    #    However, N is small (2187). 
    #    Let's try a different bitset approach:
    #    Maintain an array of uint64 chunks representing the union of all selected balls.
    #    This tells us *which* points are covered, but not *how many* times.
    #    For pure greedy set cover, we only care if it's covered or not.
    #    So, Gain(i) = popcount(ball_bitset[i] & ~current_union_bitset).
    
    # Precompute ball bitsets as arrays of uint64
    ball_bitsets = np.zeros((N, num_chunks), dtype=np.uint64)
    for i in range(N):
        for j in balls[i]:
            c = chunk_indices[j]
            m = bit_masks[j]
            ball_bitsets[i, c] |= m
            
    # Helper to calculate gains for all candidates given a current union bitset
    def calculate_gains(current_union):
        # current_union is an array of shape (num_chunks,)
        # We need popcount of (ball_bitsets[i] & ~current_union) for each i
        # ~current_union needs to be masked to N bits? No, popcount handles it, 
        # but we must ensure we don't count bits beyond N.
        # Since ball_bitsets only have bits set for valid indices, AND with ~union is safe.
        
        # Vectorized bitwise AND
        # ball_bitsets: (N, num_chunks)
        # ~current_union: (num_chunks,) -> broadcast to (1, num_chunks)
        uncovered_bits = ball_bitsets & (~current_union)[np.newaxis, :]
        
        # Popcount
        # numpy doesn't have native popcount for uint64 in all versions, 
        # but we can use a lookup table or simple bit manipulation.
        # Given N=2187 and num_chunks=35, and we do this in a loop over candidates,
        # it's better to use a precomputed popcount table for 16-bit chunks?
        # Or just use int.bit_count() in a loop? 
        # Let's try a vectorized approach using a lookup table for 8-bit bytes.
        # This might be overkill. Let's just use a simple loop over chunks if needed, 
        # but vectorized numpy operations on uint64 are fast.
        
        # Alternative: Since N is small, maybe a simple numpy sum of booleans is faster?
        # Let's stick to the bitset idea but implement popcount efficiently.
        # For uint64, we can split into 4x 16-bit or use a trick.
        # Actually, for this size, a pure Python loop over N candidates calculating gain
        # via precomputed lists might be competitive if we use sets.
        # BUT, the bitset approach allows us to update the union in O(num_chunks) instead of O(ball_size).
        
        # Let's use a lookup table for popcount of uint8 to compute popcount of uint64.
        # This is a standard technique.
        
        # However, to keep it simple and robust within the time limit, 
        # let's use a different strategy for the main loop:
        # We will use the "Fast Greedy" with a boolean array and precomputed ball lists.
        # The bottleneck in the seed was the LP and the slow SA.
        # A pure greedy with randomized restarts is very fast.
        
        pass # Placeholder, we will implement the loop below
    
    best_code = None
    best_len = N
    
    # Strategy: Multi-Start Randomized Greedy Set Cover
    # 1. Start with empty set.
    # 2. At each step, pick the word that covers the most uncovered points.
    #    To avoid deterministic bias, we can add small random noise to gains or randomize tie-breaking.
    # 3. Once a cover is found, perform local search (removal/swap) to minimize size.
    
    num_restarts = 5000 # We will run as many as fit in time
    
    for r in range(num_restarts):
        if time.time() - t_start > deadline - 5:
            break
            
        # Randomize the order or weights? 
        # Simple randomization: shuffle the candidates at each step? No, that's O(N^2 log N).
        # Better: Add small random noise to the gain to break ties and explore different basins.
        
        covered = np.zeros(N, dtype=bool)
        code = []
        
        # Precompute initial gains? No, gains change as covered changes.
        # We need to calculate gains for all N items at each step.
        # N=2187, steps ~ 200. 2187 * 200 * (cost of gain calc).
        # If gain calc is O(ball_size) = 15, then 2187 * 200 * 15 = 6.5M ops per restart.
        # 5000 restarts * 6.5M = 32.5B ops. Too slow in Python.
        
        # We MUST use the bitset/vectorized approach to make gain calculation O(1) or O(num_chunks).
        # Let's implement the bitset gain calculation properly.
        
        # Popcount implementation for uint64 array
        # We will use a lookup table for 16-bit values.
        # Lookup table size: 65536.
        # This is fast.
        
        # Actually, let's try a simpler optimization first:
        # Use numpy to compute gains.
        # We have ball_bitsets (N, num_chunks).
        # We have current_union (num_chunks,).
        # Gains[i] = sum(popcount(ball_bitsets[i, k] & ~current_union[k]) for k in range(num_chunks))
        
        # Let's create a popcount table for uint8 to compute uint64 popcount vectorized?
        # Or just use a loop over chunks (35 iterations) inside a vectorized numpy op?
        
        # Let's use a precomputed popcount table for 16-bit integers.
        popcount_table = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int32)
        
        def vectorized_popcount(arr):
            # arr is uint64 array
            # Split into high and low 32 bits? Or 4x 16 bits.
            # Let's do 4x 16 bits for simplicity and speed.
            # Shift and mask.
            # This is tricky to do purely vectorized without loops over 4 parts.
            # Given num_chunks=35, a loop of 35 is fine.
            total = np.zeros(len(arr), dtype=np.int32)
            for shift in [0, 16, 32, 48]:
                masked = (arr >> shift) & 0xFFFF
                # Look up in table
                # masked is uint64, need to cast to int32 for indexing
                total += popcount_table[masked.astype(np.int32)]
            return total
        
        # Re-evaluate speed:
        # In each step of greedy:
        # 1. Compute ~current_union (35 ops)
        # 2. AND with ball_bitsets (N * 35 ops)
        # 3. Popcount (N * 35 * 4 lookups)
        # This is roughly 2187 * 35 * 4 = 300k lookups per step.
        # 200 steps = 60M lookups per restart.
        # 5000 restarts = 300B lookups. Still too slow in Python/NumPy overhead.
        
        # We need a faster gain calculation.
        # Alternative: Maintain a "gain" array explicitly and update it incrementally?
        # When we add a codeword, we update 'covered'.
        # The gain of *other* codewords decreases if they overlap with the new one.
        # Updating all gains is O(N * ball_size).
        
        # Let's go back to the "Inspiration 1" finding: 
        # "Randomized greedy set-cover with bitset optimization allows thousands of restarts... Measured: 1 -> 1.034"
        # This implies the bitset approach IS fast enough if implemented correctly.
        # Maybe the popcount can be done faster or the number of restarts is lower but effective.
        
        # Let's try a different bitset trick:
        # Instead of calculating gain for ALL N, we can maintain a list of "active" candidates?
        # No, in set cover, any candidate can be optimal.
        
        # Let's optimize the Popcount.
        # NumPy doesn't have native popcount.
        # But we can use a simple trick:
        # Since N is small, maybe we can just use Python sets for the "covered" set?
        # No, sets are slow for intersection size.
        
        # Let's try the most direct fast method:
        # Use `numpy` to represent the coverage as a single large integer? No.
        
        # Let's assume the "Inspiration" used a very tight loop or C-extension? No, standard lib.
        # Maybe they used `int.bit_count()` in a tight Python loop over N?
        # 2187 * 200 = 437k iterations of bit_count.
        # 437k * 5000 = 2.1B bit_counts.
        # Python bit_count is C-speed. 2.1B operations is ~10-20 seconds?
        # If we do 1000 restarts, it's 2-4 seconds. This is feasible!
        
        # Strategy:
        # 1. Represent each ball as a Python integer (bitmask).
        #    N=2187, so we need a 2187-bit integer. Python handles big ints natively.
        # 2. Maintain `current_union` as a big int.
        # 3. Gain(i) = (ball_mask[i] & ~current_union).bit_count()
        # 4. Greedy step: find max gain.
        
        # Precompute ball masks as Python ints
        ball_masks = [0] * N
        for i in range(N):
            mask = 0
            for j in balls[i]:
                mask |= (1 << j)
            ball_masks[i] = mask
            
        # Run Greedy
        current_union = 0
        code = []
        # We need to find max gain among N items.
        # To speed this up, we can maintain a list of indices and sort? No, gains change.
        # We just have to loop through N and calculate gain.
        
        while True:
            if time.time() - t_start > deadline - 5:
                break
                
            max_gain = -1
            best_idx = -1
            # Randomize tie-breaking by adding small noise?
            # Or just pick first max?
            # To explore different basins, let's randomize the order of indices we check?
            # No, that doesn't change the max.
            # Let's add random noise to the gain: gain + random.uniform(0, 0.1)
            # This effectively randomizes ties and near-ties.
            
            for i in range(N):
                # If already fully covered by this ball, gain is 0
                # Quick check: if ball_masks[i] & ~current_union == 0: continue
                # But bit_count is fast.
                gain = (ball_masks[i] & ~current_union).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
                elif gain == max_gain and max_gain > 0:
                    # Random tie break
                    if random.random() < 0.5:
                        best_idx = i
                        
            if max_gain == 0:
                break
                
            code.append(best_idx)
            current_union |= ball_masks[best_idx]
            
            # Check if all covered
            if current_union == (1 << N) - 1:
                break
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
            # Local Search: Try to remove codewords
            # This is crucial for improvement.
            # We have a valid cover. Can we remove any?
            # If we remove i, is the cover still valid?
            # Valid if (current_union_without_i) == (1 << N) - 1
            # current_union_without_i = current_union & ~ball_masks[i]
            # But this only works if each point is covered by exactly one? No.
            # If a point is covered by multiple, removing one doesn't uncover it.
            # So we need to check if any point becomes uncovered.
            # Uncovered points = ~ (current_union & ~ball_masks[i])
            # This is equivalent to: (ball_masks[i] & ~ (current_union & ~ball_masks[i]))?
            # Actually, simpler:
            # A point p is uncovered after removing i if:
            # p is in ball_masks[i] AND p is NOT in any other selected ball.
            # This is hard to check with just big ints without counting.
            
            # Alternative Local Search:
            # Maintain a count array for coverage.
            # When we do greedy, we don't have counts.
            # Let's do a "Repair" step after greedy:
            # 1. Convert code to list.
            # 2. Compute coverage counts.
            # 3. Try to remove any codeword where min coverage in its ball > 1.
            
            coverage_counts = np.zeros(N, dtype=np.int32)
            for idx in code:
                coverage_counts[balls[idx]] += 1
                
            changed = True
            while changed and time.time() - t_start < deadline - 5:
                changed = False
                for i in range(len(code) - 1, -1, -1):
                    idx = code[i]
                    # Check if any point in this ball has count == 1
                    # If min(coverage_counts[balls[idx]]) > 1, we can remove.
                    if np.min(coverage_counts[balls[idx]]) > 1:
                        coverage_counts[balls[idx]] -= 1
                        code.pop(i)
                        changed = True
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code.copy()
                            # Update current_union? We don't use it in local search.
                            
        # If we have time, do swaps?
        # Swap: Remove i, Add j.
        # This is complex with the big int method.
        # Let's stick to Greedy + Removal for now, as it's robust.

    # Ensure best_code is not None
    if best_code is None:
        # Fallback: return a trivial valid code (all words)
        best_code = list(range(N))
        
    # Convert indices back to list of lists of integers
    result = []
    for idx in best_code:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result

