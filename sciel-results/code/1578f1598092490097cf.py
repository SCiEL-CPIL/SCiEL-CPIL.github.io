import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # Parameters for this specific task
    # q=3, n=7, radius=1
    total_words = q ** n
    words_per_ball = 1 + n * (q - 1)
    
    # 1. Precompute all possible balls (sets of words covered by a codeword)
    # We represent each ball as a single uint64 bitmask because total_words < 2**64
    # Actually, 2187 bits fits easily in a few uint64s, but since it's < 64 bits, 
    # we can just use one uint64? No, 2187 > 64. 
    # Wait, 3^7 = 2187. A single uint64 holds 64 bits. 
    # So we need a bitmask of length 2187. 
    # NumPy doesn't have a native arbitrary-precision integer type for bitmasks that is fast for popcount.
    # However, we can use an array of uint64s. 
    # 2187 / 64 = 34.17, so we need 35 uint64s per ball.
    
    num_chunks = (total_words + 63) // 64
    # Balls: shape (total_words, num_chunks), dtype uint64
    # This is 2187 * 35 * 8 bytes ~ 611 KB. Very manageable.
    
    # Generate all words as indices 0..total_words-1
    # We need to map a word (tuple of digits) to an index.
    # Index = d0 * q^(n-1) + d1 * q^(n-2) + ... + d_{n-1}
    
    # Precompute neighbors for each word index
    # neighbor_indices[i] is a list of indices within distance 'radius' of word i
    # For radius=1, neighbors are the word itself and all words at distance 1.
    
    # Efficient way to generate balls:
    # 1. Create an array of all words as tuples? No, too slow.
    # 2. Use base-q expansion.
    
    # Let's generate the ball bitmasks.
    # For each word w, its ball B(w) contains w and all words at dist 1.
    
    # We can precompute the bitmasks for all words.
    # balls[i] is a uint64 array of length num_chunks representing the set of words covered by codeword i.
    
    balls = np.zeros((total_words, num_chunks), dtype=np.uint64)
    
    # Helper to set a bit in the array
    def set_bit(arr, idx):
        chunk_idx = idx >> 6
        bit_idx = idx & 63
        arr[chunk_idx] |= (1 << bit_idx)
        
    # We need to iterate through all words and set bits for their neighbors.
    # This is O(total_words * words_per_ball).
    # 2187 * 15 = 32805 operations. Very fast.
    
    # To do this efficiently in Python without pure loops being too slow, 
    # we can generate the indices for each word's ball.
    
    # Let's precompute the "delta" indices for distance 1.
    # For a word w (represented as digits d0..d6), a neighbor at distance 1 changes one digit.
    # d_i changes to (d_i + 1) % q or (d_i - 1) % q.
    
    # We will compute the list of neighbor indices for each word index.
    # Since total_words is small, we can just do this in a loop.
    
    # Precompute powers of q
    powers = [q ** i for i in range(n)]
    
    # Convert index to digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = (idx // powers[n - 1 - i]) % q
        return digits
        
    # Convert digits to index
    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx += digits[i] * powers[n - 1 - i]
        return idx

    # Precompute balls
    for i in range(total_words):
        digits = idx_to_digits(i)
        # The word itself
        set_bit(balls[i], i)
        
        # Neighbors at distance 1
        for pos in range(n):
            for delta in [1, -1]:
                new_digit = (digits[pos] + delta) % q
                if new_digit != digits[pos]:
                    new_digits = digits.copy()
                    new_digits[pos] = new_digit
                    j = digits_to_idx(new_digits)
                    set_bit(balls[i], j)
                    
    # 2. Greedy Set Cover Algorithm
    # State:
    # - covered: uint64 array of length num_chunks, representing the union of all selected balls.
    # - selected: list of selected word indices.
    # - best_score: number of codewords in the best solution found.
    # - best_solution: the list of codewords.
    
    best_score = total_words # Upper bound
    best_solution = list(range(total_words))
    
    # We will run multiple random restarts.
    # In each restart, we use a greedy approach with random tie-breaking.
    
    max_restarts = 500 # We will check time in the loop
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        covered = np.zeros(num_chunks, dtype=np.uint64)
        selected = []
        
        # To speed up, we can pre-sort or use a priority queue, but given the small size,
        # a simple scan for max gain might be sufficient if vectorized.
        # Gain of selecting ball i = popcount(balls[i] & ~covered)
        
        # We will use a list of available balls. Initially all balls.
        # Instead of removing balls, we can just scan all balls each time?
        # 2187 balls * ~200 selections = 437k scans.
        # Each scan involves bitwise AND and popcount.
        # NumPy bitwise AND is fast. Popcount is not directly vectorized for uint64 arrays in older NumPy, 
        # but we can use `np.bitwise_count` if available (NumPy 2.0+?) or a lookup table.
        # Let's check if we can use a fast popcount.
        # For 35 chunks, we can just sum the popcounts of each chunk.
        # Python's `int.bit_count()` is fast.
        
        # Let's try a simpler approach: 
        # Maintain an array `gains` of length total_words.
        # Initially, gains[i] = popcount(balls[i]).
        # When we select a ball, we update gains for all other balls? That's O(N^2).
        # 2187^2 ~ 4.7M. With 200 iterations, that's 1B ops. Too slow in Python.
        
        # Alternative: Scan all balls each step to find the max gain.
        # Step k:
        #   for i in range(total_words):
        #       gain = popcount(balls[i] & ~covered)
        #   select i with max gain.
        
        # Vectorized gain calculation:
        # ~covered is a uint64 array.
        # balls & ~covered is a 2D array (total_words, num_chunks).
        # We need to sum the popcounts along axis 1.
        
        # Since NumPy doesn't have a direct `popcount` for uint64 arrays that returns an array of ints efficiently 
        # without casting to object or using a lookup, let's use a trick.
        # We can cast to `np.uint64` and use a lookup table for 8-bit chunks?
        # Or simply, since num_chunks is small (35), we can loop over chunks in Python?
        # No, that's slow.
        
        # Let's use `np.bitwise_and` and then count bits.
        # If we have a 1D array of uint64s, we can convert to Python ints and use bit_count?
        # That's slow.
        
        # Better: Use a precomputed lookup for 16-bit chunks?
        # Or, since the number of chunks is small, we can just do:
        # gains = np.zeros(total_words)
        # for c in range(num_chunks):
        #    gains += popcount(balls[:, c] & ~covered[c])
        
        # This requires a fast popcount for a 1D array of uint64s.
        # We can use `np.unpackbits`? No, that's for uint8.
        
        # Let's try a different representation.
        # Since total_words is 2187, we can use a single Python integer for the bitmask!
        # Python integers have arbitrary precision.
        # Bitwise AND and bit_count() are implemented in C and very fast.
        
        # Let's restart the precomputation using Python integers.
        
        # Re-do precomputation with Python ints
        balls_int = [0] * total_words
        for i in range(total_words):
            digits = idx_to_digits(i)
            mask = 1 << i
            balls_int[i] |= mask
            for pos in range(n):
                for delta in [1, -1]:
                    new_digit = (digits[pos] + delta) % q
                    if new_digit != digits[pos]:
                        new_digits = digits.copy()
                        new_digits[pos] = new_digit
                        j = digits_to_idx(new_digits)
                        balls_int[i] |= (1 << j)
                        
        # Now the greedy loop
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            covered = 0
            selected = []
            
            # We need to find the ball with max new coverage.
            # To avoid O(N) scan with Python int ops being slow, we can just do it.
            # 2187 balls.
            # ~200 selections.
            # 437k iterations of (AND + bit_count).
            # Python int bit_count is very fast (C implementation).
            
            # Add randomization: add a small random noise to the gain?
            # Or just randomize the order of balls with equal gain?
            # Simpler: randomize the starting order or add tiny noise.
            
            # Let's just scan all balls.
            
            for step in range(total_words): # Max steps
                if time.time() > deadline:
                    break
                    
                max_gain = -1
                best_ball_idx = -1
                
                # To introduce diversity, we can shuffle the balls or add noise.
                # Let's add a small random noise to the gain to break ties randomly.
                # noise = np.random.rand(total_words) * 0.1 ?
                # But we are using Python ints.
                
                # We will just pick the first max gain, but to diversify, 
                # we can randomize the scan order? No, that's expensive to shuffle 2187 items 200 times.
                
                # Instead, we can add a small random float to the gain?
                # Gain is an integer. 
                # Let's just use a random tie-breaker.
                
                # Optimization: We can keep a list of "active" balls? 
                # But checking if a ball is fully covered is also costly.
                
                # Let's just scan.
                for i in range(total_words):
                    new_cover = balls_int[i] & ~covered
                    gain = new_cover.bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_ball_idx = i
                    elif gain == max_gain and gain > 0:
                        # Random tie break
                        if np.random.rand() < 0.5:
                            best_ball_idx = i
                            
                if best_ball_idx == -1 or max_gain == 0:
                    break
                    
                covered |= balls_int[best_ball_idx]
                selected.append(best_ball_idx)
                
                # Check if fully covered
                if covered == (1 << total_words) - 1:
                    break
                    
            # End greedy step
            
            if len(selected) < best_score:
                best_score = len(selected)
                best_solution = selected.copy()
                
    # 3. Local Search / Polish
    # Try to remove redundant codewords.
    # If a codeword's ball is subset of the union of others, remove it.
    
    if best_score < total_words:
        # Sort by "importance" (gain when removed?) or just iterate
        # We will try to remove each codeword if possible.
        
        # To make it faster, we can do this in a loop until no more removals are possible.
        # Or just one pass.
        
        current_solution = best_solution.copy()
        changed = True
        while changed:
            changed = False
            for i in range(len(current_solution)):
                if time.time() > deadline:
                    break
                # Compute union of all others
                # This is O(N^2) if done naively.
                # Union of others = (Union of all) ^ (Ball i) ? No, because Ball i might be partially covered by others.
                # Union of others = OR of balls[j] for j != i.
                
                # We can precompute the total union?
                # Total union is just `covered` from the greedy phase? No, `covered` was the final state.
                # Let's recompute the union of the current solution.
                
                union_all = 0
                for idx in current_solution:
                    union_all |= balls_int[idx]
                    
                # Now for each i, check if balls_int[i] is subset of (union_all ^ balls_int[i]?) 
                # No. 
                # We need to check if balls_int[i] is covered by the union of the OTHERS.
                # Let U = union of all.
                # Let O_i = union of all except i.
                # We can't easily get O_i from U and B_i.
                
                # However, if B_i is a subset of U, it might still be "essential" if some word in B_i is ONLY covered by B_i.
                # So we need to check if there is any word w in B_i such that w is NOT in (U \ B_i).
                # i.e., w is in B_i AND w is NOT in (U & ~B_i).
                # This is equivalent to: B_i & ~(U & ~B_i) != 0?
                # No. 
                # Let's define:
                # Covered_by_others = U & ~B_i
                # If B_i is subset of Covered_by_others, then B_i is redundant.
                # i.e., B_i & ~Covered_by_others == 0
                # i.e., B_i & ~(U & ~B_i) == 0
                # i.e., B_i & (~U | B_i) == 0
                # Since B_i is always in B_i, this is B_i & ~U == 0.
                # Wait. If B_i has a bit that is NOT in U, that's impossible since B_i is part of U.
                # So B_i is always a subset of U.
                
                # The condition for redundancy is:
                # Every word in B_i is covered by at least one other codeword.
                # Let O_i be the union of balls of all codewords except i.
                # Redundant if B_i is a subset of O_i.
                
                # How to compute O_i efficiently?
                # We can't easily subtract B_i from U if other balls also cover parts of B_i.
                
                # So we must compute O_i explicitly or check word by word.
                # Given the small size, we can just compute O_i by ORing all others.
                # This is O(N^2) per removal check.
                # N is ~200. N^2 = 40,000. 
                # 200 iterations * 40,000 = 8M ops. Fast enough.
                
                o_i = 0
                for j in range(len(current_solution)):
                    if j != i:
                        o_i |= balls_int[current_solution[j]]
                        
                if (balls_int[current_solution[i]] & ~o_i) == 0:
                    # Redundant
                    current_solution.pop(i)
                    changed = True
                    break # Restart the while loop since indices changed
                    
        if len(current_solution) < best_score:
            best_score = len(current_solution)
            best_solution = current_solution

    # Convert indices back to codewords (lists of integers)
    final_codewords = []
    for idx in best_solution:
        digits = idx_to_digits(idx)
        final_codewords.append(digits)
        
    return final_codewords

