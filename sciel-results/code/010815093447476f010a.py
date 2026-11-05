import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # The evaluator kills at 150s. We keep a strict margin to ensure return.
    deadline = start_time + 145.0
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighborhoods) as bitmasks
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Helper to get the cover mask
    def get_cover_mask(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask

    # --- Algebraic Construction Strategy ---
    # For q=3, n=8, radius=1, the optimal code size is 486.
    # The structure is related to the ternary Hamming code H(3,4) which has size 3^4=81 words and covers 3^8=6561 words? 
    # Wait, H(3,4) is a [8,4,4] code. Its covering radius is 1?
    # Size of H(3,4) is 3^4 = 81.
    # Ball size for r=1 in 3-ary length 8 is 1 + 8*2 = 17.
    # 81 * 17 = 1377. This is much less than 6561. So H(3,4) does NOT cover the space with radius 1.
    # The covering radius of H(3,4) is actually higher.
    
    # Let's look at the known result: K_3(8,1) = 486.
    # 486 = 2 * 3^5 = 2 * 243.
    # This suggests a construction based on a code of length 5 or similar.
    
    # Alternative approach: Use a strong Greedy initialization but optimized.
    # The "Inspiration" mentions algebraic construction. 
    # A common construction for K_3(n,1) uses the fact that the ternary Hamming code H(3, m) has covering radius 1 if and only if...?
    # Actually, the ternary Hamming code H(3, 4) has covering radius 2? No.
    # Let's stick to a very fast Greedy Set Cover, which is effectively what the "Vectorized greedy initialization" in the cell findings refers to as a speedup.
    
    # However, to be "MATERIALLY DIFFERENT", we should try to exploit the structure if possible.
    # If we can't easily code the specific algebraic construction for 486, we can use a 
    # "Recursive Construction" or "Direct Sum" approach if applicable, or just a 
    # much faster greedy + aggressive local search.
    
    # Let's implement a fast Greedy Set Cover using NumPy for vectorization of the gain calculation.
    # This addresses the "Vectorized greedy initialization" finding which improved score.
    
    def greedy_construct_vectorized():
        uncovered = all_mask
        code = []
        
        # Convert balls to a numpy array of uint64? 
        # 6561 bits is ~103 uint64s. 
        # Representing 6561 balls as a (6561, 103) uint64 array is 6561*103*8 bytes ~ 5.4 MB. Feasible.
        
        # But Python int bit operations are fast. Let's stick to Python ints but optimize the loop.
        # The bottleneck in greedy is scanning 6561 items to find the max gain.
        
        while uncovered:
            if time.time() > deadline - 20.0:
                return code
                
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Optimization: Iterate through all indices.
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
                    
            if max_cover == 0:
                break
                
            # Random tie-breaking
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # --- Advanced Local Search: Kick and Prune ---
    # This method is known to be very effective for covering codes.
    # 1. Start with a valid cover.
    # 2. Prune: Remove any redundant words.
    # 3. Kick: Add a random word, then prune again. This allows the code to "move" in the space.
    # 4. Repeat.
    
    def prune_code(code):
        # Remove redundant words
        changed = True
        while changed:
            changed = False
            # Compute cover mask
            cover = 0
            for c in code:
                cover |= balls[c]
            
            for i in range(len(code)):
                c = code[i]
                # If removing c keeps cover valid
                if (cover & ~balls[c]) == all_mask:
                    code.pop(i)
                    changed = True
                    break # Restart loop to be safe, or continue? 
                    # Continuing might be faster if we update cover, but restarting is safer.
                    # Given small size, restart is fine.
        return code

    def kick_and_prune(initial_code, time_limit):
        code = initial_code[:]
        code = prune_code(code)
        best_code = code[:]
        best_size = len(code)
        
        # Precompute a list of all indices
        all_indices = list(range(total_words))
        
        while time.time() < time_limit:
            # 1. Kick: Add a random word
            # To make it more effective, we can add a word that covers the most "hard" words, 
            # but random is often sufficient for exploration.
            new_word = random.choice(all_indices)
            if new_word not in code:
                code.append(new_word)
                
            # 2. Prune
            code = prune_code(code)
            
            # 3. Check if improved
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                
            # Optional: If stuck, do a more aggressive kick (add multiple words?)
            # Or swap?
            # For now, single kick + prune is a solid baseline for this method.
            
            # To escape deep local minima, we can occasionally do a "double kick"
            if random.random() < 0.1:
                # Add 2 random words
                w1 = random.choice(all_indices)
                w2 = random.choice(all_indices)
                if w1 not in code: code.append(w1)
                if w2 not in code: code.append(w2)
                code = prune_code(code)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]

        return best_code

    # Run Greedy to get initial solution
    initial_code = greedy_construct_vectorized()
    
    # If greedy didn't finish or is bad, fallback to random valid cover?
    # Greedy should always finish for this size.
    
    # Run Kick and Prune with the remaining time
    # Reserve 5 seconds for final processing
    sa_deadline = time.time() + 130.0
    
    best_code = kick_and_prune(initial_code, sa_deadline)
    
    # Final cleanup: Remove any redundant words from the best code found
    if best_code:
        best_code = prune_code(best_code)
        
    # Fallback if something went wrong
    if not best_code:
        best_code = initial_code
        best_code = prune_code(best_code)

    # Convert indices back to words
    return [list(digits_of[i]) for i in best_code]
