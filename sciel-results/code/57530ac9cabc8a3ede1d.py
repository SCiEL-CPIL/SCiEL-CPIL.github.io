import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate the seed code: direct sum of ternary tetracode and all words of length n-4
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all tails of length n-4
    tails = list(itertools.product(range(q), repeat=n-4))
    
    # Construct the full code
    code = [h + list(tail) for h in tetracode for tail in tails]
    
    # Convert to numpy array for efficient operations
    code_arr = np.array(code, dtype=np.int8)
    
    # Generate all words in the space {0..q-1}^n
    # For q=3, n=6, this is 729 words.
    # We can generate them as a numpy array of shape (q**n, n)
    space_size = q ** n
    if space_size > 100000:
        # Fallback for larger spaces, though task specifies n=6, q=3
        pass 
        
    # Efficiently generate all words
    # Using np.indices or similar might be slow for large n, but n=6 is small.
    # Let's use a simple iterative method or np.array(itertools.product(...))
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute coverage: for each codeword, which space words does it cover?
    # Distance <= 1 means Hamming distance 0 or 1.
    # Instead of checking all pairs, we can generate the ball of radius 1 for each codeword.
    # Ball size: 1 + n*(q-1) = 1 + 6*2 = 13.
    
    # Map word tuple to index in all_words
    # Since q=3, n=6, we can use base-q encoding
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
        
    # Create a lookup table for word to index? 
    # Or just use the fact that all_words are sorted in lexicographic order?
    # itertools.product(range(q), repeat=n) generates in lexicographic order.
    # So word_to_idx can be computed directly from the word values.
    
    # Initialize coverage counts for each space word
    # cover_count[i] = number of codewords covering space word i
    cover_count = np.zeros(space_size, dtype=np.int32)
    
    # For each codeword, mark its ball
    # We need to be able to unmark when removing a codeword.
    # So we store for each codeword the list of space word indices it covers.
    codeword_balls = []
    
    for cw in code:
        ball_indices = []
        # Center
        idx_center = word_to_idx(cw)
        ball_indices.append(idx_center)
        
        # Radius 1 neighbors
        for pos in range(n):
            for val in range(q):
                if val == cw[pos]:
                    continue
                new_word = list(cw)
                new_word[pos] = val
                idx = word_to_idx(new_word)
                ball_indices.append(idx)
                
        codeword_balls.append(ball_indices)
        # Update cover counts
        cover_count[ball_indices] += 1
        
    # Greedy removal
    # We maintain a set of active codewords
    active = [True] * len(code)
    
    # To speed up, we can prioritize codewords with smaller balls? All balls are same size.
    # We can try to remove codewords that are "least unique".
    
    # Simple loop: check each active codeword. If it has no exclusive coverage, remove it.
    # Repeat until no more can be removed.
    
    removed_any = True
    while removed_any:
        if time.time() > deadline:
            break
        removed_any = False
        for i in range(len(code)):
            if not active[i]:
                continue
            
            # Check if any word in this codeword's ball is covered ONLY by this codeword
            # i.e., cover_count[idx] == 1
            has_exclusive = False
            for idx in codeword_balls[i]:
                if cover_count[idx] == 1:
                    has_exclusive = True
                    break
            
            if not has_exclusive:
                # Remove codeword i
                active[i] = False
                # Update cover counts
                for idx in codeword_balls[i]:
                    cover_count[idx] -= 1
                removed_any = True
                # We can break and restart the loop to check from the beginning,
                # as removing one might make another removable.
                # Or continue checking. Continuing is faster.
                
    # Construct the final list
    final_code = [code[i] for i in range(len(code)) if active[i]]
    
    # Verification (optional but good for debugging, skip for speed in production if confident)
    # The problem guarantees we just need to return the list.
    
    return final_code
