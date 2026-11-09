import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute masks for distances 1 to 5
    # masks[k] is a list of integers with bit_count == k
    masks = [[] for _ in range(MIN_D)] # 1 to 5
    for i in range(SIZE):
        w = i.bit_count()
        if 1 <= w < MIN_D:
            masks[w].append(i)
            
    # Helper to mark a word and its neighbors as forbidden
    # forbidden is a bytearray of size SIZE
    def mark_forbidden(forbidden, w):
        forbidden[w] = 1
        for k in range(1, MIN_D):
            for m in masks[k]:
                forbidden[w ^ m] = 1

    # Step 1: Construct a good linear code seed
    # We use a fast greedy linear construction
    # This is fast enough to run once or a few times
    def get_linear_seed():
        # Greedy basis construction
        basis = []
        code = [0]
        # Try to find 8 basis vectors
        # Candidates: all non-zero words
        # We can just pick random words or sorted by weight
        # Let's use a deterministic high-weight order for the seed to be stable
        indices = list(range(1, SIZE))
        indices.sort(key=lambda x: -x.bit_count())
        
        for g in indices:
            if len(basis) >= 8:
                break
            # Check if g is independent and maintains distance
            valid = True
            for c in code:
                if (g ^ c).bit_count() < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                new_words = [c ^ g for c in code]
                code.extend(new_words)
        return code

    seed_code = get_linear_seed()
    
    best_size = len(seed_code)
    best_code = seed_code[:]
    
    # Step 2: Greedy Extension with Random Restarts
    # We will run multiple iterations.
    # Each iteration:
    # 1. Start with seed_code (or a subset if we want to try different bases? No, seed is fixed)
    # 2. Initialize forbidden array
    # 3. Mark all seed words and their neighbors as forbidden
    # 4. Create list of available words (not forbidden)
    # 5. Shuffle available words
    # 6. Greedily pick from available, updating forbidden
    
    # Optimization: 
    # Instead of creating a list of available words (which is ~100k items), 
    # we can just iterate through all SIZE words and check forbidden[w].
    # This is O(SIZE) per iteration, which is 131k checks. Very fast.
    
    # To vary the result, we shuffle the order of checking.
    # But shuffling 131k items takes time.
    # Alternative: Use a random permutation generated once? No, we need different orders.
    # We can use `random.sample` or just `random.shuffle` on a range?
    # Actually, we can just pick random indices? No, we need to process them in a specific order to maintain the greedy property (once forbidden, stay forbidden).
    # So we need a permutation of 0..SIZE-1.
    
    # Let's estimate time.
    # Marking neighbors for 1 seed word: ~9400 ops.
    # 256 seed words: ~2.4M ops.
    # This is done once per restart? 
    # If we do 100 restarts, 240M ops. Might be too slow in Python.
    
    # Optimization:
    # We can pre-compute the forbidden state for the seed code once.
    # `seed_forbidden` is a bytearray.
    # Then for each restart, we copy it? `bytearray` copy is fast (128KB).
    # Then we need to add more words.
    # The list of candidates is `all_indices = list(range(SIZE))`.
    # We shuffle `all_indices`? That's expensive.
    
    # Better:
    # We maintain a list `candidates` of words that are NOT in seed_forbidden.
    # This list is created once.
    # For each restart, we shuffle this list? 
    # Size of candidates: SIZE - (forbidden by seed).
    # Forbidden by seed is roughly 256 * 9400 / SIZE * something? 
    # Actually, the union of neighborhoods of 256 words is likely large.
    # Let's just use the full range and check `forbidden[w]`.
    
    # To avoid shuffling 131k items, we can use a "Random Walk" or "Random Start" strategy?
    # No, Greedy MIS needs a total order.
    
    # Let's try:
    # 1. Compute `seed_forbidden`.
    # 2. `candidates = [i for i in range(SIZE) if not seed_forbidden[i]]`
    # 3. In loop:
    #    `current_forbidden = bytearray(seed_forbidden)`
    #    `shuffled_candidates = candidates[:]`
    #    `random.shuffle(shuffled_candidates)`
    #    `code = seed_code[:]`
    #    for w in shuffled_candidates:
    #       if not current_forbidden[w]:
    #          code.append(w)
    #          mark_forbidden(current_forbidden, w)
    #    if len(code) > best_size: ...
    
    # Time check:
    # `candidates` size ~ 100k.
    # Copy + Shuffle 100k items: ~0.01s?
    # Loop 100k items: ~0.05s?
    # Marking neighbors: If we add 100 more words, 100 * 9400 = 1M ops. ~0.05s?
    # Total per restart: ~0.1s - 0.2s.
    # In 85s, we can do 400-800 restarts.
    # This should be sufficient to find a code larger than 256.
    
    # Compute seed_forbidden
    seed_forbidden = bytearray(SIZE)
    for w in seed_code:
        mark_forbidden(seed_forbidden, w)
        
    candidates = [i for i in range(SIZE) if not seed_forbidden[i]]
    
    # Run restarts
    while time.time() < deadline:
        current_forbidden = bytearray(seed_forbidden)
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        current_code = seed_code[:]
        
        for w in shuffled_candidates:
            if not current_forbidden[w]:
                current_code.append(w)
                mark_forbidden(current_forbidden, w)
                
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
    return best_code
