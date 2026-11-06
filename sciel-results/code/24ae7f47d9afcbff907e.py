import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin

    N = 17
    MIN_DIST = 6

    # 1. Generate Seed Code (32 words)
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)

    best_code = seed_words[:]
    best_size = len(best_code)

    # 2. Prepare Candidates
    # Exclude seed words from candidates to save time
    seed_set = set(seed_words)
    all_candidates = [w for w in range(2**N) if w not in seed_set]

    # 3. Greedy Search Loop
    # We will run multiple randomized greedy passes
    
    # Optimization: Pre-compute weights? No, bit_count is fast.
    
    # To improve chances of finding the 256-word linear code,
    # we can try different ordering strategies.
    
    for trial in range(200):  # Max trials, will break on time
        if time.time() > deadline:
            break

        # Strategy 1: Random Shuffle
        # Strategy 2: Weight Sorted (centered around 8.5)
        # Let's alternate or pick randomly
        
        if trial % 2 == 0:
            random.shuffle(all_candidates)
        else:
            # Sort by weight, then random tie-breaker?
            # Sorting 131k items takes time.
            # Let's just shuffle for simplicity and speed.
            # Or use a fixed "smart" order?
            # Actually, random shuffle is sufficient for finding the basin.
            random.shuffle(all_candidates)

        current_code = seed_words[:]
        code_set = seed_set.copy()

        # Local variable for speed
        min_dist = MIN_DIST
        
        for w in all_candidates:
            # Check compatibility
            # We can break early if incompatible
            compatible = True
            for c in current_code:
                if ((w ^ c).bit_count() < min_dist):
                    compatible = False
                    break
            
            if compatible:
                current_code.append(w)
                code_set.add(w)

        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]

    return best_code
