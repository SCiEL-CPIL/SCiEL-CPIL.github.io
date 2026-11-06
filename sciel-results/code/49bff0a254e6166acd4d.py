import random
import time
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    # Helper to compute Hamming distance
    def hamming_dist(a, b):
        return bin(a ^ b).count('1')
    
    # Precompute bit counts for speed if needed, but N=17 is small.
    # We will maintain a list of codewords.
    codewords = []
    
    # 1. Greedy Phase: Add random words that satisfy distance constraint
    # To make it faster, we can try to build from a known good start or just random.
    # Random start is often sufficient for greedy if we try many candidates.
    
    # Let's try to include the all-zero word first? 
    # It's a valid codeword.
    codewords.append(0)
    
    # Greedy addition
    attempts = 0
    max_attempts_greedy = 50000
    while len(codewords) < 200 and attempts < max_attempts_greedy:
        if time.time() - start_time > time_limit * 0.5:
            break
            
        w = random.getrandbits(N)
        valid = True
        for c in codewords:
            if hamming_dist(w, c) < MIN_DIST:
                valid = False
                break
        if valid:
            codewords.append(w)
        attempts += 1
        
    # 2. Local Search Phase: Try to improve by replacing codewords or adding more
    # A common strategy: 
    # - Try to add more randoms.
    # - Try to "move" a codeword to a better position to allow others in? 
    # - Or simply run a hill-climbing on the set size.
    
    # Let's implement a simple "swap" or "add" loop.
    # If we can't add a new one, maybe we can replace an existing one with a new one that allows more additions later?
    # That's complex. A simpler heuristic: 
    # Randomly pick a codeword, remove it, try to add 2 new ones in its place? No, that reduces size.
    # Better: Just keep trying to add. If stuck, do random restarts?
    # Given the time limit, a single good greedy run + local search on the set is best.
    
    # Local Search: Try to perturb existing codewords to see if we can pack more?
    # Actually, for packing problems, "randomized greedy" with many restarts is often effective.
    # But we have a time limit.
    
    # Let's try a specific local search: 
    # 1. Get current size K.
    # 2. Try to add a new word. If fail, try to modify an existing word slightly to see if it opens up space?
    # This is risky.
    
    # Alternative: Use the known structure? 
    # The hint suggests greedy random + local search swaps.
    # "Swaps" might mean: Pick a codeword c_i, remove it. Try to add two new codewords c_a, c_b such that they are valid with the rest. If successful, size increases by 1.
    
    best_codewords = codewords[:]
    
    # Local search loop
    while time.time() - start_time < time_limit:
        improved = False
        # Try to find a pair of new words that can replace one existing word
        # This is expensive O(K^2 * N). 
        # Simpler: Just keep trying to add random words.
        # If we are stuck at size K, it might be a local optimum.
        # Let's try "kick" strategies: randomly flip bits in a few codewords to break symmetry?
        
        # A very effective simple strategy for small N:
        # 1. Randomly generate a set of M candidate words (M > current size).
        # 2. Greedily select the largest subset with min distance.
        # But we already did greedy.
        
        # Let's try a simple hill climb:
        # Pick a random codeword c.
        # Generate a random new word w.
        # If w is valid with all OTHERS, replace c with w? 
        # This doesn't increase size.
        
        # To increase size:
        # We need to find a "hole".
        # Let's just run more greedy attempts with different random seeds/starts?
        # No, we are in one function call.
        
        # Let's try this: 
        # If len(codewords) < 50, do more greedy.
        # If len(codewords) >= 50, try to optimize.
        
        # Actually, the hint says "local search swaps".
        # Swap: Remove c_i. Try to add c_new1 and c_new2.
        # If both fit with (codewords \ {c_i}), then replace c_i with c_new1, c_new2.
        
        # To make this fast, we precompute distances? 
        # N=17, K~50-100. K^2 is 10k-10k. Hamming dist is fast.
        
        # Let's implement the "replace 1 with 2" move.
        
        K = len(codewords)
        if K == 0:
            continue
            
        # Pick a random index to remove
        idx = random.randint(0, K - 1)
        c_remove = codewords[idx]
        others = [c for i, c in enumerate(codewords) if i != idx]
        
        # Try to find two new words
        # We can sample random words and check against 'others'
        found_new = []
        # We need to check against 'others' (size K-1)
        # And the two new words must be distance >= 6 from each other.
        
        # Optimization: Just try to add one word at a time to a temporary set?
        # Let's try to add 2 random words to the set `others`.
        # If we find w1 valid with others, and w2 valid with others and w1, we have a candidate.
        
        # This inner loop might be slow if we do it many times.
        # Let's limit attempts per swap.
        
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
                
            w1 = random.getrandbits(N)
            # Check w1 against others
            ok1 = True
            for c in others:
                if hamming_dist(w1, c) < MIN_DIST:
                    ok1 = False
                    break
            if not ok1:
                continue
                
            # Try to find w2
            for _ in range(100):
                w2 = random.getrandbits(N)
                # Check w2 against others
                ok2 = True
                for c in others:
                    if hamming_dist(w2, c) < MIN_DIST:
                        ok2 = False
                        break
                if not ok2:
                    continue
                
                # Check w2 against w1
                if hamming_dist(w2, w1) < MIN_DIST:
                    continue
                
                # Found w1, w2
                # New set: others + [w1, w2]
                # Size increases by 1
                new_set = others + [w1, w2]
                codewords = new_set
                improved = True
                break
            if improved:
                break
        
        if not improved:
            # If we can't improve with swaps, maybe try just adding?
            # Or restart? 
            # Let's just break if we've tried many times without improvement?
            # For now, keep looping until time runs out, as random restarts might help.
            # But we are modifying the current list.
            # If we are stuck, we might be in a local optimum.
            # Let's try a "kick": randomly flip 1-3 bits in a random codeword.
            # This might break the local optimum.
            if random.random() < 0.1:
                idx_kick = random.randint(0, len(codewords) - 1)
                c_kick = codewords[idx_kick]
                # Flip 1-3 random bits
                bits_to_flip = random.sample(range(N), random.randint(1, 3))
                for b in bits_to_flip:
                    c_kick ^= (1 << b)
                codewords[idx_kick] = c_kick
                # Note: This might make the set invalid? 
                # No, the codeword is still a valid integer. 
                # But is the *set* still valid? 
                # The distances change! 
                # If we change c_kick, its distance to others changes.
                # It might now be < 6.
                # So we must verify the whole set after a kick?
                # That's O(K^2).
                # Let's verify.
                valid_set = True
                for i in range(len(codewords)):
                    for j in range(i+1, len(codewords)):
                        if hamming_dist(codewords[i], codewords[j]) < MIN_DIST:
                            valid_set = False
                            break
                    if not valid_set:
                        break
                if not valid_set:
                    # Revert kick
                    for b in bits_to_flip:
                        codewords[idx_kick] ^= (1 << b)
                    # Or just discard the kick and try again?
                    # Let's revert.
                    pass

    # Ensure distinctness (safety check)
    # The logic above should maintain distinctness, but let's be sure.
    # If duplicates exist, remove them.
    unique_codewords = []
    seen = set()
    for c in codewords:
        if c not in seen:
            seen.add(c)
            unique_codewords.append(c)
            
    return unique_codewords
