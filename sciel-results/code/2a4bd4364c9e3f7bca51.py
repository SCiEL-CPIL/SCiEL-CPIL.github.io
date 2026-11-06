import time
import random
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    def hamming_dist(a, b):
        return bin(a ^ b).count('1')

    def is_compatible_with_all(w, word_list):
        for c in word_list:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True

    rng = random.Random(42)
    
    # Generate a large pool of random candidates
    # We need a good mix of weights.
    # Random vectors have weight ~8.5.
    # We want to pack as many as possible.
    
    def greedy_pack(candidates, existing):
        """Add compatible candidates to existing list."""
        new_words = []
        for w in candidates:
            if w in existing:
                continue
            if is_compatible_with_all(w, existing):
                new_words.append(w)
                existing.append(w)
        return new_words

    def local_search(current, time_deadline):
        """
        Perform local search on the current set.
        Strategy: 1-out 2-in swap.
        """
        current = list(current)
        best_size = len(current)
        
        # Pre-generate a pool of candidates for swaps
        # We can generate new ones on the fly or use a static pool.
        # On the fly is better for diversity.
        
        while time.time() < time_deadline:
            if len(current) < 2:
                break
            
            # Pick a random codeword to remove
            idx = rng.randint(0, len(current) - 1)
            c_out = current[idx]
            
            # Temporarily remove it
            current[idx] = current[-1]
            current.pop()
            
            # Try to add two new codewords
            added = 0
            for _ in range(20): # Try a few times to find compatible pairs
                if time.time() > time_deadline:
                    break
                
                # Generate two random candidates
                c1 = rng.getrandbits(N)
                c2 = rng.getrandbits(N)
                
                # Check if c1 is compatible with current
                if c1 in current or not is_compatible_with_all(c1, current):
                    continue
                
                # Check if c2 is compatible with current and c1
                if c2 in current or c2 == c1 or not is_compatible_with_all(c2, current):
                    continue
                if hamming_dist(c1, c2) < MIN_DIST:
                    continue
                
                # Both are compatible
                current.append(c1)
                current.append(c2)
                added = 2
                break
            
            if added == 2:
                # We removed 1, added 2. Net +1.
                if len(current) > best_size:
                    best_size = len(current)
            else:
                # Try to add just 1
                found = False
                for _ in range(10):
                    if time.time() > time_deadline:
                        break
                    c1 = rng.getrandbits(N)
                    if c1 in current or not is_compatible_with_all(c1, current):
                        continue
                    current.append(c1)
                    found = True
                    break
                
                if not found:
                    # Put c_out back
                    current.append(c_out)

    best_codewords = []
    
    # Phase 1: Multiple Greedy Restarts
    # We run greedy several times to find a good starting point.
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > time_limit * 0.2:
            break
        
        current = []
        # Generate a batch of candidates
        # Use a mix of random and structured
        candidates = [rng.getrandbits(N) for _ in range(20000)]
        
        # Shuffle to avoid bias
        rng.shuffle(candidates)
        
        greedy_pack(candidates, current)
        
        if len(current) > len(best_codewords):
            best_codewords = list(current)
            
    # Phase 2: Local Search on the best found set
    if best_codewords:
        # Allocate remaining time to local search
        local_search(best_codewords, time_limit)
        
    # Final verification (optional, but good for safety)
    # The local search maintains validity by construction.
    
    return best_codewords
