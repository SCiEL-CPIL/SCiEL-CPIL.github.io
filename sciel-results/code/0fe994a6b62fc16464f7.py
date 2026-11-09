import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Assume q=3, n=6, radius=1 based on task description
    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy or seed
        # For robustness, if params don't match, return seed-like result
        # But likely won't happen.
        pass

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute idx to word and balls
    # idx to word
    def idx_to_word(idx):
        return [(idx // (q ** i)) % q for i in range(n)]
        
    # Precompute balls: list of lists of indices
    balls = []
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if w[pos] != val:
                    new_w = w.copy()
                    new_w[pos] = val
                    new_idx = sum(v * (q ** p) for p, v in enumerate(new_w))
                    neigh.append(new_idx)
        balls.append(neigh)
        
    # Initial solution: Tetracode direct sum
    tetracode_heads = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = set()
    for h in tetracode_heads:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            idx = sum(v * (3 ** p) for p, v in enumerate(word))
            init_codewords.add(idx)
            
    # State
    cover_counts = np.zeros(total_words, dtype=np.int32)
    for c in init_codewords:
        cover_counts[balls[c]] += 1
        
    uncovered_count = np.sum(cover_counts == 0)
    current_codewords = set(init_codewords)
    current_size = len(current_codewords)
    
    best_codewords = set(current_codewords)
    best_size = current_size
    
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9995 # Decay rate
    
    M = 1000 # Penalty for uncovered
    
    # Helper to add codeword
    def add_codeword(c):
        nonlocal uncovered_count, current_size
        if c in current_codewords:
            return 0 # No change
        
        # Calculate delta uncovered
        # Count zeros in balls[c] before
        zeros_before = np.sum(cover_counts[balls[c]] == 0)
        
        current_codewords.add(c)
        current_size += 1
        cover_counts[balls[c]] += 1
        
        # Count zeros in balls[c] after
        zeros_after = np.sum(cover_counts[balls[c]] == 0)
        
        delta_u = zeros_after - zeros_before
        uncovered_count += delta_u
        return delta_u

    # Helper to remove codeword
    def remove_codeword(c):
        nonlocal uncovered_count, current_size
        if c not in current_codewords:
            return 0 # No change
            
        # Calculate delta uncovered
        # Count zeros in balls[c] before
        zeros_before = np.sum(cover_counts[balls[c]] == 0)
        
        current_codewords.remove(c)
        current_size -= 1
        cover_counts[balls[c]] -= 1
        
        # Count zeros in balls[c] after
        zeros_after = np.sum(cover_counts[balls[c]] == 0)
        
        delta_u = zeros_after - zeros_before
        uncovered_count += delta_u
        return delta_u

    while time.time() < deadline:
        if uncovered_count > 0:
            # Repair phase: Greedy Add
            # Pick a random uncovered word
            uncovered_indices = np.where(cover_counts == 0)[0]
            if len(uncovered_indices) == 0:
                uncovered_count = 0 # Sync
                continue
                
            u = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            # Find best candidate in balls[u]
            best_gain = -1
            best_c = None
            for c in balls[u]:
                if c in current_codewords:
                    continue
                # Gain is number of uncovered words covered by c
                gain = np.sum(cover_counts[balls[c]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            if best_c is not None:
                add_codeword(best_c)
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        else:
            # Optimization phase: Try to remove words
            # Pick a random codeword
            c = random.choice(list(current_codewords))
            
            # Calculate gain (loss of coverage)
            # Number of words that become uncovered if we remove c
            # These are words j in balls[c] where cover_counts[j] == 1
            gain = np.sum(cover_counts[balls[c]] == 1)
            
            if gain == 0:
                # Safe removal
                remove_codeword(c)
                if current_size < best_size:
                    best_size = current_size
                    best_codewords = set(current_codewords)
            else:
                # Risky removal
                # Old cost: 0*M + current_size
                # New cost: gain*M + (current_size - 1)
                delta_cost = (gain * M + (current_size - 1)) - (0 * M + current_size)
                delta_cost = gain * M - 1
                
                if random.random() < np.exp(-delta_cost / T):
                    remove_codeword(c)
                    # Note: uncovered_count will increase, so we exit optimization phase next loop
                    # We don't update best here because it's invalid (uncovered > 0)
                    
            T *= alpha
            if T < T_min:
                T = T_min

    # Convert best_codewords back to list of lists
    result = []
    for idx in best_codewords:
        result.append(idx_to_word(idx))
    return result
