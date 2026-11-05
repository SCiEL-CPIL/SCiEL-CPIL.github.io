import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate the tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all product words
    # We need to map words to indices 0..3^8-1
    # Index = sum(w[i] * 3^i) or similar base-3 encoding
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute neighbors for all 3^n words
    # This might take a moment, but 6561 * 17 is small
    all_words = [idx_to_word(i) for i in range(3**n)]
    neighbors = np.zeros((3**n, 1 + n * (q - 1)), dtype=np.int32)
    
    for i, w in enumerate(all_words):
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                # Efficient way: calculate index difference
                # idx = sum(w[j] * 3^j). 
                # If we change w[pos] to new_val, the index changes by (new_val - orig_val) * 3^pos
                # Wait, my word_to_idx uses base 3 where w[0] is most significant?
                # idx = w[0]*3^(n-1) + ... + w[n-1]*3^0
                # So change at pos i (0-indexed from left) affects 3^(n-1-i)
                # Let's stick to the idx calculation directly to be safe and fast.
                pass
        
    # Let's optimize neighbor generation
    # Powers of 3
    powers = [3**i for i in range(n)]
    # In my word_to_idx: idx = w[0]*3^(n-1) + ... + w[n-1]*3^0
    # So weight of w[i] is 3^(n-1-i)
    weights = [3**(n - 1 - i) for i in range(n)]
    
    for i, w in enumerate(all_words):
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            weight = weights[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * weight
                neighbors[i, count] = new_idx
                count += 1
                
    # Initial set of codewords
    initial_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_words.append(w)
            
    initial_indices = [word_to_idx(w) for w in initial_words]
    
    # Counts array
    counts = np.zeros(3**n, dtype=np.int32)
    for idx in initial_indices:
        counts[neighbors[idx]] += 1
        
    active_indices = list(initial_indices)
    
    best_size = len(active_indices)
    best_set = list(active_indices)
    
    # Iterative removal
    # We will run multiple passes
    # Each pass: shuffle active, try to remove
    
    while time.time() < deadline:
        # Shuffle active indices
        np.random.shuffle(active_indices)
        
        # Iterate through a copy? No, we modify the list.
        # If we remove an item, the list shrinks.
        # We can iterate backwards or use a new list for next round?
        # Simpler: iterate through the current list, and if removable, mark for removal.
        # But removing changes counts, which affects subsequent checks.
        # This is actually good: dynamic removal.
        
        to_remove = []
        for idx in active_indices:
            # Check if all neighbors have count >= 2
            # neighbors[idx] is an array of 17 indices
            if np.min(counts[neighbors[idx]]) >= 2:
                to_remove.append(idx)
                
        if not to_remove:
            break
            
        for idx in to_remove:
            counts[neighbors[idx]] -= 1
            active_indices.remove(idx) # O(N) removal, but N is small
            
        if len(active_indices) < best_size:
            best_size = len(active_indices)
            best_set = list(active_indices)
            
    # Convert back to words
    result = [idx_to_word(idx) for idx in best_set]
    return result
