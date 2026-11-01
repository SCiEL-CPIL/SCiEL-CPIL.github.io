import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all words in the space
    # We can represent a word as an integer from 0 to q**n - 1
    # But for distance calculation, list/tuple is easier.
    # For speed, we can use base-q representation.
    
    # 1. Initial Construction: Direct sum of tetracode and all words of length n-4
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Generate tails
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    # Construct initial codewords
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    # Convert to numpy array for easier manipulation? 
    # Actually, lists of tuples are fine for the coverage logic if we precompute neighbors.
    
    # Precompute neighbors for each possible word? No, space is 729.
    # Let's map word -> index.
    # Index = sum(w[i] * q**(n-1-i))
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors (distance <= radius) for each word in the space
    # Space size is q**n = 729
    space_size = q ** n
    neighbors = [[] for _ in range(space_size)]
    
    for i in range(space_size):
        w = idx_to_word(i)
        # Neighbor is w itself
        neighbors[i].append(i)
        # Neighbors are words differing in exactly 1 position
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val != original_val:
                    if radius >= 1:
                        w[pos] = new_val
                        neighbors[i].append(word_to_idx(w))
            w[pos] = original_val
            
    # Initialize coverage counts
    counts = np.zeros(space_size, dtype=np.int32)
    current_codewords = initial_codewords
    
    # Calculate initial counts
    for c in current_codewords:
        idx_c = word_to_idx(c)
        for nb in neighbors[idx_c]:
            counts[nb] += 1
            
    # Greedy Deletion Loop
    # We try to remove codewords that are redundant.
    # A codeword is redundant if removing it doesn't make any count drop to 0.
    
    # To be efficient, we can iterate in a random order or fixed order.
    # Fixed order is deterministic and fine.
    # We need to keep track of which codewords are currently active.
    # active_mask or just a list.
    
    # Let's store codewords as a list.
    # We will iterate multiple passes.
    
    def try_delete(codewords, cnts):
        """Try to delete redundant codewords. Returns (new_codewords, new_counts, removed_count)"""
        removed = 0
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = codewords[i]
            idx_c = word_to_idx(c)
            
            # Check if any neighbor count would become 0
            # We need to know if min(counts[nb] for nb in neighbors[idx_c]) > 1
            # If yes, we can remove c.
            
            # Optimization: 
            # If we just subtracted counts for previous removals, counts are current.
            # We need to check if removing c causes any count to be 0.
            # This is equivalent to checking if any neighbor has count == 1.
            
            # If any neighbor has count 1, c is essential (for now).
            # If all neighbors have count > 1, c is redundant.
            
            # We can check this by looking at the neighbors.
            redundant = True
            for nb in neighbors[idx_c]:
                if cnts[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                # Update counts
                for nb in neighbors[idx_c]:
                    cnts[nb] -= 1
                codewords.pop(i)
                removed += 1
        return removed
    
    # Multiple passes with different orderings
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        removed_any = False
        # Try different orderings: reverse, random, etc.
        order = list(range(len(current_codewords)))
        random.shuffle(order)
        
        for idx in reversed(order):
            if time.time() > deadline:
                break
            if idx >= len(current_codewords):
                continue
            c = current_codewords[idx]
            idx_c = word_to_idx(c)
            
            redundant = True
            for nb in neighbors[idx_c]:
                if counts[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                for nb in neighbors[idx_c]:
                    counts[nb] -= 1
                current_codewords.pop(idx)
                removed_any = True
                # Adjust order indices
                for j in range(idx, len(order)):
                    if order[j] > idx:
                        order[j] -= 1
        
        if not removed_any:
            break

    # Local search: try to swap a codeword with another to potentially allow more deletions
    best_size = len(current_codewords)
    best_codewords = current_codewords[:]
    best_counts = counts.copy()
    
    # Try swaps: replace one codeword with another from the space
    # Only try if we have time
    if time.time() < deadline - 20:
        # Precompute all possible words
        all_words = [idx_to_word(i) for i in range(space_size)]
        
        # For each codeword, try replacing it with each other word
        # and see if we can then delete more
        swap_attempts = 0
        max_swaps = 5000
        
        while swap_attempts < max_swaps and time.time() < deadline - 10:
            swap_attempts += 1
            # Pick a random codeword to potentially swap
            i = random.randint(0, len(current_codewords) - 1)
            old_c = current_codewords[i]
            old_idx = word_to_idx(old_c)
            
            # Try a random new word
            new_idx = random.randint(0, space_size - 1)
            if new_idx == old_idx:
                continue
            new_c = all_words[new_idx]
            
            # Check if new_c is already in current_codewords
            new_idx_set = set()
            for c in current_codewords:
                new_idx_set.add(word_to_idx(c))
            if new_idx in new_idx_set:
                continue
            
            # Simulate the swap
            temp_counts = counts.copy()
            # Remove old_c's contribution
            for nb in neighbors[old_idx]:
                temp_counts[nb] -= 1
            # Add new_c's contribution
            for nb in neighbors[new_idx]:
                temp_counts[nb] += 1
            
            # Check if coverage is still valid (no count < 1)
            if np.any(temp_counts < 1):
                continue
            
            # Try to delete from this state
            temp_codewords = current_codewords[:]
            temp_codewords[i] = new_c
            temp_removed = try_delete(temp_codewords, temp_counts)
            
            if len(temp_codewords) < len(current_codewords):
                # Accept the swap
                current_codewords = temp_codewords
                counts = temp_counts
                if len(current_codewords) < best_size:
                    best_size = len(current_codewords)
                    best_codewords = current_codewords[:]
                    best_counts = counts.copy()
            else:
                # Reject
                pass
        
        # Do another deletion pass after swaps
        try_delete(current_codewords, counts)
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = current_codewords[:]
            best_counts = counts.copy()

    # Return the best result found
    if len(best_codewords) <= len(current_codewords):
        return best_codewords
    else:
        return current_codewords
