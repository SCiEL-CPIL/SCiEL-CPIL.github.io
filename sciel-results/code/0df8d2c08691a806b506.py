import numpy as np
import time
import itertools
import random
from collections import Counter

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    space_size = q ** n
    neighbors_per_word = 1 + n * (q - 1)
    
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

    # Precompute neighbors for each word in the space using numpy
    neighbors = np.zeros((space_size, neighbors_per_word), dtype=np.int32)
    
    for i in range(space_size):
        w = idx_to_word(i)
        neighbors[i, 0] = i  # itself
        pos = 1
        for p in range(n):
            original_val = w[p]
            for new_val in range(q):
                if new_val != original_val:
                    if radius >= 1:
                        w[p] = new_val
                        neighbors[i, pos] = word_to_idx(w)
                        pos += 1
            w[p] = original_val
    
    # Convert to sets for faster membership tests
    neighbor_sets = [set(nb.tolist()) for nb in neighbors]
    
    # Initial construction: direct sum of tetracode and full space of length n-4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
    
    def compute_counts(codewords):
        counts = np.zeros(space_size, dtype=np.int32)
        for c in codewords:
            idx_c = word_to_idx(c)
            counts[neighbors[idx_c]] += 1
        return counts

    def greedy_delete(codewords, counts, deadline):
        """Perform greedy deletion to reduce codeword count."""
        current = list(codewords)
        cnt = counts.copy()
        
        max_passes = 100
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            removed_any = False
            # Process in reverse order to avoid index issues
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                idx_c = word_to_idx(current[i])
                
                # Check if any neighbor has count == 1
                if np.any(cnt[neighbors[idx_c]] == 1):
                    continue
                
                # Remove it
                cnt[neighbors[idx_c]] -= 1
                current.pop(i)
                removed_any = True
                
            if not removed_any:
                break
        
        return current, cnt

    def local_search(codewords, counts, deadline, max_iterations=5000):
        """Local search with moves: remove, swap, add+remove."""
        current = list(codewords)
        cnt = counts.copy()
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to remove a codeword
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                
                idx_c = word_to_idx(current[i])
                
                # Check if this codeword can be removed
                if not np.any(cnt[neighbors[idx_c]] == 1):
                    # Remove it
                    cnt[neighbors[idx_c]] -= 1
                    current.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: replace codeword i with a random word
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                
                idx_c = word_to_idx(current[i])
                
                # Find essential neighbors (only covered by this codeword)
                essential = neighbors[idx_c][cnt[neighbors[idx_c]] == 1]
                
                if len(essential) == 0:
                    # Can remove directly
                    cnt[neighbors[idx_c]] -= 1
                    current.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement
                # Sample candidates near the essential neighbors
                candidates = set()
                for nb in essential:
                    candidates.add(nb)
                    # Also add neighbors of essential words
                    for nb2 in neighbors[nb]:
                        candidates.add(nb2)
                
                # Add some random candidates
                for _ in range(10):
                    candidates.add(random.randint(0, space_size - 1))
                
                for cand_idx in candidates:
                    if time.time() > deadline:
                        return current, cnt
                    
                    if cand_idx == idx_c:
                        continue
                    
                    # Check if candidate covers all essential neighbors
                    cand_nb = neighbors[cand_idx]
                    cand_nb_set = set(cand_nb.tolist())
                    covers_essential = all(nb in cand_nb_set for nb in essential)
                    if not covers_essential:
                        continue
                    
                    # Simulate the swap
                    # Words that lose coverage: in neighbors[idx_c] but not in cand_nb
                    # with count == 1
                    old_nb = neighbors[idx_c]
                    old_nb_set = set(old_nb.tolist())
                    valid = True
                    for nb in old_nb:
                        if cnt[nb] == 1 and nb not in cand_nb_set:
                            valid = False
                            break
                    
                    if valid:
                        # Perform the swap
                        cnt[old_nb] -= 1
                        cnt[cand_nb] += 1
                        current[i] = idx_to_word(cand_idx)
                        improved = True
                        break
                
                if improved:
                    break
        
        return current, cnt

    def full_optimize(codewords, deadline):
        """Run greedy deletion followed by local search."""
        counts = compute_counts(codewords)
        
        # Greedy deletion
        codewords, counts = greedy_delete(codewords, counts, deadline)
        
        # Local search
        codewords, counts = local_search(codewords, counts, deadline)
        
        # Another round of greedy deletion
        codewords, counts = greedy_delete(codewords, counts, deadline)
        
        return codewords, counts

    # Multiple restarts with different shuffles
    best_codewords = None
    best_size = float('inf')
    
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart * 7919
        random.seed(seed)
        
        shuffled = list(initial_codewords)
        random.shuffle(shuffled)
        
        codewords, counts = full_optimize(shuffled, deadline)
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords
    
    if best_codewords is None:
        best_codewords = initial_codewords
    
    return best_codewords
