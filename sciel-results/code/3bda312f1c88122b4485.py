import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # 1. Generate initial code (Tetracode product)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
            
    # Ensure distinct
    initial_code = list(set(tuple(c) for c in initial_code))
    initial_code = [list(c) for c in initial_code]
    
    # 2. Precompute word-to-index and neighbors
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    neighbors = []
    for w in all_words:
        nbs = [word_to_idx[w]]
        for i in range(n):
            orig = w[i]
            for s in range(q):
                if s == orig: continue
                new_w = list(w)
                new_w[i] = s
                nbs.append(word_to_idx[tuple(new_w)])
        neighbors.append(nbs)
        
    # 3. Initialize cover counts
    cover_count = np.zeros(num_words, dtype=np.int32)
    current_code_indices = [word_to_idx[tuple(c)] for c in initial_code]
    
    for idx in current_code_indices:
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    # Helper to check if a codeword index can be removed
    def can_remove(idx):
        for nb in neighbors[idx]:
            if cover_count[nb] <= 1:
                return False
        return True

    # Helper to remove a codeword
    def remove_codeword(idx):
        for nb in neighbors[idx]:
            cover_count[nb] -= 1
            
    # Helper to add a codeword
    def add_codeword(idx):
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    # 4. Greedy Removal
    changed = True
    while changed:
        changed = False
        for i in range(len(current_code_indices)):
            if can_remove(current_code_indices[i]):
                remove_codeword(current_code_indices[i])
                current_code_indices.pop(i)
                changed = True
                break # Restart scan
                
    best_size = len(current_code_indices)
    best_indices = list(current_code_indices)
    
    # 5. Local Search (Swaps)
    start_time = time.time()
    deadline = 110 # 10s margin
    
    # Randomize for reproducibility? No, just use system random
    random.seed(42)
    
    while time.time() < deadline:
        # Try a swap
        if len(current_code_indices) == 0: break
        
        i = random.randint(0, len(current_code_indices) - 1)
        c_old_idx = current_code_indices[i]
        
        # Choose a new word. 
        # Strategy: Pick a random word from the space, or a neighbor of c_old.
        # Neighbors are more likely to be useful for covering similar areas.
        # Let's mix: 50% random, 50% neighbor.
        if random.random() < 0.5:
            c_new_idx = random.randint(0, num_words - 1)
        else:
            c_new_idx = random.choice(neighbors[c_old_idx])
            
        if c_new_idx in current_code_indices:
            continue
            
        # Try swap
        # 1. Remove c_old
        remove_codeword(c_old_idx)
        
        # 2. Check validity
        valid = True
        for nb in neighbors[c_old_idx]:
            if cover_count[nb] == 0:
                valid = False
                break
                
        if not valid:
            # Revert
            add_codeword(c_old_idx)
            continue
            
        # 3. Add c_new
        add_codeword(c_new_idx)
        
        # Update list
        current_code_indices[i] = c_new_idx
        
        # 4. Try to remove any now-removable codewords
        changed = True
        while changed:
            changed = False
            for j in range(len(current_code_indices)):
                if can_remove(current_code_indices[j]):
                    remove_codeword(current_code_indices[j])
                    current_code_indices.pop(j)
                    changed = True
                    break
                    
        if len(current_code_indices) < best_size:
            best_size = len(current_code_indices)
            best_indices = list(current_code_indices)
            
    # Return the best found
    final_code = [list(all_words[idx]) for idx in best_indices]
    return final_code
