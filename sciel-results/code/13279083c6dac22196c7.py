import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        # Use a simple greedy with bitsets
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as bitsets (Python int)
        neighbors = [0] * N
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    j = word_to_idx[new_word]
                    mask |= (1 << j)
            neighbors[i] = mask
            
        uncovered = (1 << N) - 1
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = (neighbors[c] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            uncovered &= ~neighbors[best_c]
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words) # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as bitsets (Python int)
    # This is much faster for set operations than lists of indices
    neighbors = [0] * N
    for i, w in enumerate(all_words):
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                new_word = tuple(new_word)
                j = word_to_idx[new_word]
                mask |= (1 << j)
        neighbors[i] = mask
        
    # Helper: Prune redundant codewords from a list of indices
    # Returns a new list
    def prune(code):
        if not code:
            return []
        code_list = list(code)
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid order bias
            random.shuffle(code_list)
            for i in range(len(code_list)):
                c = code_list[i]
                # Compute mask of all others
                mask_others = 0
                for j in range(len(code_list)):
                    if j == i:
                        continue
                    mask_others |= neighbors[code_list[j]]
                
                if (neighbors[c] & ~mask_others) == 0:
                    # c is redundant
                    code_list.pop(i)
                    changed = True
                    break # Restart loop after removal
        return code_list

    # Initial Solution: Greedy
    def greedy_init():
        uncovered = (1 << N) - 1
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = (neighbors[c] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            uncovered &= ~neighbors[best_c]
        return solution

    random.seed(42)
    current_code = greedy_init()
    current_code = prune(current_code)
    best_code = list(current_code)
    best_size = len(best_code)
    
    start_time = time.time()
    timeout = 115.0
    
    # Local Search: 1-swap
    while time.time() - start_time < timeout:
        improved = False
        
        # Prune first
        current_code = prune(current_code)
        
        # Try swaps
        indices = list(range(len(current_code)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(current_code):
                continue
            c_i = current_code[idx]
            
            # Remove c_i
            temp_code = [c for i, c in enumerate(current_code) if i != idx]
            
            # Check validity
            mask_temp = 0
            for c in temp_code:
                mask_temp |= neighbors[c]
            
            if mask_temp == (1 << N) - 1:
                # Valid, size decreased
                current_code = temp_code
                improved = True
                break
            
            # Invalid, try to repair with one swap
            uncovered = ((1 << N) - 1) & ~mask_temp
            if uncovered == 0:
                current_code = temp_code
                improved = True
                break
            
            best_c_j = -1
            best_gain = -1
            # Scan all c_j not in current_code
            for c_j in range(N):
                if c_j in temp_code:
                    continue
                gain = (neighbors[c_j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c_j = c_j
            
            if best_c_j != -1 and best_gain > 0:
                # Try adding best_c_j
                candidate_code = temp_code + [best_c_j]
                # Prune candidate
                candidate_code = prune(candidate_code)
                
                if len(candidate_code) < len(current_code):
                    current_code = candidate_code
                    improved = True
                    break
                elif len(candidate_code) == len(current_code):
                    # Accept neutral swap to escape local minima?
                    if set(candidate_code) != set(current_code):
                        current_code = candidate_code
                        break
                        
        if not improved:
            # Perturb if stuck
            if random.random() < 0.1:
                k = random.randint(1, 2)
                for _ in range(k):
                    if current_code:
                        idx_to_remove = random.randint(0, len(current_code)-1)
                        current_code.pop(idx_to_remove)
                
                # Repair
                mask_current = 0
                for c in current_code:
                    mask_current |= neighbors[c]
                uncovered = ((1 << N) - 1) & ~mask_current
                while uncovered:
                    best_c = -1
                    best_gain = -1
                    for c in range(N):
                        if c in current_code:
                            continue
                        gain = (neighbors[c] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    if best_c == -1:
                        break
                    current_code.append(best_c)
                    uncovered &= ~neighbors[best_c]
                
                current_code = prune(current_code)
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            
    return [list(all_words[i]) for i in best_code]
