"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
import math

def construct(q, n, radius):
    # Parameters for the specific task
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (simple greedy)
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            best_count = -1
            for w in all_words:
                count = sum(1 for u in uncovered if sum(a != b for a, b in zip(w, u)) <= radius)
                if count > best_count:
                    best_count = count
                    best_word = w
            code.append(list(best_word))
            uncovered = {u for u in uncovered if sum(a != b for a, b in zip(best_word, u)) > radius}
        return code

    # Specific optimization for q=3, n=6, r=1
    total_words = q ** n
    indices = list(range(total_words))
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighborhoods for all possible words
    # A word covers itself and all words at distance 1
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = set([i])
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.add(word_to_idx(new_w))
        neighborhoods.append(list(neigh))
    
    # Greedy Initialization
    uncovered = set(indices)
    code_set = set()
    coverage = np.zeros(total_words, dtype=np.int32)
    
    while uncovered:
        best_idx = -1
        best_count = -1
        # Randomize tie-breaking to get diverse starts? No, deterministic is fine for init.
        for i in indices:
            if i in code_set:
                continue
            count = sum(1 for nb in neighborhoods[i] if nb in uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        
        code_set.add(best_idx)
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.discard(nb)
                coverage[nb] += 1
            else:
                coverage[nb] += 1
        
    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0
    current_code = set(code_set)
    current_coverage = coverage.copy()
    current_size = len(current_code)
    
    best_code = set(current_code)
    best_size = current_size
    best_coverage = current_coverage.copy()
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    
    random.seed(42)
    
    while time.time() - start_time < time_limit and T > T_min:
        # Choose a move: Remove or Add
        # Probability of Remove vs Add depends on current state
        # If we can remove, we should try to remove more often?
        # Let's try 50% Remove, 50% Add.
        
        if random.random() < 0.5:
            # Try Remove
            if len(current_code) == 0:
                continue
            c_out = random.choice(list(current_code))
            
            # Check if removing c_out keeps all words covered
            valid = True
            for nb in neighborhoods[c_out]:
                if current_coverage[nb] <= 1:
                    valid = False
                    break
            
            if valid:
                # Apply
                current_code.remove(c_out)
                for nb in neighborhoods[c_out]:
                    current_coverage[nb] -= 1
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = set(current_code)
                    best_coverage = current_coverage.copy()
            else:
                # Rejected, do nothing
                pass
        else:
            # Try Add
            # Pick a random word not in code
            # To be efficient, we can pick a random index and check if it's in code
            # Or maintain a list of candidates? Set membership is O(1).
            c_in = random.choice(indices)
            if c_in in current_code:
                continue
            
            # Adding is always valid (increases coverage)
            # But it increases size, so we only accept if T is high enough
            # Delta = +1 (size increases by 1)
            # P(accept) = exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                current_code.add(c_in)
                for nb in neighborhoods[c_in]:
                    current_coverage[nb] += 1
                current_size += 1
            
        T *= cooling_rate
        
    # Final cleanup: try to remove redundant codewords from the best found
    final_code = set(best_code)
    final_coverage = best_coverage.copy()
    
    changed = True
    while changed:
        changed = False
        for c in list(final_code):
            removable = True
            for nb in neighborhoods[c]:
                if final_coverage[nb] <= 1:
                    removable = False
                    break
            if removable:
                final_code.remove(c)
                for nb in neighborhoods[c]:
                    final_coverage[nb] -= 1
                changed = True

    return [list(idx_to_word(i)) for i in final_code]

# EVOLVE-BLOCK-END
