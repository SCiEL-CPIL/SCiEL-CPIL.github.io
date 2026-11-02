"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # For the specific case of q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if n < 4:
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            code = []
            while uncovered:
                best_word = None
                best_count = -1
                for w in all_words:
                    count = sum(1 for u in uncovered if sum(1 for a, b in zip(w, u) if a != b) <= radius)
                    if count > best_count:
                        best_count = count
                        best_word = w
                code.append(list(best_word))
                to_remove = set()
                for u in uncovered:
                    if sum(1 for a, b in zip(best_word, u) if a != b) <= radius:
                        to_remove.add(u)
                uncovered -= to_remove
            return code
        else:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** 6
    # Precompute neighborhoods
    def word_to_idx(word):
        idx = 0
        for i in range(6):
            idx = idx * 3 + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * 6
        for i in range(5, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    neighborhoods = []
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = {i}
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.add(word_to_idx(new_w))
        neighborhoods.append(neigh)

    # 1. Greedy Set Cover to get an initial valid solution
    uncovered = set(range(total_words))
    code_indices = []
    
    # Pre-compute coverage potential for speed? No, just iterate.
    # To speed up greedy, we can pick randomly from top candidates.
    while uncovered:
        best_idx = -1
        best_count = -1
        # Sample a subset of candidates to speed up, or just scan all. 729 is small.
        for i in range(total_words):
            count = 0
            for nb in neighborhoods[i]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        
        code_indices.append(best_idx)
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    current_code = set(code_indices)
    current_size = len(current_code)
    
    # 2. Local Search: Simulated Annealing / Iterated Local Search
    # We want to minimize size.
    # State: set of codewords.
    # Moves:
    #  a. Remove a codeword if valid (size - 1).
    #  b. Swap a codeword with another word. If valid, keep. Then try to remove.
    #  c. Add a codeword (size + 1) if it allows removing 2+ others? Hard to check directly.
    
    # We will use a "Best Improvement" local search with random restarts.
    
    def get_coverage(code_set):
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighborhoods[c]:
                coverage[nb] += 1
        return coverage

    def is_valid(coverage):
        return np.all(coverage >= 1)

    best_code = set(current_code)
    best_size = current_size
    
    start_time = time.time()
    time_limit = 110
    
    random.seed(42)
    
    # Iterated Local Search
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        code_set = set(best_code)
        coverage = get_coverage(code_set)
        
        # Local Search Phase
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant codewords
            for c in list(code_set):
                # Check if any neighbor has coverage 1
                removable = True
                for nb in neighborhoods[c]:
                    if coverage[nb] == 1:
                        removable = False
                        break
                if removable:
                    code_set.remove(c)
                    for nb in neighborhoods[c]:
                        coverage[nb] -= 1
                    improved = True
                    best_size = len(code_set)
                    best_code = set(code_set)
                    
            # 2. Try swaps to enable removals
            # For each codeword, try to move it to a random position
            # If the new position is valid, check for removals.
            # To be efficient, we only try swaps that might help.
            # Actually, let's just try random swaps on the current set.
            
            code_list = list(code_set)
            random.shuffle(code_list)
            
            for c_out in code_list:
                if time.time() - start_time > time_limit:
                    break
                    
                # Try a few random candidates for c_in
                for _ in range(20):
                    c_in = random.randint(0, total_words - 1)
                    if c_in in code_set:
                        continue
                        
                    # Tentative swap
                    code_set.remove(c_out)
                    code_set.add(c_in)
                    
                    # Update coverage
                    for nb in neighborhoods[c_out]:
                        coverage[nb] -= 1
                    for nb in neighborhoods[c_in]:
                        coverage[nb] += 1
                        
                    if is_valid(coverage):
                        # Check for removals
                        removed_any = False
                        for c in list(code_set):
                            removable = True
                            for nb in neighborhoods[c]:
                                if coverage[nb] == 1:
                                    removable = False
                                    break
                            if removable:
                                code_set.remove(c)
                                for nb in neighborhoods[c]:
                                    coverage[nb] -= 1
                                removed_any = True
                                best_size = len(code_set)
                                best_code = set(code_set)
                                
                        if removed_any:
                            improved = True
                            # Keep the swap and removals
                            break
                        else:
                            # Valid but no improvement, revert
                            code_set.remove(c_in)
                            code_set.add(c_out)
                            for nb in neighborhoods[c_in]:
                                coverage[nb] -= 1
                            for nb in neighborhoods[c_out]:
                                coverage[nb] += 1
                    else:
                        # Invalid, revert
                        code_set.remove(c_in)
                        code_set.add(c_out)
                        for nb in neighborhoods[c_in]:
                            coverage[nb] -= 1
                        for nb in neighborhoods[c_out]:
                            coverage[nb] += 1
                        
        # If we improved, continue local search from new best
        # If not, we might need to perturb.
        # For simplicity, we restart from best_code.
        
    return [list(idx_to_word(i)) for i in best_code]

# EVOLVE-BLOCK-END
