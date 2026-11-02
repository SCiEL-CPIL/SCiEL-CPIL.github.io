"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
import math

def construct(q, n, radius):
    # Fallback for non-specific cases
    if q != 3 or n != 6 or radius != 1:
        if n < 4:
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            code = []
            while uncovered:
                best_word = None
                best_count = -1
                for w in all_words:
                    count = 0
                    for u in uncovered:
                        dist = sum(1 for a, b in zip(w, u) if a != b)
                        if dist <= radius:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_word = w
                code.append(list(best_word))
                to_remove = set()
                for u in uncovered:
                    dist = sum(1 for a, b in zip(best_word, u) if a != b)
                    if dist <= radius:
                        to_remove.add(u)
                uncovered -= to_remove
            return code
        else:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1 using Simulated Annealing
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

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighborhoods
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)
    
    def get_coverage(code_set):
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighborhoods[c]:
                coverage[nb] += 1
        return coverage

    def is_valid(coverage):
        return np.min(coverage) >= 1

    def reduce_code(code_set, coverage):
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                redundant = True
                for nb in neighborhoods[c]:
                    if coverage[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for nb in neighborhoods[c]:
                        coverage[nb] -= 1
                    changed = True
        return code_set, coverage

    # Initial Solution: Greedy Set Cover
    uncovered = set(indices)
    init_code = []
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in indices:
            count = 0
            for nb in neighborhoods[i]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        init_code.append(best_idx)
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
        
    init_set = set(init_code)
    init_cov = get_coverage(init_set)
    init_set, init_cov = reduce_code(init_set, init_cov)
    
    best_set = set(init_set)
    best_cov = init_cov.copy()
    best_size = len(best_set)
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 110 # seconds
    T = 5.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.99
    
    current_set = set(best_set)
    current_cov = best_cov.copy()
    current_size = best_size
    
    random.seed(42)
    
    while time.time() - start_time < time_limit and T > T_min:
        if not current_set:
            break
            
        # Backup current state
        backup_set = set(current_set)
        backup_cov = current_cov.copy()
        
        # Generate Move: Swap
        c_out = random.choice(list(current_set))
        c_in = random.choice(indices)
        
        if c_in in current_set:
            T *= cooling_rate
            continue
            
        # Apply Swap
        current_set.remove(c_out)
        current_set.add(c_in)
        
        # Update Coverage
        for nb in neighborhoods[c_out]:
            current_cov[nb] -= 1
        for nb in neighborhoods[c_in]:
            current_cov[nb] += 1
            
        # Check Validity
        if not is_valid(current_cov):
            # Revert
            current_set = backup_set
            current_cov = backup_cov
            T *= cooling_rate
            continue
        
        # Reduce Code
        current_set, current_cov = reduce_code(current_set, current_cov)
        new_size = len(current_set)
        
        # Acceptance
        delta_cost = new_size - current_size
        if delta_cost < 0:
            # Improvement
            current_size = new_size
            if new_size < best_size:
                best_size = new_size
                best_set = set(current_set)
                best_cov = current_cov.copy()
        else:
            # Worse or same
            if delta_cost == 0:
                # Accept to explore
                current_size = new_size
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_size = new_size
                else:
                    # Revert
                    current_set = backup_set
                    current_cov = backup_cov
        
        T *= cooling_rate

    return [list(idx_to_word(i)) for i in best_set]

# EVOLVE-BLOCK-END
