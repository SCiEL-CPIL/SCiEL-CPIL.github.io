import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(neigh)

    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            count = 0
            for nb in neighbors[j]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
    
    # Initial solution size: len(selected)
    # Now, try to reduce the size using local search.
    
    # Maintain coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        for nb in neighbors[j]:
            coverage_count[nb] += 1
            
    # Helper function to check if a codeword is redundant
    def is_redundant(j, cov):
        for nb in neighbors[j]:
            if cov[nb] == 1:
                return False
        return True

    # Helper function to try to reduce a set
    def try_reduce(selected_set, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                if is_redundant(j, cov):
                    for nb in neighbors[j]:
                        cov[nb] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set

    # Local Search with Simulated Annealing / Random Walk
    best_selected = list(selected)
    best_cov = coverage_count.copy()
    
    # First, try to reduce the initial greedy solution
    best_selected = try_reduce(best_selected, best_cov, deadline)
    
    # Multi-start local search
    global_best = list(best_selected)
    global_best_size = len(best_selected)
    
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        
        # Start from current best
        current_selected = list(global_best)
        current_cov = np.zeros(N, dtype=np.int32)
        for j in current_selected:
            for nb in neighbors[j]:
                current_cov[nb] += 1
        
        current_selected = try_reduce(current_selected, current_cov, deadline)
        current_size = len(current_selected)
        selected_set = set(current_selected)
        
        # If we already improved, continue
        if current_size < global_best_size:
            global_best = list(current_selected)
            global_best_size = current_size
        
        # Random walk on valid covers of current size
        while time.time() < deadline - 5.0 and current_size > 0:
            # Try to remove
            new_selected = try_reduce(current_selected, current_cov, deadline)
            if len(new_selected) < current_size:
                current_size = len(new_selected)
                current_selected = new_selected
                selected_set = set(current_selected)
                if current_size < global_best_size:
                    global_best = list(current_selected)
                    global_best_size = current_size
                continue
            
            # Try a swap: remove one, add one
            if len(current_selected) == 0:
                break
                
            i_idx = random.randint(0, len(current_selected) - 1)
            i_word = current_selected[i_idx]
            
            j_word = random.randint(0, N - 1)
            if j_word in selected_set:
                continue
                
            # Remove i
            for nb in neighbors[i_word]:
                current_cov[nb] -= 1
            
            # Check if still covered (all counts >= 1)
            valid = True
            for nb in neighbors[i_word]:
                if current_cov[nb] < 1:
                    valid = False
                    break
                    
            if not valid:
                # Revert
                for nb in neighbors[i_word]:
                    current_cov[nb] += 1
                continue
                
            # Add j
            for nb in neighbors[j_word]:
                current_cov[nb] += 1
                
            # Update selected
            temp_selected = list(current_selected)
            temp_selected[i_idx] = j_word
            selected_set = set(temp_selected)
            
            # Try to reduce
            new_selected = try_reduce(temp_selected, current_cov, deadline)
            
            if len(new_selected) < current_size:
                current_size = len(new_selected)
                current_selected = new_selected
                selected_set = set(current_selected)
                if current_size < global_best_size:
                    global_best = list(current_selected)
                    global_best_size = current_size
            else:
                # Same size: accept with high probability to enable random walk
                if random.random() < 0.7:
                    current_selected = new_selected
                    selected_set = set(current_selected)
                else:
                    # Revert
                    for nb in neighbors[j_word]:
                        current_cov[nb] -= 1
                    for nb in neighbors[i_word]:
                        current_cov[nb] += 1
                    
                    # Also need to revert any reductions in new_selected
                    # But new_selected is a local variable, so we just discard it
                    # However, current_cov was modified by try_reduce on new_selected
                    # We need to recompute current_cov from current_selected
                    current_cov = np.zeros(N, dtype=np.int32)
                    for j in current_selected:
                        for nb in neighbors[j]:
                            current_cov[nb] += 1
    
    # Final result
    best_selected = global_best
    codewords = [words[i] for i in best_selected]
    return codewords
