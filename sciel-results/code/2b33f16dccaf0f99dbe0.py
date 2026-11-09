import itertools
import random
import time
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for cleanup/return
    
    # Generate all words in {0, 1, 2}^6
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                new_w = tuple(w_list)
                neighbors[i].append(word_to_idx[new_w])
            w_list[pos] = orig_val
    
    # Initial construction: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.add(tuple(h + list(tail)))
    
    # State variables
    current_code = set(initial_code)
    coverage = [0] * num_words
    uncovered_count = num_words  # Initially all uncovered
    
    def add_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
        current_code.add(w)
        
    def remove_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            coverage[nb_idx] -= 1
            if coverage[nb_idx] == 0:
                uncovered_count += 1
        current_code.remove(w)
        
    # Initialize coverage
    for w in initial_code:
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
            
    best_code = set(current_code)
    best_size = len(current_code)
    
    # Greedy reduction: repeatedly try to remove words that don't leave any word uncovered
    while time.time() < deadline:
        removed_any = False
        for w in list(current_code):
            idx = word_to_idx[w]
            can_remove = True
            for nb_idx in neighbors[idx]:
                if coverage[nb_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                remove_word(w)
                removed_any = True
        if not removed_any:
            break
        if len(current_code) < best_size:
            best_code = set(current_code)
            best_size = len(current_code)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    while time.time() < deadline:
        # Update best
        if uncovered_count == 0 and len(current_code) < best_size:
            best_code = set(current_code)
            best_size = len(current_code)
            
        if not current_code:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Try Removal
            w_old = random.choice(list(current_code))
            
            # Check if removal is valid: no neighbor has coverage 1
            can_remove = True
            idx_old = word_to_idx[w_old]
            for nb_idx in neighbors[idx_old]:
                if coverage[nb_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                remove_word(w_old)
                # uncovered_count unchanged by valid removal
            else:
                # Try a replacement to potentially enable removal
                w_new = random.choice(all_words)
                if w_new == w_old:
                    continue
                
                u_before = uncovered_count
                
                # Apply removal
                idx_old = word_to_idx[w_old]
                for nb_idx in neighbors[idx_old]:
                    coverage[nb_idx] -= 1
                    if coverage[nb_idx] == 0:
                        uncovered_count += 1
                current_code.remove(w_old)
                
                # Apply addition
                idx_new = word_to_idx[w_new]
                for nb_idx in neighbors[idx_new]:
                    if coverage[nb_idx] == 0:
                        uncovered_count -= 1
                    coverage[nb_idx] += 1
                current_code.add(w_new)
                
                u_after = uncovered_count
                delta_u = u_after - u_before
                
                # Acceptance
                accept = False
                if delta_u < 0:
                    accept = True
                elif delta_u == 0:
                    accept = random.random() < math.exp(-0.1 / T)
                else:
                    accept = random.random() < math.exp(-2.0 * delta_u / T)
                    
                if not accept:
                    # Revert
                    for nb_idx in neighbors[idx_new]:
                        coverage[nb_idx] -= 1
                        if coverage[nb_idx] == 0:
                            uncovered_count += 1
                    current_code.remove(w_new)
                    
                    for nb_idx in neighbors[idx_old]:
                        if coverage[nb_idx] == 0:
                            uncovered_count -= 1
                        coverage[nb_idx] += 1
                    current_code.add(w_old)
                
        else:
            # Replacement
            w_old = random.choice(list(current_code))
            w_new = random.choice(all_words)
            if w_new == w_old:
                continue
                
            u_before = uncovered_count
            
            # Apply removal
            idx_old = word_to_idx[w_old]
            for nb_idx in neighbors[idx_old]:
                coverage[nb_idx] -= 1
                if coverage[nb_idx] == 0:
                    uncovered_count += 1
            current_code.remove(w_old)
            
            # Apply addition
            idx_new = word_to_idx[w_new]
            for nb_idx in neighbors[idx_new]:
                if coverage[nb_idx] == 0:
                    uncovered_count -= 1
                coverage[nb_idx] += 1
            current_code.add(w_new)
            
            u_after = uncovered_count
            delta_u = u_after - u_before
            
            # Acceptance
            accept = False
            if delta_u < 0:
                accept = True
            elif delta_u == 0:
                accept = random.random() < math.exp(-0.1 / T)
            else:
                accept = random.random() < math.exp(-2.0 * delta_u / T)
                
            if not accept:
                # Revert
                for nb_idx in neighbors[idx_new]:
                    coverage[nb_idx] -= 1
                    if coverage[nb_idx] == 0:
                        uncovered_count += 1
                current_code.remove(w_new)
                
                for nb_idx in neighbors[idx_old]:
                    if coverage[nb_idx] == 0:
                        uncovered_count -= 1
                    coverage[nb_idx] += 1
                current_code.add(w_old)
                
        if T > min_T:
            T *= cooling

    # Final greedy reduction
    while time.time() < deadline:
        removed_any = False
        for w in list(current_code):
            idx = word_to_idx[w]
            can_remove = True
            for nb_idx in neighbors[idx]:
                if coverage[nb_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                remove_word(w)
                removed_any = True
        if not removed_any:
            break
        if len(current_code) < best_size:
            best_code = set(current_code)
            best_size = len(current_code)

    return [list(w) for w in best_code]
