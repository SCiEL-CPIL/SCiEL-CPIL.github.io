import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for cleanup/return
    
    # Generate all words in {0, 1, 2}^n
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (Hamming distance <= radius)
    # For radius=1, neighbors are the word itself + all words at distance 1
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
    
    # Convert neighbors to numpy arrays for faster iteration
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Initial construction: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.add(tuple(h + list(tail)))
    
    # State variables
    current_code = set(initial_code)
    coverage = np.zeros(num_words, dtype=np.int32)
    uncovered_count = num_words  # Initially all uncovered
    
    def add_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors_np[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
        current_code.add(w)
        
    def remove_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors_np[idx]:
            coverage[nb_idx] -= 1
            if coverage[nb_idx] == 0:
                uncovered_count += 1
        current_code.remove(w)
        
    # Initialize coverage
    for w in initial_code:
        idx = word_to_idx[w]
        for nb_idx in neighbors_np[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
            
    best_code = set(current_code)
    best_size = len(current_code)
    
    # Simulated Annealing with improved acceptance criteria
    T = 2.0
    cooling = 0.9995
    min_T = 0.0005
    
    # Precompute list of all words for random sampling
    all_words_list = all_words
    
    # Track the number of iterations for better convergence
    it_count = 0
    
    while time.time() < deadline:
        # Update best
        if uncovered_count == 0 and len(current_code) < best_size:
            best_code = set(current_code)
            best_size = len(current_code)
            
        if not current_code:
            break
            
        it_count += 1
        move_type = random.random()
        
        if move_type < 0.35:
            # Try Removal: check if any word can be removed without breaking coverage
            code_list = list(current_code)
            w_old = random.choice(code_list)
            
            # Check if removal is valid: no neighbor has coverage 1
            can_remove = True
            idx_old = word_to_idx[w_old]
            for nb_idx in neighbors_np[idx_old]:
                if coverage[nb_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                remove_word(w_old)
                # uncovered_count unchanged by valid removal
            else:
                # Try a replacement to potentially enable removal later
                # Pick a random word to replace w_old with
                w_new = random.choice(all_words_list)
                if w_new == w_old:
                    continue
                
                # Calculate the effect of replacement
                u_before = uncovered_count
                
                # Apply removal
                for nb_idx in neighbors_np[idx_old]:
                    coverage[nb_idx] -= 1
                    if coverage[nb_idx] == 0:
                        uncovered_count += 1
                current_code.remove(w_old)
                
                # Apply addition
                idx_new = word_to_idx[w_new]
                for nb_idx in neighbors_np[idx_new]:
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
                    # Neutral coverage. Accept with some probability to explore.
                    # Use a slightly higher probability for neutral moves
                    accept = random.random() < math.exp(-0.05 / T)
                else:
                    # Worsens coverage.
                    # Use a smoother penalty
                    accept = random.random() < math.exp(-1.5 * delta_u / T)
                
                if not accept:
                    # Revert
                    # Remove w_new
                    for nb_idx in neighbors_np[idx_new]:
                        coverage[nb_idx] -= 1
                        if coverage[nb_idx] == 0:
                            uncovered_count += 1
                    current_code.remove(w_new)
                    
                    # Add w_old
                    for nb_idx in neighbors_np[idx_old]:
                        if coverage[nb_idx] == 0:
                            uncovered_count -= 1
                        coverage[nb_idx] += 1
                    current_code.add(w_old)
                    
        else:
            # Replacement
            code_list = list(current_code)
            w_old = random.choice(code_list)
            w_new = random.choice(all_words_list)
            if w_new == w_old:
                continue
                
            # Save current uncovered count
            u_before = uncovered_count
            
            # Apply removal
            idx_old = word_to_idx[w_old]
            for nb_idx in neighbors_np[idx_old]:
                coverage[nb_idx] -= 1
                if coverage[nb_idx] == 0:
                    uncovered_count += 1
            current_code.remove(w_old)
            
            # Apply addition
            idx_new = word_to_idx[w_new]
            for nb_idx in neighbors_np[idx_new]:
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
                # Neutral coverage. Accept with some probability to explore.
                # Use a slightly higher probability for neutral moves
                accept = random.random() < math.exp(-0.05 / T)
            else:
                # Worsens coverage.
                # Use a smoother penalty
                accept = random.random() < math.exp(-1.5 * delta_u / T)
                
            if not accept:
                # Revert
                # Remove w_new
                for nb_idx in neighbors_np[idx_new]:
                    coverage[nb_idx] -= 1
                    if coverage[nb_idx] == 0:
                        uncovered_count += 1
                current_code.remove(w_new)
                
                # Add w_old
                for nb_idx in neighbors_np[idx_old]:
                    if coverage[nb_idx] == 0:
                        uncovered_count -= 1
                    coverage[nb_idx] += 1
                current_code.add(w_old)
                
        if T > min_T:
            T *= cooling
    
    # Final polish: try to remove any redundant words
    # Sort by how "central" they are (words that cover more uniquely covered points are more valuable to keep)
    # Actually, let's just do a simple greedy removal pass
    code_list = list(best_code)
    random.shuffle(code_list)
    
    # Rebuild coverage for best_code
    coverage_final = np.zeros(num_words, dtype=np.int32)
    uncovered_final = num_words
    for w in best_code:
        idx = word_to_idx[w]
        for nb_idx in neighbors_np[idx]:
            if coverage_final[nb_idx] == 0:
                uncovered_final -= 1
            coverage_final[nb_idx] += 1
    
    for w in code_list:
        idx = word_to_idx[w]
        can_remove = True
        for nb_idx in neighbors_np[idx]:
            if coverage_final[nb_idx] == 1:
                can_remove = False
                break
        if can_remove:
            for nb_idx in neighbors_np[idx]:
                coverage_final[nb_idx] -= 1
                if coverage_final[nb_idx] == 0:
                    uncovered_final += 1
            best_code.remove(w)
    
    return [list(w) for w in best_code]
