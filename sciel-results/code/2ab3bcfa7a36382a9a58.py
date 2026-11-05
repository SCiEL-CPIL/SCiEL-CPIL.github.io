import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to the task: q=3, n=8, radius=1
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists
    # neighbors[i] is a numpy array of indices j such that dist(i, j) <= radius
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
        
    # --- Phase 1: Fast Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words
    
    while uncovered_remaining > 0 and time.time() < deadline - 60.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
            
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1
                
    current_code = code_list[:]
    current_size = len(current_code)
    
    # --- Phase 2: Simulated Annealing ---
    # Objective: Minimize size. 
    # State: set of codewords.
    # Moves: 
    #   1. Remove a random codeword (if valid)
    #   2. Swap a random codeword with a random non-codeword (if valid)
    #   3. Add a random codeword (if valid, though this increases size, used for perturbation)
    # Actually, standard SA for set cover minimization often allows invalid states 
    # with a penalty, but here we must return a VALID code. 
    # So we restrict moves to valid states (like the previous program) but use 
    # temperature to accept "worse" moves? 
    # Since we minimize size, a "worse" move is one that increases size. 
    # But we can't increase size and stay valid if we just add? 
    # If we add a word, size increases. Is it valid? Yes. 
    # So we can add words to escape local minima where no removal/swap is possible.
    
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    best_code = current_code[:]
    best_size = current_size
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute random non-code indices for speed? 
    # No, just pick random and check set.
    
    while time.time() < deadline - 5.0 and T > T_min:
        # 1. Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a random word
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                c = current_code[idx]
                
                # Check if removable
                # Vectorized check
                if np.all(current_cov_count[neighbors[c]] > 1):
                    # Remove
                    current_cov_count[neighbors[c]] -= 1
                    current_is_code[c] = False
                    current_code_set.remove(c)
                    current_code.pop(idx)
                    current_size -= 1
                    T = max(T * cooling_rate, T_min)
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                else:
                    # Cannot remove, stay (or try swap?)
                    pass
        elif move_type < 0.9:
            # Try to swap
            if current_size > 0:
                c1_idx = random.randint(0, current_size - 1)
                c1 = current_code[c1_idx]
                
                # Pick c2 not in code
                c2_found = False
                for _ in range(20):
                    c2 = random.randint(0, total_words - 1)
                    if c2 not in current_code_set:
                        c2_found = True
                        break
                
                if c2_found:
                    # Check validity of swap
                    # Words dropping to 0: x in N(c1) s.t. cov[x]==1 and x not in N(c2)
                    set_c2 = set(neighbors[c2].tolist())
                    invalid = False
                    for x in neighbors[c1]:
                        if current_cov_count[x] == 1 and x not in set_c2:
                            invalid = True
                            break
                    
                    if not invalid:
                        # Apply swap
                        current_cov_count[neighbors[c1]] -= 1
                        current_is_code[c1] = False
                        current_code_set.remove(c1)
                        current_code[c1_idx] = c2
                        
                        current_cov_count[neighbors[c2]] += 1
                        current_is_code[c2] = True
                        current_code_set.add(c2)
                        
                        T = max(T * cooling_rate, T_min)
        else:
            # Perturbation: Add a random word
            # This increases size, so it's a "bad" move, but might allow future removals
            c_add = random.randint(0, total_words - 1)
            if c_add not in current_code_set:
                current_cov_count[neighbors[c_add]] += 1
                current_is_code[c_add] = True
                current_code_set.add(c_add)
                current_code.append(c_add)
                current_size += 1
                
                # Accept with probability exp(-delta_E / T)
                # delta_E = +1 (size increase)
                # p = exp(-1/T)
                if random.random() > np.exp(-1.0 / T):
                    # Reject move, revert
                    current_cov_count[neighbors[c_add]] -= 1
                    current_is_code[c_add] = False
                    current_code_set.remove(c_add)
                    current_code.pop()
                    current_size -= 1
                
                T = max(T * cooling_rate, T_min)
            
    # Final Polish: Try to remove more words from best_code
    final_code = best_code[:]
    final_code_set = set(final_code)
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code_set.remove(c)
                final_code.pop(i)
                improved = True
                
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
