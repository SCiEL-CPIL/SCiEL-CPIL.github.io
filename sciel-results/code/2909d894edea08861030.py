import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Build cover matrix
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            gains = cm_int8.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random choice among best
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = 0
            
        return code

    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        
        if max_time is None:
            max_time = deadline - time.time()
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        # Update cover_count
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline - 2.0:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Check if removable: all covered words have count > 1
                min_count = np.min(cover_count[mask])
                
                if min_count > 1:
                    # Remove
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try swap
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        if len(unique_indices) > 0:
                            # Create exclusion mask
                            exclude_mask = np.zeros(total_space, dtype=bool)
                            for j in current_code:
                                exclude_mask[j] = True
                            
                            # Find candidates that cover all unique words
                            candidates = np.where(cover_matrix[:, unique_indices].all(axis=1))[0]
                            candidates = candidates[~exclude_mask[candidates]]
                        
                            if len(candidates) > 0:
                                new_idx = candidates[random.randint(0, len(candidates)-1)]
                                
                                # Update cover_count
                                cover_count -= mask
                                cover_count += cover_matrix[new_idx]
                                
                                current_code[i] = int(new_idx)
                                improved = True
                                break
                            
        return current_code

    def simulated_annealing(code, max_time=None):
        if not code:
            return code
        
        if max_time is None:
            max_time = deadline - time.time()
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        # Energy function: heavily penalize uncovered words, lightly penalize code size
        def energy(code_list, cc):
            uncovered_count = np.sum(cc == 0)
            return uncovered_count * 1000 + len(code_list)
        
        current_energy = energy(current_code, cover_count)
        best_code = current_code[:]
        best_energy = current_energy
        
        T_start = 100.0
        T_end = 1.0
        cooling_rate = 0.9995
        T = T_start
        
        sa_start = time.time()
        
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
            
            # Random move: add, remove, or swap a codeword
            move_type = random.random()
            
            if move_type < 0.3 and len(current_code) < total_space:
                # Add a random codeword
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= cooling_rate
                    continue
                
                # Compute new energy
                new_cover_count = cover_count.copy()
                new_cover_count += cover_matrix[new_idx]
                new_code = current_code + [new_idx]
                new_energy = energy(new_code, new_cover_count)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_code = new_code
                    cover_count = new_cover_count
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        
            elif move_type < 0.6 and len(current_code) > 1:
                # Remove a random codeword
                remove_idx = random.randint(0, len(current_code) - 1)
                idx = current_code[remove_idx]
                
                # Compute new energy
                new_cover_count = cover_count.copy()
                new_cover_count -= cover_matrix[idx]
                new_code = current_code[:remove_idx] + current_code[remove_idx+1:]
                new_energy = energy(new_code, new_cover_count)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_code = new_code
                    cover_count = new_cover_count
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        
            else:
                # Swap: replace a random codeword with another
                if len(current_code) == 0:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[remove_pos]
                new_idx = random.randint(0, total_space - 1)
                
                if new_idx == old_idx:
                    T *= cooling_rate
                    continue
                
                # Compute new energy
                new_cover_count = cover_count.copy()
                new_cover_count -= cover_matrix[old_idx]
                new_cover_count += cover_matrix[new_idx]
                new_code = current_code[:]
                new_code[remove_pos] = new_idx
                new_energy = energy(new_code, new_cover_count)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_code = new_code
                    cover_count = new_cover_count
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
            
            T *= cooling_rate
        
        # Return the best found (which should be valid if energy < 1000)
        # Check if best is valid
        if best_energy < 1000:
            return best_code
        else:
            # Fallback to current if it's valid
            if current_energy < 1000:
                return current_code
            else:
                return code  # Return original if neither is valid

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run restarts if time permits
    while time.time() < deadline - 15.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        
        # Apply simulated annealing to further reduce
        cand_code = simulated_annealing(cand_code)
        
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Final polish: run local search and SA on best_code with remaining time
    best_code = local_search_reduce(best_code)
    if time.time() < deadline - 5.0:
        best_code = simulated_annealing(best_code)
    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
