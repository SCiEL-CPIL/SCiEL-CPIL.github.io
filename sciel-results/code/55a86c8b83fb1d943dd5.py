"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    total_words = q ** n  # 2187
    
    # Precompute neighbor lists for all words as integer values
    def get_neighbors(w_int):
        """Return list of integer values of words within Hamming distance 1 of w_int."""
        digits = []
        temp = w_int
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        neighbors = [w_int]
        for i in range(n):
            orig = digits[i]
            for new_val in range(q):
                if new_val != orig:
                    digits[i] = new_val
                    val = 0
                    for d in digits:
                        val = val * q + d
                    neighbors.append(val)
            digits[i] = orig
        return neighbors
    
    # Precompute all neighbor lists
    all_neighbors = [get_neighbors(i) for i in range(total_words)]
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result
    
    code_ints = [word_to_int(c) for c in initial_code]
    code_set = set(code_ints)
    
    # Coverage count using numpy for speed
    covered_count = np.zeros(total_words, dtype=np.int32)
    
    for c_int in code_ints:
        for nb in all_neighbors[c_int]:
            covered_count[nb] += 1
    
    # Simulated annealing approach
    # State: binary vector of which words are in the code
    # Energy: penalty for uncovered words + number of codewords
    
    def compute_energy(covered_count, num_codewords):
        """Compute energy: heavy penalty for uncovered words + codeword count."""
        uncovered = np.sum(covered_count == 0)
        return uncovered * 100 + num_codewords
    
    # Start from current solution
    current_energy = compute_energy(covered_count, len(code_ints))
    best_code_ints = list(code_ints)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    iteration = 0
    
    while time.time() < deadline - 10:
        iteration += 1
        
        # Try a move: remove a random codeword or add a random codeword
        move_type = random.random()
        
        if move_type < 0.6 and len(code_ints) > 0:
            # Try removing a codeword
            idx = random.randint(0, len(code_ints) - 1)
            c_int = code_ints[idx]
            
            # Temporarily remove
            for nb in all_neighbors[c_int]:
                covered_count[nb] -= 1
            
            new_num = len(code_ints) - 1
            new_energy = compute_energy(covered_count, new_num)
            
            # Accept if better, or with probability exp(-(new_energy - current_energy)/T)
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / T):
                code_ints.pop(idx)
                code_set.remove(c_int)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code_ints = list(code_ints)
            else:
                # Roll back
                for nb in all_neighbors[c_int]:
                    covered_count[nb] += 1
        
        elif move_type < 0.8:
            # Try adding a codeword
            new_int = random.randint(0, total_words - 1)
            if new_int in code_set:
                continue
            
            # Temporarily add
            for nb in all_neighbors[new_int]:
                covered_count[nb] += 1
            
            new_num = len(code_ints) + 1
            new_energy = compute_energy(covered_count, new_num)
            
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / T):
                code_ints.append(new_int)
                code_set.add(new_int)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code_ints = list(code_ints)
            else:
                # Roll back
                for nb in all_neighbors[new_int]:
                    covered_count[nb] -= 1
        
        else:
            # Try swap: remove one, add another
            if len(code_ints) == 0:
                continue
                
            remove_idx = random.randint(0, len(code_ints) - 1)
            remove_int = code_ints[remove_idx]
            add_int = random.randint(0, total_words - 1)
            
            if add_int in code_set:
                continue
            
            # Temporarily apply swap
            for nb in all_neighbors[remove_int]:
                covered_count[nb] -= 1
            for nb in all_neighbors[add_int]:
                covered_count[nb] += 1
            
            new_num = len(code_ints)
            new_energy = compute_energy(covered_count, new_num)
            
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / T):
                code_ints[remove_idx] = add_int
                code_set.remove(remove_int)
                code_set.add(add_int)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code_ints = list(code_ints)
            else:
                # Roll back
                for nb in all_neighbors[add_int]:
                    covered_count[nb] -= 1
                for nb in all_neighbors[remove_int]:
                    covered_count[nb] += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Use best solution found
    code_ints = best_code_ints
    
    # Final repair: ensure full coverage
    # Recompute coverage
    covered_count = np.zeros(total_words, dtype=np.int32)
    for c_int in code_ints:
        for nb in all_neighbors[c_int]:
            covered_count[nb] += 1
    
    # Add codewords for uncovered words
    for w in range(total_words):
        if covered_count[w] == 0 and time.time() < deadline - 5:
            code_ints.append(w)
            for nb in all_neighbors[w]:
                covered_count[nb] += 1
    
    # Final pruning pass
    for _ in range(100):
        if time.time() > deadline - 3:
            break
        removed = False
        for idx in range(len(code_ints)):
            c_int = code_ints[idx]
            can_remove = True
            for nb in all_neighbors[c_int]:
                if covered_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for nb in all_neighbors[c_int]:
                    covered_count[nb] -= 1
                code_ints.pop(idx)
                removed = True
                break
        if not removed:
            break
    
    # Convert to list of words
    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word
    
    return [int_to_word(c) for c in code_ints]
# EVOLVE-BLOCK-END
