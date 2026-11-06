"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Precompute compatibility matrix
    # compatible[i][j] is True if lcm(i+2, j+2) > 256
    # Note: indices 0..254 correspond to values 2..256
    n_vals = 255
    compatible = [[False] * n_vals for _ in range(n_vals)]
    
    for i in range(n_vals):
        val_i = i + 2
        for j in range(n_vals):
            val_j = j + 2
            if i != j:
                compatible[i][j] = (math.lcm(val_i, val_j) > 256)
    
    # 2. Initial Solution: Greedy by descending value (largest values have smallest reciprocals, but often better compatibility? 
    #    Actually, small values have large reciprocals. Greedy by ascending value might get stuck. 
    #    Let's start with a random valid set or a greedy one.)
    
    # Greedy initial: pick smallest values that fit? 
    # Or just random valid set.
    # Let's try a simple greedy: sort by value, try to add if compatible with all current.
    # To get a good start, maybe try multiple greedy seeds or just random restarts.
    
    def get_score(selected_indices):
        if not selected_indices:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for idx in selected_indices:
            s += Fraction(1, idx + 2)
        return s

    def is_valid(selected_indices):
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                if not compatible[selected_indices[i]][selected_indices[j]]:
                    return False
        return True

    # Generate initial solution
    # Greedy: try adding elements in random order, or sorted order.
    # Let's use a randomized greedy to generate a few candidates and pick the best.
    
    best_set = []
    best_score = Fraction(0, 1)
    
    # Try 100 random greedy initializations
    for _ in range(100):
        if time.time() > deadline:
            break
        current = []
        candidates = list(range(n_vals))
        random.shuffle(candidates)
        for idx in candidates:
            if all(compatible[idx][c] for c in current):
                current.append(idx)
        
        if is_valid(current):
            score = get_score(current)
            if score > best_score:
                best_score = score
                best_set = current[:]
    
    # If no set found (unlikely), return empty? No, must be non-empty.
    # 2 is always compatible with any odd number > 128? 
    # lcm(2, x) = 2x. 2x > 256 => x > 128.
    # So {2, 129} is valid.
    if not best_set:
        best_set = [0] # {2}
        best_score = Fraction(1, 2)

    # 3. Local Search: Simulated Annealing
    # Moves:
    # 1. Add a random element (if compatible)
    # 2. Remove a random element
    # 3. Swap a random element with another (if compatible)
    
    current_set = best_set[:]
    current_score = best_score
    T = 0.1
    cooling_rate = 0.999
    
    # Precompute reciprocal values for speed
    recips = [Fraction(1, i+2) for i in range(n_vals)]
    
    while time.time() < deadline:
        # Generate move
        move_type = random.choice([0, 1, 2])
        new_set = current_set[:]
        
        if move_type == 0: # Add
            # Find a random index not in current_set
            if len(current_set) < n_vals:
                # Pick random candidate
                while True:
                    candidate = random.randrange(n_vals)
                    if candidate not in current_set:
                        break
                # Check compatibility
                if all(compatible[candidate][c] for c in current_set):
                    new_set.append(candidate)
                    delta = recips[candidate]
                    new_score = current_score + delta
                else:
                    continue
            else:
                continue
                
        elif move_type == 1: # Remove
            if current_set:
                remove_idx = random.randrange(len(current_set))
                removed_val = current_set[remove_idx]
                new_set.pop(remove_idx)
                delta = -recips[removed_val]
                new_score = current_score + delta
            else:
                continue
                
        else: # Swap
            if current_set:
                # Pick one to remove
                remove_pos = random.randrange(len(current_set))
                removed_val = current_set[remove_pos]
                
                # Pick one to add
                # Candidates: all not in current_set
                # To make it efficient, just pick random and check
                while True:
                    add_val = random.randrange(n_vals)
                    if add_val not in current_set:
                        break
                
                # Check compatibility of add_val with all in new_set (current_set minus removed_val)
                ok = True
                for c in new_set:
                    if not compatible[add_val][c]:
                        ok = False
                        break
                if ok:
                    new_set[remove_pos] = add_val
                    delta = recips[add_val] - recips[removed_val]
                    new_score = current_score + delta
                else:
                    continue
        
        # Acceptance
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            # Accept with probability exp(-delta / T)
            # delta is negative
            prob = math.exp(-delta / T) if delta < 0 else 1.0
            if random.random() < prob:
                current_set = new_set
                current_score = new_score
        
        T *= cooling_rate
    
    # Return the best set found as a list of values
    return [idx + 2 for idx in best_set]
# EVOLVE-BLOCK-END
