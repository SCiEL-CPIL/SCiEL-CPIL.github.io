# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    n = 256
    nums = list(range(1, n + 1))
    
    # Precompute gcd matrix for fast compatibility checks
    # gcd_matrix[i][j] is 1 if gcd(i, j) == 1, else 0
    # Using a list of lists for speed, or just a function
    # 256x256 is 65536 entries, very small.
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = math.gcd(i, j)
            val = 1 if g == 1 else 0
            gcd_mat[i][j] = val
            gcd_mat[j][i] = val

    def is_compatible(x, current_set):
        """Check if x is coprime to all elements in current_set"""
        for y in current_set:
            if gcd_mat[x][y] == 0:
                return False
        return True

    def is_compatible_swap(x, current_set, remove_elem):
        """Check if x is coprime to all elements in current_set except remove_elem"""
        for y in current_set:
            if y == remove_elem:
                continue
            if gcd_mat[x][y] == 0:
                return False
        return True

    # Initial Solution: Greedy
    def get_greedy():
        used = set()
        result = []
        # Try to add numbers from largest to smallest
        for x in range(n, 0, -1):
            if is_compatible(x, used):
                result.append(x)
                used.add(x)
        return result

    best_set = get_greedy()
    best_sum = sum(best_set)
    
    current_set = list(best_set)
    current_sum = best_sum
    
    start_time = time.time()
    TIME_LIMIT = 55.0
    
    # Simulated Annealing Parameters
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of numbers for random selection
    all_nums = list(range(1, n + 1))
    
    while time.time() - start_time < TIME_LIMIT and T > min_T:
        # Generate a neighbor
        # Strategy: 
        # 1. Try to add a random number
        # 2. Try to remove a random number
        # 3. Try to swap a random number in with a random number out
        
        move_type = random.randint(1, 3)
        
        if move_type == 1:
            # Add
            candidates = [x for x in all_nums if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            if is_compatible(x, current_set):
                new_set = current_set + [x]
                new_sum = current_sum + x
                delta = new_sum - current_sum
                # Always accept if better
                if delta > 0:
                    current_set = new_set
                    current_sum = new_sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = list(current_set)
            # If not compatible, we could still "accept" a worse state? 
            # No, adding an incompatible number breaks the constraint.
            # The state must always be valid.
            # So if not compatible, this move is invalid. We skip.
            # To allow exploration, we might need to allow "invalid" states? 
            # No, the problem requires valid pairs. 
            # So we are constrained to the space of valid independent sets.
            # The "energy" is just the sum. We want to maximize.
            # Since we can only move to valid states, we always accept improvements.
            # This is just Hill Climbing if we don't accept worse states.
            # But in a valid independent set, adding a number always increases sum (since x >= 1).
            # So "Add" is always an improvement if valid.
            # "Remove" is always a decrease.
            # "Swap" can be increase or decrease.
            
        elif move_type == 2:
            # Remove
            if not current_set:
                continue
            idx = random.randint(0, len(current_set) - 1)
            x = current_set[idx]
            new_set = current_set[:idx] + current_set[idx+1:]
            new_sum = current_sum - x
            delta = new_sum - current_sum
            # Accept worse state with probability exp(delta/T)
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set = new_set
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = list(current_set)
                    
        else:
            # Swap
            if not current_set:
                continue
            idx = random.randint(0, len(current_set) - 1)
            u = current_set[idx]
            
            candidates = [x for x in all_nums if x not in current_set]
            if not candidates:
                continue
            v = random.choice(candidates)
            
            # Check if v is compatible with current_set \ {u}
            if is_compatible_swap(v, current_set, u):
                new_set = current_set[:idx] + [v] + current_set[idx+1:]
                new_sum = current_sum - u + v
                delta = new_sum - current_sum
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_set = new_set
                    current_sum = new_sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = list(current_set)
        
        T *= cooling

    return best_set
# EVOLVE-BLOCK-END
