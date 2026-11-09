import time
import sys

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

import random

def get_subset_sum_bitsets(A):
    """
    Returns a list of bitsets S where S[j] has bit s set if sum s is achievable with cardinality j.
    Returns None if invalid.
    """
    n = len(A)
    if n == 0:
        return [1]
    S = [0] * (n + 1)
    S[0] = 1
    
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            new_part = S[j] << x
            # Check collision with other cardinalities
            for k in range(n + 1):
                if k == j + 1:
                    continue
                if new_part & S[k]:
                    return None
            S[j+1] |= new_part
            
    # Final check
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return None
    return S

def local_search(initial_size, max_time=50.0):
    """
    Uses Simulated Annealing to maximize set size.
    """
    start_time = time.time()
    all_elements = list(range(1, 257))
    
    # Initialize with a random valid set of size initial_size
    # Greedy initialization: pick random elements, check validity
    current_set = []
    random.shuffle(all_elements)
    for x in all_elements:
        if len(current_set) >= initial_size:
            break
        test_set = current_set + [x]
        if get_subset_sum_bitsets(test_set) is not None:
            current_set.append(x)
            
    if not current_set:
        return None
        
    best_set = current_set[:]
    best_size = len(current_set)
    
    temperature = 10.0
    cooling_rate = 0.95
    
    while elapsed() < DEADLINE and (time.time() - start_time) < max_time:
        if is_done():
            break
            
        # Generate a move
        move_type = random.random()
        new_set = current_set[:]
        
        if move_type < 0.4: # Add
            candidates = [x for x in all_elements if x not in current_set]
            if candidates:
                x = random.choice(candidates)
                new_set.append(x)
        elif move_type < 0.8: # Remove
            if current_set:
                idx = random.randint(0, len(current_set) - 1)
                new_set.pop(idx)
        else: # Swap
            if current_set:
                idx = random.randint(0, len(current_set) - 1)
                candidates = [x for x in all_elements if x not in current_set]
                if candidates:
                    x_new = random.choice(candidates)
                    x_old = new_set.pop(idx)
                    new_set.append(x_new)
        
        if new_set == current_set:
            continue
            
        # Evaluate
        S_new = get_subset_sum_bitsets(new_set)
        
        if S_new is None:
            # Invalid move. Reject.
            continue
            
        new_size = len(new_set)
        current_size = len(current_set)
        
        if new_size > current_size:
            # Accept
            current_set = new_set
            if new_size > best_size:
                best_size = new_size
                best_set = new_set[:]
                temperature = 10.0 # Reheat on improvement
        else:
            # new_size <= current_size
            # Accept with probability exp(-delta / T)
            # Delta = current_size - new_size (positive if we lose size)
            delta = current_size - new_size
            if delta == 0:
                # Neutral move, accept with high prob to explore
                if random.random() < 0.5:
                    current_set = new_set
            else:
                # Prob = exp(-delta / T)
                if temperature > 0.01:
                    prob = 2.718281828 ** (-delta / temperature)
                    if random.random() < prob:
                        current_set = new_set
                # else reject
            
        temperature *= cooling_rate
        
    return best_set

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Try to find larger sets using local search
    if elapsed() < DEADLINE - 5:
        res = local_search(12, max_time=DEADLINE - elapsed() - 2)
        if res is not None and len(res) > len(base_solution):
            return res
            
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
