import time
import math
import random

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    if 0 not in U_set:
        # Invalid per contract, but we can return 0 or handle it.
        # To be safe, we assume the caller ensures 0 is present.
        # If not, we can add it? No, better to return 0.
        return 0.0
        
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
            
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _generate_powers_of_b(b, n, include_zero=True):
    U = []
    if include_zero:
        U.append(0)
    for i in range(n):
        val = int(b ** i)
        if val not in U:
            U.append(val)
    U.sort()
    return U

def _simulated_annealing(initial_U, deadline, T0=10.0, cooling=0.99):
    current_U = initial_U[:]
    current_score = _score(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    T = T0
    while time.time() < deadline and T > 1e-5:
        # Generate a neighbor
        move = random.random()
        candidate_U = current_U[:]
        
        if move < 0.4:
            # Replace
            if len(candidate_U) > 1:
                idx = random.randint(1, len(candidate_U) - 1)
                max_val = max(candidate_U)
                # Sample from a wide range
                new_val = random.randint(1, max_val * 2 + 100)
                if new_val not in candidate_U:
                    candidate_U[idx] = new_val
                    candidate_U.sort()
        elif move < 0.7:
            # Add
            max_val = max(candidate_U)
            new_val = random.randint(1, max_val * 2 + 100)
            if new_val not in candidate_U:
                candidate_U.append(new_val)
                candidate_U.sort()
        else:
            # Remove
            if len(candidate_U) > 2:
                idx = random.randint(1, len(candidate_U) - 1)
                candidate_U.pop(idx)
                candidate_U.sort()
        
        if candidate_U == current_U:
            continue
            
        candidate_score = _score(candidate_U)
        
        # Acceptance
        delta = candidate_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_U = candidate_U
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
                
        T *= cooling
        
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0]
    best_score = 0.0
    
    # Phase 1: Structured Seeds
    seeds = []
    # Powers of 2
    for n in range(5, 25):
        seeds.append(_generate_powers_of_b(2, n))
    # Powers of 3
    for n in range(5, 20):
        seeds.append(_generate_powers_of_b(3, n))
    # Powers of 1.5 (approx)
    for n in range(5, 30):
        seeds.append(_generate_powers_of_b(1.5, n))
    # Random sets
    for _ in range(10):
        size = random.randint(5, 20)
        max_val = random.randint(50, 500)
        U = set([0])
        while len(U) < size:
            U.add(random.randint(1, max_val))
        seeds.append(sorted(list(U)))
        
    for U in seeds:
        if time.time() > deadline:
            break
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # Phase 2: Simulated Annealing from best seed
    if time.time() < deadline - 10:
        sa_set, sa_score = _simulated_annealing(best_set, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
            
    return best_set
