import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin for 90s limit

# Pre-compute indices for A+A and A*A (i <= j)
# 20*21/2 = 210 pairs
INDICES = np.triu_indices(MAX_N)

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of 20 distinct positive integers.
    A is a list or array of 20 ints.
    """
    A_arr = np.array(A, dtype=np.int64)
    # Extract pairs
    a_i = A_arr[INDICES[0]]
    a_j = A_arr[INDICES[1]]
    
    sums = a_i + a_j
    products = a_i * a_j
    
    # Count unique
    # np.unique returns sorted unique values
    n_sums = len(np.unique(sums))
    n_products = len(np.unique(products))
    
    return max(n_sums, n_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != MAX_N:
        return False
    if len(set(A)) != MAX_N:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _generate_ap(start=1, diff=1, n=20):
    A = []
    val = start
    for _ in range(n):
        if val > MAX_VAL:
            break
        A.append(val)
        val += diff
    # If not enough, fill with randoms
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL: break
            A.append(last)
    return A[:n]

def _generate_gp(start=1, ratio=2, n=20):
    A = []
    val = start
    for _ in range(n):
        if val > MAX_VAL:
            break
        A.append(val)
        val *= ratio
    # If not enough, we need to fill.
    # For GP, filling with small increments might break the product structure.
    # Better to just return what we have and let the caller fix it?
    # Or add randoms?
    if len(A) < n:
        last = A[-1] if A else start
        # Add elements that are multiples of last or close to last?
        # For now, just increment
        while len(A) < n:
            last += 1
            if last > MAX_VAL: break
            A.append(last)
    return A[:n]

def _generate_logspace(n=20):
    # Log-uniform distribution
    A = np.logspace(0, np.log10(MAX_VAL), n)
    A = np.round(A).astype(int)
    # Ensure distinctness and bounds
    unique_A = []
    for val in A:
        if 1 <= val <= MAX_VAL and val not in unique_A:
            unique_A.append(val)
    if len(unique_A) < n:
        # Fill gaps
        last = unique_A[-1] if unique_A else 1
        while len(unique_A) < n:
            last += 1
            if last > MAX_VAL: break
            unique_A.append(last)
    return unique_A[:n]

def _local_search(A, time_limit, rng):
    """
    Simulated Annealing / Local Search.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 10.0  # Start with small temp since objective is integer and changes are discrete
    cooling_rate = 0.99
    
    # Pre-compute some helper data? Not really needed for N=20.
    
    while temperature > 0.1:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, MAX_N - 1)
        old_val = A[idx]
        
        # Generate a new value
        # Strategy: Mix of random jumps and structured moves
        if rng.random() < 0.7:
            # Structured move: try to create multiplicative or additive relation
            # Pick another element
            j = rng.randint(0, MAX_N - 1)
            if j == idx:
                j = (j + 1) % MAX_N
            other_val = A[j]
            
            r = rng.random()
            if r < 0.3:
                # Multiplicative: v = k * other_val
                k = rng.randint(2, 5)
                new_val = other_val * k
            elif r < 0.6:
                # Additive: v = other_val + k
                k = rng.randint(-10, 10)
                new_val = other_val + k
            else:
                # Random in range
                new_val = rng.randint(1, MAX_VAL)
        else:
            # Random local jump
            delta = rng.randint(-50, 50)
            new_val = old_val + delta
        
        if new_val < 1 or new_val > MAX_VAL:
            continue
        if new_val in A:
            continue
            
        # Test the change
        A[idx] = new_val
        new_obj = _objective(A)
        
        delta_obj = new_obj - current_obj
        
        # Accept if better or with probability exp(-delta/T)
        if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / (temperature + 1e-10)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
        else:
            # Revert
            A[idx] = old_val
        
        temperature *= cooling_rate
        
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    # Generate diverse starting points
    starts = []
    
    # 1. AP
    starts.append(_generate_ap(1, 1))
    
    # 2. GP (2)
    starts.append(_generate_gp(1, 2))
    
    # 3. GP (3)
    starts.append(_generate_gp(1, 3))
    
    # 4. Logspace
    starts.append(_generate_logspace())
    
    # 5. Random
    starts.append(random.sample(range(1, MAX_VAL + 1), MAX_N))
    
    # 6. Mixed: Small AP + Large GP
    s6 = list(range(1, 11)) + [1000, 2000, 4000, 8000, 8100, 8200, 8300, 8400, 8500]
    # Ensure distinct and valid
    s6 = sorted(set(s6))
    if len(s6) < MAX_N:
        s6 += random.sample([x for x in range(1, MAX_VAL+1) if x not in s6], MAX_N - len(s6))
    starts.append(s6[:MAX_N])
    
    # 7. Another GP variant: Base 4
    starts.append(_generate_gp(1, 4))
    
    # 8. Base 5
    starts.append(_generate_gp(1, 5))
    
    # Filter valid starts
    valid_starts = []
    for s in starts:
        if _is_valid(s):
            valid_starts.append(s)
        else:
            # Fix
            s = sorted(set(s))
            if len(s) < MAX_N:
                available = [x for x in range(1, MAX_VAL+1) if x not in s]
                s += random.sample(available, MAX_N - len(s))
            elif len(s) > MAX_N:
                s = s[:MAX_N]
            if _is_valid(s):
                valid_starts.append(s)
    
    if not valid_starts:
        valid_starts = [list(range(1, MAX_N + 1))]
    
    # Run local search on each start
    # Allocate time evenly
    num_starts = len(valid_starts)
    time_per_start = time_budget / num_starts
    
    for i, A in enumerate(valid_starts):
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 0.5:
            break
        
        # Allocate slightly less than remaining to ensure we finish
        alloc = min(time_per_start, remaining_time)
        
        A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.5 and best_A:
        A_ref, obj_ref = _local_search(best_A, remaining, rng)
        if obj_ref < best_obj:
            best_A = A_ref
            best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, MAX_N + 1))
        
    return sorted(best_A)
