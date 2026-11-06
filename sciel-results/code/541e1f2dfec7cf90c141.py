import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Generate all pairs (i, j) with i <= j
    # There are n*(n+1)/2 such pairs
    indices = np.triu_indices(n)
    
    # Vectorized sum and product calculation
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
    # Count unique values
    # np.unique is efficient for sorting and finding unique elements
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _generate_two_cluster_gp(start1, ratio1_num, ratio1_den, start2, ratio2_num, ratio2_den, n1, n2):
    """
    Generate a set that is the union of two geometric progressions.
    This maximizes multiplicative collisions via unique prime factorization.
    """
    A = set()
    
    # First GP
    current = float(start1)
    ratio1 = float(ratio1_num) / float(ratio1_den)
    for _ in range(n1):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        A.add(val)
        current *= ratio1
    
    # Second GP
    current = float(start2)
    ratio2 = float(ratio2_num) / float(ratio2_den)
    for _ in range(n2):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in A:
            A.add(val)
        current *= ratio2
    
    # Convert to sorted list
    A = sorted(A)
    
    # If we don't have enough, fill with small increments
    if len(A) < n1 + n2:
        last = A[-1] if A else max(start1, start2)
        while len(A) < n1 + n2:
            last += 1
            if last > MAX_VAL:
                break
            if last not in A:
                A.append(last)
    
    return A[:n1 + n2]

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _generate_rational_gp(start, num, den, n):
    """
    Generate n distinct integers from a rational geometric progression:
    a_k = start * (num/den)^k
    This helps create multiplicative collisions.
    """
    A = []
    current = float(start)
    ratio = float(num) / float(den)
    
    for _ in range(n):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in A:
            A.append(val)
        current *= ratio
        
    # If we didn't get enough, fill with small increments
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL: break
            if last not in A:
                A.append(last)
            
    return A[:n]

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    
    while temperature > 0.1:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        accepted = False
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Strategy: Mix of small perturbations and larger jumps
            if rng.random() < 0.5:
                # Small perturbation
                scale = max(1, old_val // 10)
                new_val = old_val + rng.randint(-scale, scale)
            else:
                # Larger jump or swap-like move
                # Try replacing with a value from a different range
                if rng.random() < 0.5:
                    new_val = rng.randint(1, MAX_VAL)
                else:
                    # Try to move towards the median or extremes
                    new_val = old_val + rng.randint(-max(1, old_val//2), max(1, old_val//2))
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                accepted = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _cluster_aware_local_search(A, time_limit, rng, cluster_split=10):
    """
    Local search that is aware of two-cluster structure.
    Encourages swaps between clusters to balance sumset and product-set sizes.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.99
    
    # Identify cluster boundaries (approximate)
    mid = A[cluster_split - 1] if cluster_split < len(A) else A[len(A)//2]
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Single element perturbation within its cluster
            if rng.random() < 0.5:
                idx = rng.randint(0, cluster_split - 1)
            else:
                idx = rng.randint(cluster_split, len(A) - 1)
            
            old_val = A[idx]
            scale = max(1, old_val // 5)
            new_val = old_val + rng.randint(-scale, scale)
            
            if new_val < 1 or new_val > MAX_VAL or new_val in A:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                    
        elif move_type < 0.7:
            # Swap between clusters
            idx1 = rng.randint(0, cluster_split - 1)
            idx2 = rng.randint(cluster_split, len(A) - 1)
            
            if idx1 == idx2:
                continue
            
            old_val1 = A[idx1]
            old_val2 = A[idx2]
            
            # Try to find good replacement values
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                
                # Scale factor for new values
                scale1 = max(1, old_val1 // 5)
                scale2 = max(1, old_val2 // 5)
                
                new_val1 = old_val1 + rng.randint(-scale1, scale1)
                new_val2 = old_val2 + rng.randint(-scale2, scale2)
                
                if new_val1 < 1 or new_val1 > MAX_VAL:
                    continue
                if new_val2 < 1 or new_val2 > MAX_VAL:
                    continue
                if new_val1 in A or new_val2 in A:
                    continue
                if new_val1 == new_val2:
                    continue
                
                A[idx1] = new_val1
                A[idx2] = new_val2
                new_obj = _objective(A)
                A[idx1] = old_val1
                A[idx2] = old_val2
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    A[idx1] = new_val1
                    A[idx2] = new_val2
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        
        else:
            # Larger jump: replace an element with a value from the other cluster's range
            idx = rng.randint(0, len(A) - 1)
            old_val = A[idx]
            
            if idx < cluster_split:
                # Move towards upper cluster
                new_val = rng.randint(mid, min(MAX_VAL, mid * 3))
            else:
                # Move towards lower cluster
                new_val = rng.randint(1, max(1, mid // 3))
            
            if new_val < 1 or new_val > MAX_VAL or new_val in A:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    starts = []
    
    # Start 1: Arithmetic Progression
    starts.append(list(range(1, 21)))
    
    # Start 2: Rational GP (1, 2/1) -> Powers of 2
    starts.append(_generate_rational_gp(1, 2, 1, 20))
    
    # Start 3: Rational GP (1, 3/2) -> 1, 1.5, 2.25... -> 1, 2, 3, 5, 8... (Fibonacci-like)
    starts.append(_generate_rational_gp(1, 3, 2, 20))
    
    # Start 4: Rational GP (1, 5/4)
    starts.append(_generate_rational_gp(1, 5, 4, 20))
    
    # Start 5: Random
    starts.append(sorted(rng.sample(range(1, MAX_VAL + 1), 20)))
    
    # Start 6: Mixed (Small AP + Large GP)
    small_ap = list(range(1, 11))
    large_gp = _generate_rational_gp(50, 2, 1, 10)
    # Ensure distinctness
    combined = small_ap + [x for x in large_gp if x not in small_ap]
    if len(combined) >= 20:
        starts.append(sorted(combined[:20]))
    else:
        # Fill gaps
        while len(combined) < 20:
            combined.append(combined[-1] + 1)
        starts.append(sorted(combined))
    
    # Start 7: Two-cluster GP (powers of 2 + powers of 3)
    starts.append(_generate_two_cluster_gp(1, 2, 1, 1, 3, 1, 10, 10))
    
    # Start 8: Two-cluster GP (powers of 2 + powers of 4)
    starts.append(_generate_two_cluster_gp(1, 2, 1, 1, 4, 1, 10, 10))
    
    # Start 9: Two-cluster GP (powers of 3 + powers of 5)
    starts.append(_generate_two_cluster_gp(1, 3, 1, 1, 5, 1, 10, 10))
    
    # Start 10: Two-cluster GP (small base + medium base)
    starts.append(_generate_two_cluster_gp(1, 2, 1, 10, 3, 1, 10, 10))
    
    # Start 11: Two-cluster GP with offset
    starts.append(_generate_two_cluster_gp(1, 3, 2, 1, 5, 3, 10, 10))
    
    # Start 12: Two-cluster GP (powers of 2 + shifted powers of 3)
    starts.append(_generate_two_cluster_gp(1, 2, 1, 7, 3, 1, 10, 10))

    # Run local search on each start
    for i, A in enumerate(starts):
        if not _is_valid(A):
            # Fix if invalid
            A = sorted(set(A))
            if len(A) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in A]
                A += rng.sample(available, 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
            A = sorted(A)
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 1:
            continue
        
        # Allocate time proportional to remaining budget
        # Give first few starts more time
        alloc = max(0.5, remaining_time / (len(starts) - i))
        
        # Use cluster-aware search for two-cluster starts
        if i >= 6:  # Two-cluster starts
            A_opt, obj_opt = _cluster_aware_local_search(A, alloc, rng, cluster_split=10)
        else:
            A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found using both search methods
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.5 and best_A:
        # Try cluster-aware search
        A_ref1, obj_ref1 = _cluster_aware_local_search(best_A, remaining * 0.6, rng, cluster_split=10)
        if obj_ref1 < best_obj:
            best_obj = obj_ref1
            best_A = A_ref1
        
        remaining2 = time_budget - (time.time() - start_time)
        if remaining2 > 0.3:
            # Try standard search
            A_ref2, obj_ref2 = _local_search(best_A, remaining2, rng)
            if obj_ref2 < best_obj:
                best_obj = obj_ref2
                best_A = A_ref2
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
