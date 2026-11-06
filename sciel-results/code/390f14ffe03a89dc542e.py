"""Integer sum-product extremal sets - minimize max(|A+A|, |A*A|)"""
import time
import random
import math
from itertools import combinations

def _sumset_size(A):
    """Compute |A+A| with equal summands allowed"""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    """Compute |A*A| with equal factors allowed"""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] * A[j])
    return len(s)

def _objective(A):
    """Return max(|A+A|, |A*A|)"""
    ss = _sumset_size(A)
    ps = _prodset_size(A)
    return max(ss, ps)

def _is_valid(A, n=20, maximum=10000):
    """Check if A is valid: n distinct positive integers <= maximum"""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x < 1 or x > maximum for x in A):
        return False
    return True

def _geometric_start(ratio=1.5, n=20, max_val=10000):
    """Generate a geometric-like starting set"""
    A = []
    val = 2
    for i in range(n):
        A.append(int(val))
        val *= ratio
    # Ensure distinct and within bounds
    A = sorted(set(A))
    while len(A) < n:
        # Fill in gaps
        candidates = []
        for x in range(1, min(10001, int(max(A) * 1.5) + 1)):
            if x not in A:
                candidates.append(x)
        if not candidates:
            break
        A.append(candidates[0])
    A = sorted(set(A))[:n]
    while len(A) < n:
        A.append(max(A) + 1)
    return A[:n]

def _random_start(n=20, max_val=10000):
    """Generate a random starting set"""
    # Use a mix of small numbers and geometric spacing
    base = random.randint(2, 5)
    A = set()
    while len(A) < n:
        # Mix of random small and geometric
        if random.random() < 0.5:
            A.add(random.randint(1, max_val // 2))
        else:
            A.add(random.randint(1, max_val))
    return sorted(A)

def _local_search(A, time_limit=55):
    """Local search to minimize max(|A+A|, |A*A|)"""
    start_time = time.time()
    best_A = sorted(A[:])
    best_obj = _objective(best_A)
    
    n = len(best_A)
    max_val = 10000
    
    # Try swapping elements
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit:
                    break
                # Try replacing A[j] with a nearby value
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = best_A[j] + delta
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in best_A:
                        continue
                    # Check if new_val equals any other element
                    trial = best_A[:]
                    trial[j] = new_val
                    if len(set(trial)) != n:
                        continue
                    obj = _objective(trial)
                    if obj < best_obj:
                        best_obj = obj
                        best_A = trial
                        improved = True
                        break
    
    # Try replacing single elements with better values
    for _ in range(50):
        if (time.time() - start_time) > time_limit:
            break
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            # Try a few candidate replacements
            for _ in range(20):
                new_val = random.randint(1, max_val)
                if new_val in best_A:
                    continue
                trial = best_A[:]
                trial[i] = new_val
                if len(set(trial)) != n:
                    continue
                obj = _objective(trial)
                if obj < best_obj:
                    best_obj = obj
                    best_A = trial
                    break
    
    return best_A, best_obj

def _simulated_annealing(A, time_limit=55):
    """Simulated annealing to escape local minima"""
    start_time = time.time()
    current_A = sorted(A[:])
    current_obj = _objective(current_A)
    best_A = current_A[:]
    best_obj = current_obj
    
    n = len(current_A)
    max_val = 10000
    T = 10.0
    cooling = 0.999
    
    while (time.time() - start_time) < time_limit and T > 0.1:
        # Pick a random move
        i = random.randint(0, n - 1)
        if random.random() < 0.7:
            # Small perturbation
            delta = random.randint(-5, 5)
            new_val = current_A[i] + delta
        else:
            # Larger perturbation
            new_val = random.randint(1, max_val)
        
        if new_val < 1 or new_val > max_val:
            T *= cooling
            continue
        if new_val in current_A:
            T *= cooling
            continue
            
        trial = current_A[:]
        trial[i] = new_val
        if len(set(trial)) != n:
            T *= cooling
            continue
            
        trial_obj = _objective(trial)
        delta_obj = trial_obj - current_obj
        
        if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
            current_A = trial
            current_obj = trial_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = current_A[:]
        
        T *= cooling
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 90s kill
    
    best_A = None
    best_obj = float('inf')
    
    # Multiple starting points
    starts = []
    
    # Geometric starts with different ratios
    for ratio in [1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        A = _geometric_start(ratio=ratio, n=20, max_val=10000)
        if _is_valid(A):
            starts.append(A)
    
    # Random starts
    for _ in range(10):
        A = _random_start(n=20, max_val=10000)
        if _is_valid(A):
            starts.append(A)
    
    # Consecutive integers as baseline
    starts.append(list(range(1, 21)))
    starts.append(list(range(2, 22)))
    
    for start in starts:
        if (time.time() - start_time) > deadline:
            break
        
        # Local search
        A_ls, obj_ls = _local_search(start, time_limit=min(55, deadline - (time.time() - start_time)))
        if obj_ls < best_obj:
            best_obj = obj_ls
            best_A = A_ls[:]
        
        # Simulated annealing from the locally optimized result
        if (time.time() - start_time) < deadline - 5:
            A_sa, obj_sa = _simulated_annealing(A_ls, time_limit=min(20, deadline - (time.time() - start_time)))
            if obj_sa < best_obj:
                best_obj = obj_sa
                best_A = A_sa[:]
    
    # Final polish: try more local search on the best
    if best_A is not None and (time.time() - start_time) < deadline - 2:
        A_final, obj_final = _local_search(best_A, time_limit=min(10, deadline - (time.time() - start_time)))
        if obj_final < best_obj:
            best_obj = obj_final
            best_A = A_final[:]
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return sorted(best_A)
