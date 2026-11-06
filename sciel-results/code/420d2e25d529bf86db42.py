import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using numpy."""
    A_arr = np.array(sorted(A), dtype=np.int64)
    n = len(A_arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = A_arr[i_idx] + A_arr[j_idx]
    products = A_arr[i_idx] * A_arr[j_idx]
    return max(len(np.unique(sums)), len(np.unique(products)))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _smooth_numbers(limit, count):
    """Generate the 'count' smallest smooth numbers <= limit (of form 2^i * 3^j * 5^k)."""
    smooths = set()
    i_max = int(np.log2(limit)) + 1
    j_max = int(np.log3(limit)) + 1 if limit > 1 else 1
    k_max = int(np.log5(limit)) + 1 if limit > 1 else 1
    
    for i in range(i_max + 1):
        pow2 = 1 << i
        if pow2 > limit:
            break
        for j in range(j_max + 1):
            pow3 = pow2 * (3 ** j)
            if pow3 > limit:
                break
            for k in range(k_max + 1):
                val = pow3 * (5 ** k)
                if val > limit:
                    break
                smooths.add(val)
    
    sorted_smooths = sorted(smooths)
    return sorted_smooths[:count]

def _generate_candidates():
    """Generate diverse candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (primary - best structure)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
    
    # 2. Geometric progressions with various ratios
    for ratio in [1.3, 1.5, 1.8, 2.0, 2.2]:
        for start in [1, 2, 3, 5]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > 10000:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
    
    # 3. Log-spaced (geometric in log space)
    for start in [1, 2]:
        A = [int(round(start * (10000 / start) ** (i / 19))) for i in range(20)]
        # Ensure distinctness
        seen = set()
        unique_A = []
        for x in A:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                unique_A.append(x)
        if len(unique_A) == 20 and _is_valid(unique_A):
            candidates.append(unique_A)
    
    return candidates

def _local_search(A, time_limit, rng, phase=1):
    """Multi-phase local search with simulated annealing."""
    A = list(sorted(A))
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Phase-dependent parameters
    if phase == 1:  # Coarse search
        temperature = 40.0
        cooling_rate = 0.995
        min_temp = 0.1
        perturbation_range = (5, 20)
        large_jump_prob = 0.3
    elif phase == 2:  # Medium search
        temperature = 15.0
        cooling_rate = 0.997
        min_temp = 0.05
        perturbation_range = (3, 10)
        large_jump_prob = 0.2
    else:  # Fine search
        temperature = 5.0
        cooling_rate = 0.999
        min_temp = 0.01
        perturbation_range = (1, 5)
        large_jump_prob = 0.1
    
    while temperature > min_temp:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        if rng.random() < large_jump_prob:
            # Large jump: change one element significantly
            idx = rng.randint(0, 19)
            old_val = A[idx]
            new_val = rng.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000 or new_val in A:
                temperature *= cooling_rate
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
        else:
            # Small perturbation: change one element slightly
            idx = rng.randint(0, 19)
            old_val = A[idx]
            min_pert, max_pert = perturbation_range
            
            for _ in range(10):  # Try multiple small perturbations
                if time.time() - start_time > time_limit:
                    break
                
                delta_val = rng.randint(-max_pert, max_pert)
                new_val = old_val + delta_val
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A:
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
                    break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _double_move_search(A, time_limit, rng):
    """Local search with double moves (change two elements at once)."""
    A = list(sorted(A))
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 20.0
    cooling_rate = 0.996
    min_temp = 0.05
    
    while temperature > min_temp:
        if time.time() - start_time > time_limit:
            break
        
        idx1 = rng.randint(0, 19)
        idx2 = rng.randint(0, 19)
        while idx2 == idx1:
            idx2 = rng.randint(0, 19)
        
        old_val1 = A[idx1]
        old_val2 = A[idx2]
        
        for _ in range(8):
            if time.time() - start_time > time_limit:
                break
            
            if rng.random() < 0.5:
                new_val1 = old_val1 + rng.randint(-10, 10)
            else:
                new_val1 = rng.randint(1, 10000)
            
            if rng.random() < 0.5:
                new_val2 = old_val2 + rng.randint(-10, 10)
            else:
                new_val2 = rng.randint(1, 10000)
            
            if new_val1 < 1 or new_val1 > 10000 or new_val2 < 1 or new_val2 > 10000:
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
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _polish(A, time_limit, rng):
    """Final high-precision polish with exhaustive local moves."""
    A = list(sorted(A))
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Try all single-position moves with very small perturbations
    while time.time() - start_time < time_limit:
        improved = False
        
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            
            for delta in range(-8, 9):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 3
    
    # Generate candidates and pick the best starting point
    candidates = _generate_candidates()
    
    best_start = None
    best_start_obj = float('inf')
    
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        if obj < best_start_obj:
            best_start_obj = obj
            best_start = A
    
    if best_start is None:
        best_start = list(range(1, 21))
        best_start_obj = _objective(best_start)
    
    rng = random.Random(42)
    
    # Phase 1: Coarse local search
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        phase1_time = min(remaining * 0.4, 30)
        A, obj = _local_search(best_start, phase1_time, rng, phase=1)
        if obj < best_start_obj:
            best_start = A
            best_start_obj = obj
    
    # Phase 2: Medium local search
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        phase2_time = min(remaining * 0.3, 20)
        A, obj = _local_search(best_start, phase2_time, rng, phase=2)
        if obj < best_start_obj:
            best_start = A
            best_start_obj = obj
    
    # Phase 3: Double move search
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        phase3_time = min(remaining * 0.25, 15)
        A, obj = _double_move_search(best_start, phase3_time, rng)
        if obj < best_start_obj:
            best_start = A
            best_start_obj = obj
    
    # Phase 4: Fine local search
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 5:
        phase4_time = min(remaining * 0.2, 10)
        A, obj = _local_search(best_start, phase4_time, rng, phase=3)
        if obj < best_start_obj:
            best_start = A
            best_start_obj = obj
    
    # Final polish
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 2:
        polish_time = remaining
        A, obj = _polish(best_start, polish_time, rng)
        if obj < best_start_obj:
            best_start = A
            best_start_obj = obj
    
    if not _is_valid(best_start):
        best_start = list(range(1, 21))
    
    return sorted(best_start)
