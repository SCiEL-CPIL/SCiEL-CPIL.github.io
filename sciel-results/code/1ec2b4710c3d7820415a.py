import random
import time
import numpy as np

def _compute_counts_fast(A):
    """Compute representation counts for targets [128,256] using pair enumeration."""
    counts = np.zeros(129, dtype=np.int32)
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if 128 <= s <= 256:
                counts[s - 128] += 1
    return counts

def _compute_delta_counts(A, remove_elem, add_elem=None):
    """Compute new counts after removing one element and optionally adding another."""
    A_set = set(A)
    counts = np.zeros(129, dtype=np.int32)
    
    # Build list without the removed element
    B = [x for x in A if x != remove_elem]
    if add_elem is not None:
        B.append(add_elem)
    
    n = len(B)
    for i in range(n):
        ai = B[i]
        for j in range(i, n):
            s = ai + B[j]
            if 128 <= s <= 256:
                counts[s - 128] += 1
    return counts

def _is_valid(counts):
    return np.all(counts > 0)

def _get_objective(counts):
    """Variance-aware objective: sum of squares + large penalty for invalid."""
    if not _is_valid(counts):
        return 10**9
    return int(np.sum(counts.astype(np.int64) ** 2))

def _get_peak(counts):
    return int(np.max(counts))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Phase 1: Greedy construction with balance awareness
    A = []
    A_set = set()
    counts = np.zeros(129, dtype=np.int32)
    
    # Start with endpoints to ensure coverage of extremes
    A = [0, 128]
    A_set = {0, 128}
    counts = _compute_counts_fast(A)
    
    max_greedy_steps = 60
    for step in range(max_greedy_steps):
        if time.time() > deadline - 30:
            break
        if _is_valid(counts):
            break
            
        best_candidate = None
        best_score = -10**12
        best_new_counts = None
        
        for x in range(129):
            if x in A_set:
                continue
                
            # Compute delta: adding x creates pairs (x, y) for y in A, plus (x, x)
            delta = np.zeros(129, dtype=np.int32)
            for y in A:
                t = x + y
                if 128 <= t <= 256:
                    delta[t - 128] += 1
            t_2x = 2 * x
            if 128 <= t_2x <= 256:
                delta[t_2x - 128] += 1
            
            new_counts = counts + delta
            
            # Score: prioritize covering uncovered targets, then balance
            newly_covered = np.sum((counts == 0) & (new_counts > 0))
            if newly_covered == 0:
                continue
                
            # Objective: minimize sum of squares (variance-aware)
            new_obj = int(np.sum(new_counts.astype(np.int64) ** 2))
            old_obj = int(np.sum(counts.astype(np.int64) ** 2))
            
            # Heavily weight newly covered targets
            score = newly_covered * 10000 - (new_obj - old_obj)
            
            if score > best_score:
                best_score = score
                best_candidate = x
                best_new_counts = new_counts
        
        if best_candidate is None:
            break
            
        A.append(best_candidate)
        A_set.add(best_candidate)
        counts = best_new_counts
    
    # Fallback if greedy failed
    if not _is_valid(counts):
        while not _is_valid(counts) and len(A) < 129:
            x = random.randint(0, 128)
            if x in A_set:
                continue
            A.append(x)
            A_set.add(x)
            counts = _compute_counts_fast(A)
    
    if not _is_valid(counts):
        A = list(range(64, 129))
        counts = _compute_counts_fast(A)
    
    best_peak = _get_peak(counts)
    best_obj = _get_objective(counts)
    best_A = list(A)
    
    # Phase 2: Aggressive local search with multiple move types
    T = 50.0
    cooling_rate = 0.97
    min_T = 0.1
    no_improve_steps = 0
    max_no_improve = 500
    
    A_set = set(A)
    
    while time.time() < deadline - 3:
        if T < min_T:
            T = min_T * 2  # Restart with higher temperature
            no_improve_steps = 0
            
        move_type = random.random()
        
        if len(A) > 20 and move_type < 0.15:
            # Remove an element
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            A.pop(idx)
            A_set.remove(old_elem)
            
            new_counts = _compute_counts_fast(A)
            
            if _is_valid(new_counts):
                new_obj = _get_objective(new_counts)
                if new_obj < best_obj or random.random() < np.exp(-(new_obj - best_obj) / T):
                    counts = new_counts
                    new_peak = _get_peak(new_counts)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_peak = new_peak
                        best_A = list(A)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    A.insert(idx, old_elem)
                    A_set.add(old_elem)
            else:
                A.insert(idx, old_elem)
                A_set.add(old_elem)
                
        elif move_type < 0.25 and len(A) < 80:
            # Add an element
            x = random.randint(0, 128)
            if x in A_set:
                continue
            A.append(x)
            A_set.add(x)
            
            new_counts = _compute_counts_fast(A)
            new_obj = _get_objective(new_counts)
            
            if new_obj < best_obj or random.random() < np.exp(-(new_obj - best_obj) / T):
                counts = new_counts
                new_peak = _get_peak(new_counts)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_peak = new_peak
                    best_A = list(A)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                A.pop()
                A_set.remove(x)
                
        else:
            # Swap: remove one, add another
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            
            new_elem = random.randint(0, 128)
            if new_elem == old_elem:
                continue
            if new_elem in A_set:
                # This is effectively a removal
                A.pop(idx)
                A_set.remove(old_elem)
                new_counts = _compute_counts_fast(A)
                
                if _is_valid(new_counts):
                    new_obj = _get_objective(new_counts)
                    if new_obj < best_obj or random.random() < np.exp(-(new_obj - best_obj) / T):
                        counts = new_counts
                        new_peak = _get_peak(new_counts)
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_peak = new_peak
                            best_A = list(A)
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1
                    else:
                        A.insert(idx, old_elem)
                        A_set.add(old_elem)
                else:
                    A.insert(idx, old_elem)
                    A_set.add(old_elem)
                continue
            
            # True swap
            A[idx] = new_elem
            A_set.remove(old_elem)
            A_set.add(new_elem)
            
            new_counts = _compute_counts_fast(A)
            
            if _is_valid(new_counts):
                new_obj = _get_objective(new_counts)
                if new_obj < best_obj or random.random() < np.exp(-(new_obj - best_obj) / T):
                    counts = new_counts
                    new_peak = _get_peak(new_counts)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_peak = new_peak
                        best_A = list(A)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    A[idx] = old_elem
                    A_set.remove(new_elem)
                    A_set.add(old_elem)
            else:
                A[idx] = old_elem
                A_set.remove(new_elem)
                A_set.add(old_elem)
        
        T *= cooling_rate
        
        if no_improve_steps > max_no_improve:
            # Reset to best and restart
            A = list(best_A)
            A_set = set(A)
            counts = _compute_counts_fast(A)
            T = 50.0
            no_improve_steps = 0
    
    # Final: ensure validity
    if not _is_valid(_compute_counts_fast(best_A)):
        # Repair: add elements until valid
        A = list(best_A)
        A_set = set(A)
        counts = _compute_counts_fast(A)
        while not _is_valid(counts) and len(A) < 129:
            x = random.randint(0, 128)
            if x in A_set:
                continue
            A.append(x)
            A_set.add(x)
            counts = _compute_counts_fast(A)
        best_A = A
    
    return best_A
