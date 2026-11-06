import random
import time
import numpy as np

def _compute_counts(A):
    """
    Compute representation counts for all targets in [128, 256].
    Returns a numpy array of counts.
    """
    A_set = set(A)
    counts = np.zeros(129, dtype=int) # indices 0..128 correspond to targets 128..256
    for target in range(128, 257):
        count = 0
        # Optimization: iterate over smaller set or use set intersection logic?
        # For |A| ~ 30-40, simple loop is fine.
        for a in A:
            b = target - a
            if a <= b and b in A_set:
                count += 1
        counts[target - 128] = count
    return counts

def _is_valid(counts):
    """Check if all counts are > 0."""
    return np.all(counts > 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Greedy Construction
    # We want a set A such that every t in [128, 256] is a sum of two elements in A.
    # Greedy: Add element that covers most uncovered targets with least "cost" (peak increase).
    
    A = []
    A_set = set()
    counts = np.zeros(129, dtype=int)
    uncovered = set(range(128, 257))
    
    # Precompute potential contributions? 
    # For each candidate x in [0, 128], which targets does it help cover?
    # It helps cover t if t-x is in A (or x itself if 2x=t).
    # Since A changes, we compute on the fly.
    
    max_greedy_steps = 50
    for _ in range(max_greedy_steps):
        if not uncovered:
            break
        if time.time() > deadline - 30: # Leave time for SA
            break
            
        best_candidate = None
        best_score = -1
        best_new_counts = None
        
        # Sample candidates to speed up greedy? Or iterate all 129?
        # 129 candidates * 129 targets * |A| ops. 
        # If |A|=20, 129*129*20 ~ 300k ops. Fast.
        
        for x in range(129):
            if x in A_set:
                continue
                
            # Calculate new counts if we add x
            # We only need to check targets where x could be part of a pair.
            # Target t is covered by x if t-x is in A_set or t-x == x (and x in A? no, x is new).
            # Actually, if we add x, new pairs are (x, y) where y in A.
            # So target t = x + y is newly covered if t-x == y and y in A.
            # Also t = 2x is covered if x is added (pair x,x).
            
            # Let's just recompute counts for simplicity and correctness in greedy step, 
            # but only for the delta.
            
            # Delta counts:
            # For each y in A:
            #   t = x + y. If t in [128, 256], counts[t-128] += 1
            #   (Note: if y > x, we might double count if we iterate all y? 
            #    No, we are adding x. The pairs involving x are (x, y) for all y in A.
            #    Since pairs are unordered, (x, y) is the same as (y, x).
            #    We just need to increment count for t=x+y for each y in A.
            #    And t=2x for the pair (x,x).
            #    Wait, if y=x is in A, then (x,x) is counted twice? 
            #    No, y iterates over A. If x is not in A yet, y!=x.
            #    So pairs are (x, y) for y in A. Plus (x, x) itself.
            #    So for each y in A, t = x+y.
            #    Plus t = 2x.
            
            delta = np.zeros(129, dtype=int)
            for y in A:
                t = x + y
                if 128 <= t <= 256:
                    delta[t - 128] += 1
            t_2x = 2 * x
            if 128 <= t_2x <= 256:
                delta[t_2x - 128] += 1
            
            new_counts = counts + delta
            
            # Score: Number of newly covered targets (0 -> 1)
            newly_covered = 0
            for i in range(129):
                if counts[i] == 0 and new_counts[i] > 0:
                    newly_covered += 1
            
            if newly_covered == 0:
                continue
                
            # Tie-breaker: Lower max count is better
            max_new = np.max(new_counts)
            # We want to maximize newly_covered, then minimize max_new
            score = newly_covered * 1000 - max_new
            
            if score > best_score:
                best_score = score
                best_candidate = x
                best_new_counts = new_counts
        
        if best_candidate is None:
            # No candidate improves coverage? Should not happen if valid solution exists.
            # Break or force add random.
            break
            
        A.append(best_candidate)
        A_set.add(best_candidate)
        counts = best_new_counts
        # Update uncovered set (optional, we check counts directly)
        
    if not _is_valid(counts):
        # Greedy failed to cover all. 
        # Fallback: Add random elements until valid.
        while not _is_valid(counts) and len(A) < 129:
            x = random.randint(0, 128)
            if x in A_set:
                continue
            A.append(x)
            A_set.add(x)
            # Recompute counts (slow but rare)
            counts = _compute_counts(A)
            
    if not _is_valid(counts):
        # Absolute fallback
        A = list(range(64, 129))
        counts = _compute_counts(A)
        
    current_peak = int(np.max(counts))
    best_peak = current_peak
    best_A = list(A)
    
    # 2. Simulated Annealing
    # Objective: Minimize sum of squares (flattens distribution) or Peak.
    # Using sum of squares helps avoid local minima of peak-only.
    # Constraint: Validity.
    
    T = 10.0
    cooling_rate = 0.95
    min_T = 0.01
    
    # Precompute A_set for fast lookup
    A_set = set(A)
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        # Generate a move
        move_type = random.random()
        
        if len(A) > 15 and move_type < 0.2:
            # Try to remove an element
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            A_set.remove(old_elem)
            A.pop(idx)
            
            new_counts = _compute_counts(A)
            
            if _is_valid(new_counts):
                new_peak = int(np.max(new_counts))
                new_obj = int(np.sum(new_counts**2))
                old_obj = int(np.sum(counts**2))
                
                # Accept if better or with probability exp(-delta/T)
                delta = new_obj - old_obj
                if delta < 0 or random.random() < np.exp(-delta / T):
                    counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                else:
                    # Reject
                    A.append(old_elem)
                    A_set.add(old_elem)
            else:
                # Invalid, reject
                A.append(old_elem)
                A_set.add(old_elem)
                
        elif move_type < 0.4 and len(A) < 80:
            # Try to add an element
            x = random.randint(0, 128)
            if x in A_set:
                continue
            A.append(x)
            A_set.add(x)
            
            new_counts = _compute_counts(A)
            
            # Adding an element never breaks validity (counts only increase).
            # So it's always valid.
            new_peak = int(np.max(new_counts))
            new_obj = int(np.sum(new_counts**2))
            old_obj = int(np.sum(counts**2))
            
            delta = new_obj - old_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                counts = new_counts
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(A)
            else:
                # Reject
                A.pop()
                A_set.remove(x)
                
        else:
            # Swap
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            
            new_elem = random.randint(0, 128)
            if new_elem in A_set and new_elem != old_elem:
                # If new_elem is already in A, swapping with a duplicate is not a change 
                # unless we remove old and add new, but new is already there.
                # Actually, if new_elem == old_elem, no change.
                # If new_elem in A_set and new_elem != old_elem, 
                # then we are replacing old_elem with a value that is already in the set.
                # This effectively removes old_elem and keeps the other instance of new_elem.
                # Result: Set size decreases by 1? No, list length stays same, but set loses old_elem.
                # This is a "remove old, keep new (already present)" move.
                # Equivalent to removing old_elem.
                # Let's just skip if new_elem is in A_set to keep moves simple (true swap with new value).
                continue
                
            # Perform swap
            A[idx] = new_elem
            A_set.remove(old_elem)
            A_set.add(new_elem)
            
            new_counts = _compute_counts(A)
            
            if _is_valid(new_counts):
                new_peak = int(np.max(new_counts))
                new_obj = int(np.sum(new_counts**2))
                old_obj = int(np.sum(counts**2))
                
                delta = new_obj - old_obj
                if delta < 0 or random.random() < np.exp(-delta / T):
                    counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                else:
                    # Reject
                    A[idx] = old_elem
                    A_set.remove(new_elem)
                    A_set.add(old_elem)
            else:
                # Invalid, reject
                A[idx] = old_elem
                A_set.remove(new_elem)
                A_set.add(old_elem)
                
        T *= cooling_rate
        
    return best_A
