import time
import sys
import random

def _check_validity_bitset(A):
    """Check validity using bitsets for subset sums."""
    n = len(A)
    if n == 0:
        return True
    
    # B[k] is a bitset where bit s is 1 if sum s is achievable by a subset of size k
    B = [0] * (n + 1)
    B[0] = 1  # Sum 0 with 0 elements
    
    for a in A:
        # Update in reverse to avoid using 'a' multiple times in the same step
        for k in range(n - 1, -1, -1):
            if B[k]:
                B[k + 1] |= (B[k] << a)
        
    # Check for conflicts:
    # We need B[k] and B[j] to be disjoint for all k != j.
    # This is equivalent to:
    # 1. Compute Total = OR of all B[k]
    # 2. Compute Sum_Bits = Sum of all B[k] (as integers)
    # 3. If Total != Sum_Bits, there is a carry, meaning some bit was set in multiple B[k].
    
    total = 0
    sum_bits = 0
    for k in range(n + 1):
        total |= B[k]
        sum_bits += B[k]
        
    return total == sum_bits

def _compute_bitsets(A):
    """Compute bitsets B where B[k] has bit s set if sum s achievable with k elements."""
    n = len(A)
    B = [0] * (n + 1)
    B[0] = 1
    for a in A:
        for k in range(n - 1, -1, -1):
            if B[k]:
                B[k + 1] |= (B[k] << a)
    return B

def _is_valid_bitsets(B):
    """Check if bitsets B are valid (no sum appears in multiple cardinality classes)."""
    total = 0
    sum_bits = 0
    for b in B:
        total |= b
        sum_bits += b
    return total == sum_bits

def _backtrack_search(A, B, max_size, deadline, candidates, used_set):
    """
    Backtracking search with forward checking.
    
    A: current list of selected integers
    B: list of bitsets where B[k] has bit s set if sum s achievable with k elements
    max_size: target size to reach
    deadline: absolute time deadline
    candidates: list of candidate integers to try
    used_set: set of already used integers for O(1) lookup
    
    Returns: list of selected integers if success, None if fail
    """
    n = len(A)
    
    # If we've reached max_size, success
    if n >= max_size:
        return A[:]
    
    # Check time
    if time.time() > deadline:
        return None
    
    # Try each candidate
    for cand in candidates:
        if time.time() > deadline:
            return None
        
        # Skip if already in A
        if cand in used_set:
            continue
        
        # Forward check: try adding cand and see if it creates a conflict
        # Compute new B with cand added
        new_B = B[:]  # Copy current bitsets
        # Update in reverse to avoid using cand multiple times
        for k in range(n, -1, -1):
            if new_B[k]:
                new_B[k + 1] |= (new_B[k] << cand)
        
        # Check for conflicts: new_B[k] and new_B[j] must be disjoint for k != j
        # Use the same trick: total OR should equal sum of bits
        total = 0
        sum_bits = 0
        valid = True
        for k in range(n + 2):
            if k < len(new_B):
                total |= new_B[k]
                sum_bits += new_B[k]
        if total != sum_bits:
            continue  # Conflict, skip this candidate
        
        # Recurse
        new_A = A + [cand]
        new_used = used_set | {cand}
        result = _backtrack_search(new_A, new_B, max_size, deadline, candidates, new_used)
        if result is not None:
            return result
    
    return None

def _simulated_annealing(A, deadline, iters_per_step=50):
    """
    Simulated annealing to improve the set.
    Try swapping elements or adding/removing to find larger valid sets.
    """
    n = len(A)
    B = _compute_bitsets(A)
    best_A = A[:]
    best_n = n
    
    current_temp = 100.0
    cooling_rate = 0.95
    
    start_time = time.time()
    
    for step in range(iters_per_step):
        if time.time() > deadline:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4 and len(A) > 0:
            # Remove a random element
            idx = random.randint(0, len(A) - 1)
            new_A = A[:idx] + A[idx+1:]
            new_B = _compute_bitsets(new_A)
            if _is_valid_bitsets(new_B):
                A = new_A
                B = new_B
        elif move_type < 0.8:
            # Add a random element
            candidates = [x for x in range(1, 257) if x not in A]
            if candidates:
                cand = random.choice(candidates)
                new_A = A + [cand]
                new_B = _compute_bitsets(new_A)
                if _is_valid_bitsets(new_B):
                    A = new_A
                    B = new_B
        else:
            # Swap: remove one, add another
            if A:
                idx = random.randint(0, len(A) - 1)
                candidates = [x for x in range(1, 257) if x not in A]
                if candidates:
                    cand = random.choice(candidates)
                    new_A = A[:idx] + [cand] + A[idx+1:]
                    new_B = _compute_bitsets(new_A)
                    if _is_valid_bitsets(new_B):
                        A = new_A
                        B = new_B
        
        # Track best
        if len(A) > best_n:
            best_n = len(A)
            best_A = A[:]
        
        current_temp *= cooling_rate
    
    return best_A

def construct():
    start_time = time.time()
    deadline = 55.0
    
    best_A = []
    
    # Helper to check time
    def time_left():
        return deadline - (time.time() - start_time)
        
    # Strategy 1: Greedy Ascending
    A = []
    used_set = set()
    for candidate in range(1, 257):
        if time_left() < 5:
            break
        new_A = A + [candidate]
        if _check_validity_bitset(new_A):
            A = new_A
            used_set.add(candidate)
    if len(A) > len(best_A):
        best_A = A[:]

    # Strategy 2: Greedy Descending
    if time_left() > 5:
        A = []
        used_set = set()
        for candidate in range(256, 0, -1):
            if time_left() < 5:
                break
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
                used_set.add(candidate)
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 3: Backtracking Search with Forward Checking
    # This is the key improvement: use backtracking to explore non-greedy combinations
    if time_left() > 10:
        # Start from the best greedy solution and try to extend it
        base_A = best_A[:]
        n = len(base_A)
        used_set = set(base_A)
        
        # Compute initial bitsets
        B = _compute_bitsets(base_A)
        
        # Try to extend to various target sizes
        for target_size in range(n + 1, min(n + 15, 25)):
            if time_left() < 5:
                break
            
            # Use backtracking to try to reach target_size
            # Candidate order: try elements not in base_A, sorted by some heuristic
            candidates = [x for x in range(1, 257) if x not in used_set]
            
            # Sort candidates by some heuristic: prefer values that are "spread out"
            # or try a few different orderings
            for order_type in ['ascending', 'descending']:
                if time_left() < 3:
                    break
                
                if order_type == 'ascending':
                    ordered_candidates = candidates[:]
                elif order_type == 'descending':
                    ordered_candidates = candidates[::-1]
                
                result = _backtrack_search(base_A, B, target_size, time.time() + min(time_left() - 3, 8), ordered_candidates, used_set)
                if result is not None and len(result) > len(best_A):
                    best_A = result
                    used_set = set(best_A)
                    B = _compute_bitsets(best_A)
                    break  # Found a better solution, try next target size
            
            if len(best_A) >= target_size:
                continue  # Already achieved this target, move to next
        
        # If backtracking from base didn't work, try from scratch with smaller targets
        if len(best_A) < 10 and time_left() > 5:
            for target_size in range(10, 15):
                if time_left() < 3:
                    break
                
                # Try backtracking from empty set
                empty_A = []
                empty_B = [0] * 1
                empty_B[0] = 1
                empty_used = set()
                
                candidates = list(range(1, 257))
                # Try a few orderings
                for order_type in ['ascending', 'descending']:
                    if time_left() < 2:
                        break
                    
                    if order_type == 'ascending':
                        ordered_candidates = candidates[:]
                    else:
                        ordered_candidates = candidates[::-1]
                    
                    result = _backtrack_search(empty_A, empty_B, target_size, time.time() + min(time_left() - 2, 5), ordered_candidates, empty_used)
                    if result is not None and len(result) > len(best_A):
                        best_A = result
                        break

    # Strategy 4: Local Search (Add)
    if time_left() > 3:
        A = best_A[:]
        improved = True
        while improved and time_left() > 1:
            improved = False
            n = len(A)
            
            # Try to add one element
            for candidate in range(1, 257):
                if time_left() < 1:
                    break
                if candidate in A:
                    continue
                new_A = A + [candidate]
                if _check_validity_bitset(new_A):
                    A = new_A
                    improved = True
                    break
            
            if len(A) > len(best_A):
                best_A = A[:]
            if not improved:
                break

    # Strategy 5: Multiple Randomized Greedy Restarts
    random.seed(42)
    
    for _ in range(5):
        if time_left() < 5:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        A = []
        for candidate in candidates:
            if time_left() < 5:
                break
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 6: Simulated Annealing
    if time_left() > 10:
        sa_deadline = time.time() + min(time_left() - 2, 20)
        sa_result = _simulated_annealing(best_A, sa_deadline, iters_per_step=200)
        if len(sa_result) > len(best_A):
            best_A = sa_result

    # Final local search polish
    if time_left() > 2:
        A = best_A[:]
        improved = True
        while improved and time_left() > 0.5:
            improved = False
            # Try to add one element
            for candidate in range(1, 257):
                if time_left() < 0.5:
                    break
                if candidate in A:
                    continue
                new_A = A + [candidate]
                if _check_validity_bitset(new_A):
                    A = new_A
                    improved = True
                    break
            if len(A) > len(best_A):
                best_A = A[:]
            if not improved:
                break

    return [int(x) for x in best_A]
