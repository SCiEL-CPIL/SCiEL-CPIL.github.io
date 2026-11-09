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

def _backtrack_search(A, B, max_size, deadline, candidates):
    """
    Backtracking search with forward checking.
    
    A: current list of selected integers
    B: list of bitsets where B[k] has bit s set if sum s achievable with k elements
    max_size: target size to reach
    deadline: absolute time deadline
    candidates: list of candidate integers to try
    
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
        if cand in A:
            continue
        
        # Forward check: try adding cand and see if it creates a conflict
        # Compute new B with cand added
        new_B = B[:]  # Copy current bitsets
        # Update in reverse to avoid using cand multiple times
        for k in range(n - 1, -1, -1):
            if new_B[k]:
                new_B[k + 1] |= (new_B[k] << cand)
        
        # Check for conflicts: new_B[k] and new_B[j] must be disjoint for k != j
        # Use the same trick: total OR should equal sum of bits
        total = 0
        sum_bits = 0
        valid = True
        for k in range(n + 2):  # n+1 possible cardinalities after adding one element
            if k < len(new_B):
                total |= new_B[k]
                sum_bits += new_B[k]
            else:
                break
        if total != sum_bits:
            continue  # Conflict, skip this candidate
        
        # Recurse
        new_A = A + [cand]
        result = _backtrack_search(new_A, new_B, max_size, deadline, candidates)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = 55.0
    
    best_A = []
    
    # Helper to check time
    def time_left():
        return deadline - (time.time() - start_time)
        
    # Strategy 1: Greedy Ascending
    A = []
    for candidate in range(1, 257):
        if time_left() < 5:
            break
        new_A = A + [candidate]
        if _check_validity_bitset(new_A):
            A = new_A
    if len(A) > len(best_A):
        best_A = A[:]

    # Strategy 2: Greedy Descending
    if time_left() > 5:
        A = []
        for candidate in range(256, 0, -1):
            if time_left() < 5:
                break
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 3: Backtracking Search with Forward Checking
    # This is the key improvement: use backtracking to explore non-greedy combinations
    if time_left() > 10:
        # Start from the best greedy solution and try to extend it
        base_A = best_A[:]
        n = len(base_A)
        
        # Compute initial bitsets
        B = [0] * (n + 1)
        B[0] = 1
        for a in base_A:
            for k in range(n - 1, -1, -1):
                if B[k]:
                    B[k + 1] |= (B[k] << a)
        
        # Try to extend to various target sizes
        for target_size in range(n + 1, min(n + 10, 20)):
            if time_left() < 5:
                break
            
            # Use backtracking to try to reach target_size
            # Candidate order: try elements not in base_A, sorted by some heuristic
            candidates = [x for x in range(1, 257) if x not in base_A]
            
            # Sort candidates by some heuristic: prefer values that are "spread out"
            # or try a few different orderings
            for order_type in ['ascending', 'descending', 'random']:
                if time_left() < 3:
                    break
                
                if order_type == 'ascending':
                    ordered_candidates = candidates[:]
                elif order_type == 'descending':
                    ordered_candidates = candidates[::-1]
                else:
                    random.seed(hash(tuple(base_A)) % 10000)
                    ordered_candidates = candidates[:]
                    random.shuffle(ordered_candidates)
                
                result = _backtrack_search(base_A, B, target_size, time.time() + min(time_left() - 3, 10), ordered_candidates)
                if result is not None and len(result) > len(best_A):
                    best_A = result
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
                
                candidates = list(range(1, 257))
                # Try a few orderings
                for order_type in ['ascending', 'descending']:
                    if time_left() < 2:
                        break
                    
                    if order_type == 'ascending':
                        ordered_candidates = candidates[:]
                    else:
                        ordered_candidates = candidates[::-1]
                    
                    result = _backtrack_search(empty_A, empty_B, target_size, time.time() + min(time_left() - 2, 5), ordered_candidates)
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

    return [int(x) for x in best_A]
