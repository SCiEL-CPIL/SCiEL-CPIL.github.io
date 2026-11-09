import time
import random
from collections import defaultdict

def _check_validity(A):
    """Check if set A satisfies cardinality-separated subset sums."""
    n = len(A)
    if n == 0:
        return True
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    for a in A:
        for k in range(n - 1, -1, -1):
            for s in list(sums_by_k[k]):
                sums_by_k[k + 1].add(s + a)
    
    seen = {}
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen:
                return False
            seen[s] = k
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_A = []
    best_len = 0
    
    # Strategy 1: Greedy construction with different orderings
    orderings = [
        range(1, 257),
        range(256, 0, -1),
        [i for i in range(1, 257) if i % 2 == 1],
        [i for i in range(1, 257) if i % 2 == 0],
        [i for i in range(1, 257) if i % 3 == 1],
        [i for i in range(1, 257) if i % 3 == 2],
        [i for i in range(1, 257) if i % 5 == 1],
        [i for i in range(1, 257) if i % 5 == 2],
    ]
    
    for ordering in orderings:
        if time.time() - start_time > deadline - 10:
            break
        
        A = []
        for candidate in ordering:
            if time.time() - start_time > deadline - 10:
                break
            if candidate in A:
                continue
            new_A = A + [candidate]
            if _check_validity(new_A):
                A = new_A
        
        if len(A) > best_len:
            best_len = len(A)
            best_A = A[:]
    
    # Strategy 2: Local search - try to add elements
    A = best_A[:]
    improved = True
    while improved and time.time() - start_time < deadline - 5:
        improved = False
        for candidate in range(1, 257):
            if time.time() - start_time > deadline - 5:
                break
            if candidate in A:
                continue
            new_A = A + [candidate]
            if _check_validity(new_A):
                A = new_A
                improved = True
                break
    
    if len(A) > best_len:
        best_len = len(A)
        best_A = A[:]
    
    # Strategy 3: Randomized restarts with different initial seeds
    for restart in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        # Create a random subset to start
        random.seed(restart * 12345)
        initial_size = random.randint(3, 8)
        A = random.sample(range(1, 257), initial_size)
        
        # Check if initial set is valid
        if not _check_validity(A):
            # Try smaller sets
            for size in range(initial_size - 1, 2, -1):
                A = random.sample(range(1, 257), size)
                if _check_validity(A):
                    break
            else:
                continue
        
        # Greedily add elements
        for candidate in range(1, 257):
            if time.time() - start_time > deadline - 10:
                break
            if candidate in A:
                continue
            new_A = A + [candidate]
            if _check_validity(new_A):
                A = new_A
        
        if len(A) > best_len:
            best_len = len(A)
            best_A = A[:]
    
    # Strategy 4: Backtracking search for small sets
    def backtrack(A, start_idx, target_size):
        if time.time() - start_time > deadline - 5:
            return None
        if len(A) >= target_size:
            if _check_validity(A):
                return A[:]
            return None
        
        for i in range(start_idx, 257):
            new_A = A + [i]
            # Quick check: if this is the last element, verify validity
            if len(new_A) == target_size:
                if _check_validity(new_A):
                    return new_A
            else:
                result = backtrack(new_A, i + 1, target_size)
                if result is not None:
                    return result
        return None
    
    # Try to find sets of increasing size
    for target_size in range(best_len + 1, 15):
        if time.time() - start_time > deadline - 5:
            break
        result = backtrack([], 1, target_size)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_A = result
            break
    
    # Final verification
    if best_A:
        if not _check_validity(best_A):
            # Fallback: rebuild from scratch with simple greedy
            A = []
            for candidate in range(1, 257):
                if time.time() - start_time > deadline - 1:
                    break
                new_A = A + [candidate]
                if _check_validity(new_A):
                    A = new_A
            best_A = A
    
    return [int(x) for x in best_A]
