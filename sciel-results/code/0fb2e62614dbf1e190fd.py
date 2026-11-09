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

def _incremental_check(A, sums_by_k, new_elem):
    """Check if adding new_elem to A (with current sums_by_k) keeps validity.
    Returns new sums_by_k if valid, None otherwise."""
    n = len(A)
    # Compute new sums_by_k
    new_sums = [set(s) for s in sums_by_k]
    for k in range(n, -1, -1):
        for s in sums_by_k[k]:
            new_sums[k + 1].add(s + new_elem)
    
    # Check for conflicts across different cardinalities
    seen = {}
    for k in range(n + 2):
        for s in new_sums[k]:
            if s in seen:
                return None
            seen[s] = k
    return new_sums

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_A = []
    best_len = 0
    
    # Strategy 1: Greedy construction with incremental validity checking
    orderings = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        [i for i in range(1, 257) if i % 2 == 1],
        [i for i in range(1, 257) if i % 2 == 0],
        [i for i in range(1, 257) if i % 3 == 1],
        [i for i in range(1, 257) if i % 3 == 2],
        [i for i in range(1, 257) if i % 5 == 1],
        [i for i in range(1, 257) if i % 5 == 2],
        [i for i in range(1, 257) if i % 7 == 1],
        [i for i in range(1, 257) if i % 7 == 2],
        [i for i in range(1, 257) if i % 7 == 3],
    ]
    
    for ordering in orderings:
        if time.time() - start_time > deadline - 15:
            break
        
        A = []
        sums_by_k = [set() for _ in range(1)]
        sums_by_k[0] = {0}
        
        for candidate in ordering:
            if time.time() - start_time > deadline - 15:
                break
            new_sums = _incremental_check(A, sums_by_k, candidate)
            if new_sums is not None:
                A.append(candidate)
                sums_by_k = new_sums
        
        if len(A) > best_len:
            best_len = len(A)
            best_A = A[:]
    
    # Strategy 2: Local search - try to add elements using incremental check
    A = best_A[:]
    if A:
        # Rebuild sums_by_k for current A
        sums_by_k = [set() for _ in range(len(A) + 1)]
        sums_by_k[0] = {0}
        for a in A:
            for k in range(len(A) - 1, -1, -1):
                for s in list(sums_by_k[k]):
                    sums_by_k[k + 1].add(s + a)
        
        improved = True
        while improved and time.time() - start_time < deadline - 10:
            improved = False
            for candidate in range(1, 257):
                if time.time() - start_time > deadline - 10:
                    break
                if candidate in A:
                    continue
                new_sums = _incremental_check(A, sums_by_k, candidate)
                if new_sums is not None:
                    A.append(candidate)
                    sums_by_k = new_sums
                    improved = True
                    break
    
    if len(A) > best_len:
        best_len = len(A)
        best_A = A[:]
    
    # Strategy 3: Randomized restarts with incremental checking
    for restart in range(50):
        if time.time() - start_time > deadline - 15:
            break
        
        # Create a random subset to start
        random.seed(restart * 12345 + 7)
        initial_size = random.randint(2, 6)
        A = random.sample(range(1, 257), initial_size)
        
        # Check if initial set is valid using incremental building
        sums_by_k = [set() for _ in range(1)]
        sums_by_k[0] = {0}
        valid = True
        for a in A:
            new_sums = _incremental_check([], sums_by_k, a)
            if new_sums is None:
                valid = False
                break
            sums_by_k = new_sums
        
        if not valid:
            # Try smaller sets
            for size in range(initial_size - 1, 1, -1):
                A = random.sample(range(1, 257), size)
                sums_by_k = [set() for _ in range(1)]
                sums_by_k[0] = {0}
                valid = True
                for a in A:
                    new_sums = _incremental_check([], sums_by_k, a)
                    if new_sums is None:
                        valid = False
                        break
                    sums_by_k = new_sums
                if valid:
                    break
            else:
                continue
        
        # Greedily add elements in random order
        remaining = [x for x in range(1, 257) if x not in A]
        random.shuffle(remaining)
        
        for candidate in remaining:
            if time.time() - start_time > deadline - 15:
                break
            new_sums = _incremental_check(A, sums_by_k, candidate)
            if new_sums is not None:
                A.append(candidate)
                sums_by_k = new_sums
        
        if len(A) > best_len:
            best_len = len(A)
            best_A = A[:]
    
    # Strategy 4: Try to find larger sets by starting with specific patterns
    # Powers of 2 and nearby values often work well
    pattern_sets = [
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 3, 9, 27, 81],
        [2, 5, 14, 41, 122],
        [1, 2, 5, 12, 29, 70, 169],
        [3, 7, 15, 31, 63, 127],
        [1, 4, 13, 40, 121],
        [2, 6, 18, 54, 162],
        [1, 3, 8, 21, 55, 144],
    ]
    
    for pattern in pattern_sets:
        if time.time() - start_time > deadline - 15:
            break
        
        A = [x for x in pattern if 1 <= x <= 256]
        sums_by_k = [set() for _ in range(1)]
        sums_by_k[0] = {0}
        valid = True
        for a in A:
            new_sums = _incremental_check([], sums_by_k, a)
            if new_sums is None:
                valid = False
                break
            sums_by_k = new_sums
        
        if not valid:
            continue
        
        # Try to add more elements
        for candidate in range(1, 257):
            if time.time() - start_time > deadline - 15:
                break
            if candidate in A:
                continue
            new_sums = _incremental_check(A, sums_by_k, candidate)
            if new_sums is not None:
                A.append(candidate)
                sums_by_k = new_sums
        
        if len(A) > best_len:
            best_len = len(A)
            best_A = A[:]
    
    # Strategy 5: Backtracking with pruning for small target sizes
    def backtrack_with_pruning(A, sums_by_k, start_idx, target_size):
        if time.time() - start_time > deadline - 5:
            return None, None
        if len(A) >= target_size:
            return A[:], sums_by_k
        
        for i in range(start_idx, 257):
            new_sums = _incremental_check(A, sums_by_k, i)
            if new_sums is not None:
                new_A = A + [i]
                result, result_sums = backtrack_with_pruning(new_A, new_sums, i + 1, target_size)
                if result is not None:
                    return result, result_sums
        return None, None
    
    # Try to find sets of increasing size
    for target_size in range(best_len + 1, min(best_len + 4, 15)):
        if time.time() - start_time > deadline - 5:
            break
        result, _ = backtrack_with_pruning([], [set([0])], 1, target_size)
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
