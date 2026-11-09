"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity(A):
    """Check if set A satisfies: no sum is achieved by subsets of different cardinalities."""
    n = len(A)
    if n == 0:
        return True
    # sums_by_k[k] = set of achievable sums using exactly k elements
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    
    for a in A:
        # Update in reverse to avoid using a twice
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                for s in sums_by_k[k]:
                    sums_by_k[k + 1].add(s + a)
    
    # Check pairwise disjointness
    seen = set()
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen:
                return False
            seen.add(s)
    return True

def check_validity_fast(A):
    """Faster validity check using the fact that we build incrementally."""
    n = len(A)
    if n == 0:
        return True
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    
    for a in A:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                for s in sums_by_k[k]:
                    sums_by_k[k + 1].add(s + a)
    
    seen = set()
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen:
                return False
            seen.add(s)
    return True

def greedy_construct():
    """Greedy: try to add elements in a smart order."""
    A = []
    used = set()
    
    # Try elements in various orders
    candidates = list(range(1, 257))
    
    for a in candidates:
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity(A):
            A.pop()
            used.discard(a)
    
    return A

def greedy_construct_descending():
    """Greedy with descending order."""
    A = []
    used = set()
    
    for a in range(256, 0, -1):
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity(A):
            A.pop()
            used.discard(a)
    
    return A

def random_search(num_trials=100):
    """Try random subsets and see which are valid."""
    best = []
    
    for trial in range(num_trials):
        if is_done():
            break
        
        # Random size
        size = random.randint(10, 30)
        A = random.sample(range(1, 257), size)
        
        if check_validity(A):
            if len(A) > len(best):
                best = A
    
    return best

def local_search_improve(A):
    """Try to improve A by swapping elements."""
    A = list(A)
    used = set(A)
    improved = True
    
    while improved and not is_done():
        improved = False
        for i in range(len(A)):
            for new_val in range(1, 257):
                if new_val in used:
                    continue
                old_val = A[i]
                A[i] = new_val
                used.discard(old_val)
                used.add(new_val)
                
                if check_validity(A):
                    # Try to add another element
                    for add_val in range(1, 257):
                        if add_val in used:
                            continue
                        A.append(add_val)
                        used.add(add_val)
                        if check_validity(A):
                            improved = True
                            break
                        else:
                            A.pop()
                            used.discard(add_val)
                    break
                else:
                    A[i] = old_val
                    used.discard(new_val)
                    used.add(old_val)
            if is_done():
                break
    
    return A

def construct():
    """Main entry point: try multiple strategies and return the best valid set."""
    best = []
    
    # Strategy 1: Greedy ascending
    A1 = greedy_construct()
    if len(A1) > len(best):
        best = A1
    
    if is_done():
        return best
    
    # Strategy 2: Greedy descending
    A2 = greedy_construct_descending()
    if len(A2) > len(best):
        best = A2
    
    if is_done():
        return best
    
    # Strategy 3: Local search improvement
    A3 = local_search_improve(best)
    if len(A3) > len(best):
        best = A3
    
    if is_done():
        return best
    
    # Strategy 4: Random search
    A4 = random_search(50)
    if len(A4) > len(best):
        best = A4
    
    # Ensure we return a valid result
    if not check_validity(best):
        # Fallback to a known valid small set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
