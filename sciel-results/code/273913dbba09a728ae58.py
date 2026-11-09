import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity_fast(sums, n):
    """
    Check validity in O(n) by maintaining union of all previous sums.
    sums[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    """
    # sums[0] is always 1 (sum 0)
    # We need sums[i] & sums[j] == 0 for all i != j
    # Efficient check: build cumulative union as we go
    union = 0
    for k in range(n + 1):
        if sums[k] & union:
            return False
        union |= sums[k]
    return True

def construct():
    best_set = []
    best_len = 0
    
    def try_add(A, sums, n, x):
        """Try adding x to set A with current sums. Returns (success, new_sums, new_n)"""
        new_sums = sums[:]
        for k in range(n, -1, -1):
            new_sums[k+1] |= (new_sums[k] << x)
        if check_validity_fast(new_sums, n + 1):
            return True, new_sums, n + 1
        return False, None, n
    
    def greedy_search(candidates, A=None, sums=None, n=None):
        """Greedy search over candidate list"""
        if A is None:
            A = []
            sums = [0] * 61
            sums[0] = 1
            n = 0
        for x in candidates:
            if is_done():
                break
            success, new_sums, new_n = try_add(A, sums, n, x)
            if success:
                A.append(x)
                sums = new_sums
                n = new_n
                if n > best_len:
                    nonlocal best_len
                    best_len = n
                    # Update best_set via closure
                    update_best(A)
        return A, sums, n
    
    def update_best(A):
        nonlocal best_set, best_len
        if len(A) > best_len:
            best_set = A[:]
            best_len = len(A)
    
    # Strategy 1: Greedy Ascending
    A, sums, n = greedy_search(range(1, 257))
    
    # Strategy 2: Greedy Descending
    A, sums, n = greedy_search(range(256, 0, -1))
    
    # Strategy 3: Greedy with gap-based ordering
    # Try different gap patterns
    for gap in [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64]:
        if is_done():
            break
        candidates = list(range(1, 257, gap))
        # Also try starting from different offsets
        for offset in range(gap):
            if is_done():
                break
            cands = [x for x in range(1 + offset, 257, gap)]
            greedy_search(cands)
    
    # Strategy 4: Randomized Greedy with Restarts
    max_restarts = 200
    for r in range(max_restarts):
        if is_done():
            break
        
        # Different randomization strategies
        strategy = r % 4
        
        if strategy == 0:
            # Pure random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif strategy == 1:
            # Random with bias toward larger numbers
            candidates = list(range(1, 257))
            # Weight by value - larger values get higher priority
            weighted = [(x, x**0.5) for x in candidates]
            random.shuffle(weighted)
            candidates = [x for x, w in sorted(weighted, key=lambda t: -t[1])]
        elif strategy == 2:
            # Random with bias toward smaller numbers
            candidates = list(range(1, 257))
            weighted = [(x, 257 - x) for x in candidates]
            random.shuffle(weighted)
            candidates = [x for x, w in sorted(weighted, key=lambda t: -t[1])]
        else:
            # Random with gap-based structure
            gap = random.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32])
            offset = random.randint(0, gap - 1)
            candidates = list(range(1 + offset, 257, gap))
            random.shuffle(candidates)
        
        greedy_search(candidates)
        
        # Local search: try to improve by replacing elements
        if len(best_set) > 0 and not is_done():
            current = best_set[:]
            current_sums = [0] * 61
            current_sums[0] = 1
            current_n = 0
            for x in current:
                new_sums = current_sums[:]
                for k in range(current_n, -1, -1):
                    new_sums[k+1] |= (new_sums[k] << x)
                current_sums = new_sums
                current_n += 1
            
            # Try swapping each element with a smaller unused one
            used = set(current)
            for i in range(len(current)):
                if is_done():
                    break
                for replacement in range(1, current[i]):
                    if replacement in used:
                        continue
                    # Try replacing current[i] with replacement
                    test_set = current[:]
                    test_set[i] = replacement
                    # Check if this allows adding more elements
                    test_sums = [0] * 61
                    test_sums[0] = 1
                    test_n = 0
                    valid = True
                    for x in test_set:
                        new_sums = test_sums[:]
                        for k in range(test_n, -1, -1):
                            new_sums[k+1] |= (new_sums[k] << x)
                        if not check_validity_fast(new_sums, test_n + 1):
                            valid = False
                            break
                        test_sums = new_sums
                        test_n += 1
                    if valid and test_n > len(best_set):
                        update_best(test_set)
                        break
    
    # Final check and return
    # Ensure types are correct
    return [int(x) for x in best_set]
