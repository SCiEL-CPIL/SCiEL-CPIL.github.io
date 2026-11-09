import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity_bitset_fast(sums, n, new_k):
    """
    Check if adding an element that creates new sums at cardinality new_k
    doesn't conflict with existing sums. Only check the new cardinality against all others.
    new_k is the highest cardinality that got new sums (should be n+1).
    """
    # Only need to check sums[new_k] against all other sums
    # because sums[0..n] haven't changed
    for i in range(new_k):
        if sums[i] & sums[new_k]:
            return False
    return True

def construct():
    best_set = []
    
    # Strategy 1: Greedy Ascending
    A = []
    sums = [0] * 61
    sums[0] = 1
    n = 0
    for x in range(1, 257):
        if is_done(): break
        # Try adding x
        new_sums = sums[:]
        for k in range(n, -1, -1):
            new_sums[k+1] |= (new_sums[k] << x)
        if check_validity_bitset_fast(new_sums, n + 1, n + 1):
            A.append(x)
            sums = new_sums
            n += 1
            if n > len(best_set):
                best_set = A[:]
    
    # Strategy 2: Greedy Descending
    A = []
    sums = [0] * 61
    sums[0] = 1
    n = 0
    for x in range(256, 0, -1):
        if is_done(): break
        new_sums = sums[:]
        for k in range(n, -1, -1):
            new_sums[k+1] |= (new_sums[k] << x)
        if check_validity_bitset_fast(new_sums, n + 1, n + 1):
            A.append(x)
            sums = new_sums
            n += 1
            if n > len(best_set):
                best_set = A[:]
                
    # Strategy 3: Randomized Greedy with Restarts
    # We try to add elements in a random order, restarting if we get stuck or to find better sets
    max_restarts = 100
    for _ in range(max_restarts):
        if is_done(): break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        A = []
        sums = [0] * 61
        sums[0] = 1
        n = 0
        for x in candidates:
            if is_done(): break
            new_sums = sums[:]
            for k in range(n, -1, -1):
                new_sums[k+1] |= (new_sums[k] << x)
            if check_validity_bitset_fast(new_sums, n + 1, n + 1):
                A.append(x)
                sums = new_sums
                n += 1
                if n > len(best_set):
                    best_set = A[:]
        
        # If we found a good set, try local improvement: swap out one element to allow more additions
        if len(best_set) > 5 and not is_done():
            # Try swapping each element in best_set with a candidate not in best_set
            best_set_set = set(best_set)
            for i, elem in enumerate(best_set):
                if is_done(): break
                for new_elem in range(1, 257):
                    if is_done(): break
                    if new_elem in best_set_set:
                        continue
                    # Try replacing elem with new_elem
                    trial_set = best_set[:i] + [new_elem] + best_set[i+1:]
                    trial_set.sort()
                    # Validate the trial set
                    trial_sums = [0] * (len(trial_set) + 1)
                    trial_sums[0] = 1
                    for v in trial_set:
                        for k in range(len(trial_set), -1, -1):
                            if k > 0:
                                trial_sums[k] |= (trial_sums[k-1] << v)
                    # Check validity
                    valid = True
                    for a in range(len(trial_set) + 1):
                        for b in range(a + 1, len(trial_set) + 2):
                            if trial_sums[a] & trial_sums[b]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Try to extend this set
                        extend_sums = trial_sums[:]
                        extend_n = len(trial_set)
                        for x in range(1, 257):
                            if is_done(): break
                            if x in trial_set_set:
                                continue
                            new_sums = extend_sums[:]
                            for k in range(extend_n, -1, -1):
                                new_sums[k+1] |= (new_sums[k] << x)
                            if check_validity_bitset_fast(new_sums, extend_n + 1, extend_n + 1):
                                trial_set.append(x)
                                extend_sums = new_sums
                                extend_n += 1
                                if len(trial_set) > len(best_set):
                                    best_set = trial_set[:]
                                    break
    
    # Final check and return
    # Ensure types are correct
    return [int(x) for x in best_set]
