import time
import random
import itertools
from bisect import bisect_left, bisect_right

N = 128
MAX_TIME = 87.0  # safety margin for 90s limit

def _count_conflicts(s_set):
    """
    Returns a dictionary mapping each element in s_set to the number of 
    forbidden triples (a,b,c) in s_set that contain that element.
    Uses bitset for O(1) membership with early termination.
    """
    # Build bitset: bit i set means (i+1) is in s_set
    bitset = 0
    for x in s_set:
        bitset |= (1 << (x - 1))
    
    s_list = sorted(s_set)
    n = len(s_list)
    counts = {x: 0 for x in s_list}
    
    def in_set(val):
        if val < 1 or val > 128:
            return False
        return (bitset >> (val - 1)) & 1
    
    # Precompute: for each pair (i,j) with i<j, check if s_list[i]+s_list[j] is in set
    # Then for valid pairs, find c > b such that a+c and b+c are both in set
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
            
        for j in range(i + 1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if not in_set(ab):
                continue
            # Now find c > b such that a+c and b+c are both in set
            # c must satisfy: c > b, a+c <= 128, b+c <= 128
            # So c <= 128 - b (since b >= a, 128-b <= 128-a)
            max_c = 128 - b
            if max_c <= b:
                continue
            # c ranges from b+1 to min(max_c, s_list[-1])
            found = False
            for k in range(j + 1, n):
                c = s_list[k]
                if c > max_c:
                    break
                if in_set(a + c) and in_set(b + c):
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def _greedy_remove_until_valid(s_set, start_time):
    """
    Start with s_set and remove elements with highest conflict count until valid.
    Uses a weighted score: conflict_count * (1 + 0.1 * (N - x)) to prefer removing
    smaller elements (which tend to create more conflicts).
    """
    s_set = set(s_set)
    while True:
        if time.time() - start_time > MAX_TIME - 5.0:
            break
        counts = _count_conflicts(s_set)
        if not counts or all(v == 0 for v in counts.values()):
            break # Valid
        
        # Find element with max weighted score
        # Weight: conflict count * (1 + 0.1 * (N - x))
        # This prefers removing smaller elements with high conflict counts
        best_score = -1
        remove_el = None
        for x, c in counts.items():
            if c == 0:
                continue
            score = c * (1.0 + 0.1 * (N - x))
            if score > best_score:
                best_score = score
                remove_el = x
        
        if remove_el is None:
            break  # No conflicts
        
        s_set.remove(remove_el)
        
    return s_set

def _fast_validity_check(s):
    """Faster check using bitset for O(1) membership."""
    s_set = set(s)
    bitset = 0
    for x in s_set:
        bitset |= (1 << (x - 1))
    
    s_list = sorted(s)
    n = len(s_list)
    
    def in_set(val):
        if val < 1 or val > 128:
            return False
        return (bitset >> (val - 1)) & 1
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n: break
        if a + s_list[i+1] > 128: break
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128: break
            if not in_set(ab):
                continue
            for k in range(j+1, n):
                c = s_list[k]
                if b + c > 128: break
                if in_set(a + c) and in_set(b + c):
                    return False
    return True

def _try_greedy_add(s_set, start_time):
    """Try to greedily add elements to improve the set."""
    s_set = set(s_set)
    candidates = [x for x in range(1, N+1) if x not in s_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - 5.0:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            new_s = s_set | {cand}
            if _fast_validity_check(new_s):
                s_set = new_s
                candidates = [x for x in candidates if x != cand]
                improved = True
    
    return sorted(s_set)

def _try_remove_add(s_set, start_time):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - 5.0:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        # Start with remaining_set and greedily add
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            new_s = current | {cand}
            if _fast_validity_check(new_s):
                current = new_s
        
        if len(current) > best_len:
            best_s = current
            best_len = len(current)
    
    return sorted(best_s)

def construct():
    start_time = time.time()
    
    # Strategy: Start with full set, greedily remove conflicts, then add back.
    # This was measured to yield 1.25 in knowledge.
    
    # 1. Greedy Removal Phase
    full_set = set(range(1, N+1))
    s_set = _greedy_remove_until_valid(full_set, start_time)
    
    # 2. Greedy Add Phase
    if time.time() - start_time < MAX_TIME - 5.0:
        s_set = set(_try_greedy_add(s_set, start_time))
        
    # 3. Local Search (Remove-Add) Phase
    if time.time() - start_time < MAX_TIME - 3.0:
        s_set = set(_try_remove_add(s_set, start_time))
        
    # 4. Random Perturbation / Restart if time permits
    # We can try a few random restarts of the removal process if time allows
    best_s = sorted(s_set)
    best_len = len(best_s)
    
    if time.time() - start_time < MAX_TIME - 10.0:
        for _ in range(10): # Max 10 restarts
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            # Randomize tie-breaking in removal? 
            # Or just run the same deterministic process? 
            # The deterministic process is stable. 
            # Let's try perturbing the initial set or the removal order.
            # Since the removal is deterministic based on counts, it will find the same local optimum.
            # To explore, we can randomly remove one extra element before starting greedy removal.
            perturbed_set = set(full_set)
            if perturbed_set:
                random_remove = random.choice(list(perturbed_set))
                perturbed_set.remove(random_remove)
            
            s_set = _greedy_remove_until_valid(perturbed_set, start_time)
            
            if time.time() - start_time < MAX_TIME - 3.0:
                s_set = set(_try_greedy_add(s_set, start_time))
                
            if time.time() - start_time < MAX_TIME - 1.0:
                s_set = set(_try_remove_add(s_set, start_time))
                
            if len(s_set) > best_len:
                best_s = sorted(s_set)
                best_len = len(best_s)
                
    # Final validation
    if not _fast_validity_check(set(best_s)):
        # Fallback to known valid set (top half)
        best_s = list(range(63, 129))
        
    return [int(x) for x in best_s]
