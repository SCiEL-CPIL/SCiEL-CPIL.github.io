import time
import random

def _count_3aps(A):
    """Count distinct 3-term APs with positive difference in set A."""
    A_set = set(A)
    count = 0
    # A 3-AP is (a, b, c) with a < b < c and b-a = c-b.
    # This is equivalent to a+c = 2b.
    # Iterate over b in A, and a in A such that a < b.
    # c = 2b - a. Check if c in A and c > b.
    for b in A:
        for a in A:
            if a < b:
                c = 2 * b - a
                if c in A_set and c > b:
                    count += 1
    return count

def _has_4ap(A):
    """Check if set A contains a nonconstant 4-term AP."""
    A_set = set(A)
    n = len(A)
    A_list = list(A)
    for i in range(n):
        a = A_list[i]
        for j in range(i + 1, n):
            b = A_list[j]
            d = b - a
            if d <= 0:
                continue
            c = b + d
            e = c + d
            if c in A_set and e in A_set:
                return True
    return False

def _delta_3aps_add(A_set, x):
    """Calculate increase in 3-APs if x is added to A_set."""
    count = 0
    # x can be first, middle, or last term.
    # 1. x is first: (x, b, c). b-x = c-b => c = 2b-x.
    #    Need b, c in A_set, b > x, c > b.
    for b in A_set:
        if b > x:
            c = 2 * b - x
            if c in A_set and c > b:
                count += 1
                
    # 2. x is middle: (a, x, c). x-a = c-x => c = 2x-a.
    #    Need a, c in A_set, a < x, c > x.
    for a in A_set:
        if a < x:
            c = 2 * x - a
            if c in A_set and c > x:
                count += 1
                
    # 3. x is last: (a, b, x). b-a = x-b => 2b = a+x => b = (a+x)/2.
    #    Need a, b in A_set, a < b < x.
    for a in A_set:
        if a < x:
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in A_set and a < b < x:
                    count += 1
    return count

def _delta_3aps_remove(A_set, x):
    """Calculate decrease in 3-APs if x is removed from A_set."""
    # This is the same as delta_add for the complement logic, 
    # but easier to just recount or use the same logic: 
    # How many APs involve x?
    return _delta_3aps_add(A_set - {x}, x)

def _is_4ap_free_add(A_set, x):
    """Check if adding x to A_set creates a 4-term AP."""
    # x can be 1st, 2nd, 3rd, or 4th term.
    # 1. x, x+d, x+2d, x+3d. Need x+d, x+2d, x+3d in A_set.
    for d in range(1, 257 - x):
        if x + d in A_set and x + 2 * d in A_set and x + 3 * d in A_set:
            return False
            
    # 2. x-3d, x-2d, x-d, x. Need x-3d, x-2d, x-d in A_set.
    for d in range(1, (x - 1) // 3 + 1):
        if x - 3 * d >= 1 and x - 3 * d in A_set and x - 2 * d in A_set and x - d in A_set:
            return False
            
    # 3. x-2d, x-d, x, x+d. Need x-2d, x-d, x+d in A_set.
    for d in range(1, min(x - 1, 256 - x) + 1):
        if x - 2 * d >= 1 and x - 2 * d in A_set and x - d in A_set and x + d in A_set:
            return False
            
    # 4. x-3d, x-2d, x-d, x. (Same as case 2? No, case 2 is x as 4th. Case 4 is x as 4th? 
    # Wait. 
    # AP: a, a+d, a+2d, a+3d.
    # If x is 1st: x, x+d, x+2d, x+3d.
    # If x is 2nd: x-3d, x-2d, x-d, x.
    # If x is 3rd: x-2d, x-d, x, x+d.
    # If x is 4th: x-3d, x-2d, x-d, x.
    # Case 2 and Case 4 are identical conditions for the other 3 elements being in A_set.
    # So we just need to check cases 1, 2, 3.
    
    return True

def _greedy_construct():
    """Greedy construction: add elements that maximize 3-APs while avoiding 4-APs."""
    A_set = set()
    # We want to add elements that give the biggest boost to 3-APs.
    # We can iterate multiple passes or just once in increasing order?
    # Increasing order might not be optimal. 
    # Let's try adding elements in order of increasing potential contribution?
    # Or just random restarts of greedy.
    
    # Simple greedy: try all x, pick best valid one.
    # This is O(N^3) per step, N steps. N=256. 256^3 = 16M. Might be slow in Python.
    # Let's just add in increasing order first, then refine.
    for x in range(1, 257):
        if _is_4ap_free_add(A_set, x):
            A_set.add(x)
    return sorted(A_set)

def _local_search(A, time_limit=5.0):
    """Local search: try swaps and additions to improve 3-AP count."""
    start_time = time.time()
    A_set = set(A)
    current_score = _count_3aps(A)
    best_A = list(A)
    best_score = current_score
    
    random.seed(42)
    all_elements = set(range(1, 257))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try swaps: remove x, add y
        # Randomly pick x in A, y not in A
        if len(A_set) > 0:
            x = random.choice(list(A_set))
            candidates = list(all_elements - A_set)
            if candidates:
                y = random.choice(candidates)
                
                # Check if swap is valid (no 4-AP)
                # Remove x, check if y can be added
                temp_set = A_set - {x}
                if _is_4ap_free_add(temp_set, y):
                    new_set = temp_set | {y}
                    # Calculate new score
                    # score_new = score_old - delta_remove(x) + delta_add(y, temp_set)
                    # delta_remove(x) is number of APs involving x in A_set
                    # delta_add(y, temp_set) is number of APs involving y in new_set
                    
                    # To be safe and simple, just recount if the set is small enough or changes are small.
                    # Recounting is O(N^2) ~ 16k ops. Very fast.
                    new_score = _count_3aps(new_set)
                    
                    if new_score > current_score:
                        A_set = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_A = sorted(A_set)
        
        # Try additions
        if not improved:
            candidates = list(all_elements - A_set)
            if candidates:
                y = random.choice(candidates)
                if _is_4ap_free_add(A_set, y):
                    new_set = A_set | {y}
                    new_score = _count_3aps(new_set)
                    if new_score > current_score:
                        A_set = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_A = sorted(A_set)
                            
        if time.time() - start_time > time_limit - 0.5:
            break
            
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Initial Greedy
    best_A = _greedy_construct()
    best_score = _count_3aps(best_A)
    
    # 2. Local Search from Greedy
    if time.time() < deadline - 10:
        A, score = _local_search(best_A, time_limit=min(10.0, deadline - time.time() - 5))
        if score > best_score:
            best_A = A
            best_score = score
            
    # 3. Random Restarts
    # Try a few random initial sets
    for i in range(10):
        if time.time() > deadline - 15:
            break
            
        # Generate a random set of random size
        size = random.randint(40, 100)
        random_set = set(random.sample(range(1, 257), size))
        
        # Greedy Add to this random set? Or just Local Search?
        # Local Search might get stuck if the initial set is bad.
        # Let's do a quick greedy completion.
        A_rand = set(random_set)
        # Try to add elements
        for x in range(1, 257):
            if x not in A_rand and _is_4ap_free_add(A_rand, x):
                A_rand.add(x)
                
        score_rand = _count_3aps(A_rand)
        
        if score_rand > best_score:
            best_A = sorted(A_rand)
            best_score = score_rand
            
        # Local search on this
        if time.time() < deadline - 10:
            A, score = _local_search(best_A, time_limit=min(5.0, deadline - time.time() - 5))
            if score > best_score:
                best_A = A
                best_score = score

    # Final Validation
    if _has_4ap(best_A):
        # Fallback
        best_A = _greedy_construct()
        
    return best_A
