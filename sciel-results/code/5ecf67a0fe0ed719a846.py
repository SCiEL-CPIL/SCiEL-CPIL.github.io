"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_3aps(A):
    """Count distinct 3-term APs with positive difference in set A."""
    A_set = set(A)
    count = 0
    for a in A:
        for b in A:
            if b > a:
                c = 2*b - a
                if c in A_set and c > b:
                    count += 1
    return count

def _has_4ap(A):
    """Check if set A contains a nonconstant 4-term AP."""
    A_set = set(A)
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            a, b = A[i], A[j]
            d = b - a
            if d <= 0:
                continue
            c = b + d
            e = c + d
            if c in A_set and e in A_set:
                return True
    return False

def _is_4ap_free_addition(A_set, x):
    """Check if adding x to A_set creates a 4-term AP."""
    # x can be:
    # 1st term: x, x+d, x+2d, x+3d
    # 2nd term: x-3d, x-2d, x-d, x
    # 3rd term: x-2d, x-d, x, x+d
    # 4th term: x-3d, x-2d, x-d, x
    # More systematically: for each d>0, check all 4 consecutive terms
    for d in range(1, 257):
        # x as 1st
        if x + d in A_set and x + 2*d in A_set and x + 3*d in A_set:
            return False
        # x as 2nd
        if x - 3*d >= 1 and x - 3*d in A_set and x - 2*d in A_set and x - d in A_set:
            return False
        # x as 3rd
        if x - 2*d >= 1 and x - 2*d in A_set and x - d in A_set and x + d in A_set:
            return False
        # x as 4th
        if x - 3*d >= 1 and x - 3*d in A_set and x - 2*d in A_set and x - d in A_set:
            return False
    return True

def _delta_3aps_for_addition(A_set, x):
    """Count how many 3-term APs are formed by adding x to A_set."""
    count = 0
    for a in A_set:
        # x as first: x, b, c where b = x + (c-x)/2, so 2b = x + c
        # Actually: AP (x, b, c) means b - x = c - b, so c = 2b - x
        # AP (a, x, c) means x - a = c - x, so c = 2x - a
        # AP (a, b, x) means b - a = x - b, so b = (a + x)/2
        # Case 1: x is first term: (x, b, c) with b, c in A_set
        #   2b = x + c => c = 2b - x
        for b in A_set:
            c = 2*b - x
            if c in A_set and c > b > x:
                count += 1
        # Case 2: x is middle term: (a, x, c) with a, c in A_set
        #   2x = a + c => c = 2x - a
        c = 2*x
        # Wait, a + c = 2x => c = 2x - a
        for a in A_set:
            c = 2*x - a
            if c in A_set and c > x > a:
                count += 1
        # Case 3: x is last term: (a, b, x) with a, b in A_set
        #   b - a = x - b => 2b = a + x => b = (a+x)/2
        for a in A_set:
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in A_set and x > b > a:
                    count += 1
    return count

def _greedy_construct():
    """Greedy construction: add elements that maximize 3-APs while avoiding 4-APs."""
    A_set = set()
    # Try adding elements in various orders
    for x in range(1, 257):
        if _is_4ap_free_addition(A_set, x):
            A_set.add(x)
    return sorted(A_set)

def _local_search(A, time_limit=50.0):
    """Local search: try swaps and additions/deletions to improve 3-AP count."""
    start_time = time.time()
    A_set = set(A)
    current_score = _count_3aps(A)
    best_A = list(A)
    best_score = current_score
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try adding a random element not in A
        for _ in range(20):
            x = random.randint(1, 256)
            if x not in A_set:
                if _is_4ap_free_addition(A_set, x):
                    new_score = current_score + _delta_3aps_for_addition(A_set, x)
                    if new_score > current_score:
                        A_set.add(x)
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_A = sorted(A_set)
        
        # Try removing and re-adding
        if not improved and len(A_set) > 10:
            for _ in range(10):
                x = random.choice(sorted(A_set))
                A_set.remove(x)
                # Try to add something better
                for y in range(1, 257):
                    if y not in A_set:
                        if _is_4ap_free_addition(A_set, y):
                            new_score = current_score - _delta_3aps_for_addition(A_set, x) + _delta_3aps_for_addition(A_set, y)
                            if new_score > current_score:
                                A_set.add(y)
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_A = sorted(A_set)
                                break
                if not improved:
                    A_set.add(x)  # restore
        
        if time.time() - start_time > time_limit - 5:
            break
    
    return best_A, best_score

def _cayley_construct():
    """Use a known 4-AP-free set construction (Cayley's construction or similar)."""
    # Use base-3 digits: numbers with only 0s and 1s in base 3
    A = set()
    for i in range(257):
        # Check if i in base 3 has only digits 0 and 1
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid and i >= 1:
            A.add(i)
    return sorted(A)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with a good initial set
    best_A = _greedy_construct()
    best_score = _count_3aps(best_A)
    
    # Also try Cayley construction
    cayley_A = _cayley_construct()
    cayley_score = _count_3aps(cayley_A)
    if cayley_score > best_score and not _has_4ap(cayley_A):
        best_A = cayley_A
        best_score = cayley_score
    
    # Local search to improve
    if time.time() < deadline - 10:
        best_A, best_score = _local_search(best_A, time_limit=min(40.0, deadline - time.time() - 5))
    
    # Final validation
    if _has_4ap(best_A):
        # Fallback to greedy
        best_A = _greedy_construct()
    
    return best_A
# EVOLVE-BLOCK-END
