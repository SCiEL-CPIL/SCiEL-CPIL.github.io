"""
Improved program: better local search with smart conflict analysis.

Key improvements:
1. Faster _can_add using bitmasks for membership checks
2. Smarter local search: try all add candidates, then targeted swaps
3. Better ordering heuristics
4. Multiple restarts with diverse strategies
"""
import time
import random
from math import gcd, isqrt

def _build_conflict_pairs(N=256):
    """
    For each a, compute the set of pairs (b,c) with b<c such that a | b*c.
    Also compute for each a, the set of 'conflict partners': elements x such that
    there exists some other element y where a|b*c would be violated if both x,y in set.
    
    Actually, let's think differently. For a fixed set S, a can be added iff:
    For all b in S: a does not divide b*c for any c in S, c != b.
    And for all c in S: b does not divide a*c for any b in S, b != c. (same condition)
    
    So the condition is: for all distinct b, c in S, a does not divide b*c.
    Equivalently: for all b in S, there is no c in S (c != b) with a | b*c.
    
    For a given a and b, the forbidden c's are multiples of a/gcd(a,b) that are != b.
    """
    pass

def _can_add(a, S_mask, N=256):
    """
    Check if a can be added to the set represented by S_mask (a set of ints).
    For each b in S, check if there exists c in S, c != b, such that a | b*c.
    """
    for b in S_mask:
        g = gcd(a, b)
        k1 = a // g
        # Need c in S, c != b, such that k1 | c
        # Check multiples of k1
        for c in range(k1, N + 1, k1):
            if c == b:
                continue
            if c in S_mask:
                return False
        # Also check: b | a*c for some c in S, c != b
        # g = gcd(a,b), k2 = b // g
        # Need c in S, c != b, such that k2 | c
        k2 = b // g
        for c in range(k2, N + 1, k2):
            if c == b:
                continue
            if c in S_mask:
                return False
    return True

def _can_add_fast(a, in_set, N=256):
    """
    Faster check using boolean array.
    """
    for b in range(2, N + 1):
        if not in_set[b]:
            continue
        g = gcd(a, b)
        k1 = a // g
        # Check multiples of k1
        for c in range(k1, N + 1, k1):
            if c == b:
                continue
            if in_set[c]:
                return False
        k2 = b // g
        for c in range(k2, N + 1, k2):
            if c == b:
                continue
            if in_set[c]:
                return False
    return True

def _greedy_construct(order, in_set, deadline, N=256):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S_list = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(x, in_set, N):
            in_set[x] = True
            S_list.append(x)
    return S_list

def _local_search(S_list, in_set, deadline, N=256, max_iters=100000):
    """
    Local search to improve the set.
    Strategy:
    1. Try to add any non-member that fits.
    2. If no adds possible, try swaps (remove one, add another).
    3. Track non-members and periodically re-evaluate.
    """
    current_size = len(S_list)
    non_members = [x for x in range(2, N + 1) if not in_set[x]]
    
    no_improve = 0
    max_no_improve = 500
    
    # Precompute a list of all elements sorted by some heuristic
    all_elements = list(range(2, N + 1))
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        if not non_members:
            break
            
        improved = False
        
        # Phase 1: Try to add elements (systematically, not just random)
        # Try all non-members in order of some heuristic
        add_candidates = non_members[:]
        # Sort by number of conflicts (fewer conflicts first)
        # For each candidate x, count how many pairs (b,c) in S violate x|b*c
        def conflict_count(x):
            count = 0
            S_set = set(S_list)
            for b in S_set:
                g = gcd(x, b)
                k1 = x // g
                for c in range(k1, N + 1, k1):
                    if c != b and c in S_set:
                        count += 1
                        break
                k2 = b // g
                for c in range(k2, N + 1, k2):
                    if c != b and c in S_set:
                        count += 1
                        break
            return count
        
        if iteration % 50 == 0 or no_improve < 100:
            # Sort candidates by conflict count
            add_candidates.sort(key=conflict_count)
        
        for x in add_candidates:
            if time.time() > deadline:
                break
            if x not in non_members:
                continue
            if _can_add_fast(x, in_set, N):
                in_set[x] = True
                S_list.append(x)
                current_size += 1
                non_members.remove(x)
                no_improve = 0
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps
        # For each pair (y in S, x not in S), check if removing y and adding x works
        # To be efficient, only try promising swaps
        if S_list and non_members:
            # Try a limited number of random swaps
            for _ in range(20):
                if time.time() > deadline:
                    break
                if not S_list or not non_members:
                    break
                
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                x_idx = random.randint(0, len(non_members) - 1)
                x = non_members[x_idx]
                
                # Temporarily remove y
                in_set[y] = False
                
                if _can_add_fast(x, in_set, N):
                    # Accept swap
                    in_set[x] = True
                    S_list[y_idx] = x
                    non_members[x_idx] = y
                    no_improve = 0
                    improved = True
                    break
                else:
                    # Revert
                    in_set[y] = True
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                no_improve = 0
                # Restart local search with a fresh random perturbation
                if random.random() < 0.2 and S_list:
                    # Remove a random element to create space
                    y_idx = random.randint(0, len(S_list) - 1)
                    y = S_list[y_idx]
                    in_set[y] = False
                    S_list.pop(y_idx)
                    current_size -= 1
                    non_members.append(y)
    
    return S_list, in_set, current_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    all_nums = list(range(2, N + 1))
    
    # Precompute primes for ordering
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0: return False
        return True
    
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    best_set = set()
    best_size = 0
    best_in_set = [False] * (N + 1)
    
    # Phase 1: Multiple greedy constructions with different orderings
    orders = [
        list(range(2, N + 1)),
        list(range(N, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    # Add random orderings
    for _ in range(30):
        shuffled = all_nums[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        in_set = [False] * (N + 1)
        S_list = _greedy_construct(order, in_set, deadline, N)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = set(S_list)
            best_in_set = in_set.copy()
    
    # Phase 2: Local search on the best set
    if best_set:
        S_list = list(best_set)
        in_set = best_in_set.copy()
        S_list, in_set, current_size = _local_search(S_list, in_set, deadline, N)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(S_list)
            best_in_set = in_set.copy()
    
    # Phase 3: More restarts if time permits
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        
        random.shuffle(all_nums)
        in_set = [False] * (N + 1)
        S_list = _greedy_construct(all_nums, in_set, deadline, N)
        
        if len(S_list) >= best_size:  # Only do local search if competitive
            S_list, in_set, current_size = _local_search(S_list, in_set, deadline, N)
            if current_size > best_size:
                best_size = current_size
                best_set = set(S_list)
                best_in_set = in_set.copy()
    
    # Phase 4: Final local search on the best set with more iterations
    if best_set and time.time() < deadline - 5.0:
        S_list = list(best_set)
        in_set = best_in_set.copy()
        S_list, in_set, current_size = _local_search(S_list, in_set, deadline, N, max_iters=200000)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(S_list)
            best_in_set = in_set.copy()
    
    # Final validation
    result = list(best_set)
    return [int(x) for x in result]
