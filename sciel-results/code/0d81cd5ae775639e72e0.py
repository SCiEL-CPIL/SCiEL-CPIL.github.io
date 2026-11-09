import time
import random
import sys

# EVOLVE-BLOCK-START

def _check_validity(s):
    """Check if set s has no three distinct elements a,b,c with 1/a = 1/b + 1/c."""
    s_set = set(s)
    # For each a in s, check all divisors d of a^2, d != a
    # b = a+d, c = a+a^2/d
    # Need b, c in s_set, b != c, b != a, c != a
    for a in s_set:
        a2 = a * a
        # Find all divisors d of a^2 where d != a
        # d must be such that b = a+d is in s_set and c = a+a^2/d is in s_set
        # and b != c (i.e., d != a^2/d, i.e., d*d != a^2, i.e., d != a)
        # Iterate over possible d values
        # d ranges: b = a+d must be in s_set, so d = b-a for b in s_set, b > a
        # But more efficiently, iterate divisors of a^2
        # For a up to 180, a^2 up to 32400, divisors are manageable
        # Get divisors of a^2
        divs = []
        for i in range(1, int(a2**0.5) + 1):
            if a2 % i == 0:
                divs.append(i)
                if i != a2 // i:
                    divs.append(a2 // i)
        for d in divs:
            if d == a:
                continue
            b = a + d
            c = a + a2 // d
            if b == c:
                continue
            if b in s_set and c in s_set:
                return False
    return True

def _fast_check_validity(s):
    """Faster validity check using the divisor relationship."""
    s_set = set(s)
    for a in s_set:
        a2 = a * a
        # For each b in s_set where b > a, check if c = a + a2/(b-a) is integer and in s_set
        # and b != c
        for b in s_set:
            if b <= a:
                continue
            d = b - a
            if d == 0:
                continue
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c == a:
                continue
            if c == b:
                continue
            if c in s_set:
                return False
    return True

def _greedy_build(start_set, candidates, time_limit):
    """Greedily add candidates to start_set if validity is maintained."""
    current = list(start_set)
    current_set = set(current)
    deadline = time.time() + time_limit
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in current_set:
            continue
        new_set = current_set | {cand}
        if _fast_check_validity(new_set):
            current.append(cand)
            current_set = new_set
    
    return current

def _backtracking_search(elements, current_set, index, best_set, deadline):
    """
    Backtracking search to find a large valid subset.
    elements: sorted list of candidates
    current_set: set of elements chosen so far
    index: current index in elements
    best_set: current best valid set found
    deadline: time limit
    """
    if time.time() > deadline:
        return best_set
    
    if index == len(elements):
        if len(current_set) > len(best_set):
            best_set = set(current_set)
        return best_set
    
    # Pruning: even if we take all remaining, can we beat best?
    if len(current_set) + (len(elements) - index) <= len(best_set):
        return best_set
    
    # Option 1: Include elements[index]
    elem = elements[index]
    new_set = current_set | {elem}
    if _fast_check_validity(new_set):
        best_set = _backtracking_search(elements, new_set, index + 1, best_set, deadline)
    
    # Option 2: Exclude elements[index]
    best_set = _backtracking_search(elements, current_set, index + 1, best_set, deadline)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Strategy: 
    # 1. Start with a known large valid set (e.g., high numbers)
    # 2. Use local search / simulated annealing style swaps to improve
    # 3. Try to find a structure. The condition 1/a = 1/b + 1/c implies a < b, a < c.
    #    So small numbers are more "dangerous" as 'a'.
    #    Large numbers are safer.
    
    # Initial guess: All numbers from 91 to 180 are likely valid.
    # Let's verify and try to expand.
    
    # A better initial set might be derived from a greedy approach on the whole range,
    # but prioritizing numbers that are "harder" to place (small numbers) or "easier" (large numbers).
    
    # Let's try a few different greedy strategies and then refine with local search.
    
    best_set = set()
    
    # Strategy A: Greedy from top down (180 to 1)
    # Strategy B: Greedy from bottom up (1 to 180)
    # Strategy C: Greedy by "degree" of constraint involvement
    
    # For now, let's use the high-number base as a strong start.
    base_set = set(range(91, 181))
    if _fast_check_validity(base_set):
        best_set = base_set
    else:
        # Fallback: find a valid subset of base_set
        for x in range(180, 90, -1):
            if _fast_check_validity(base_set | {x}):
                best_set.add(x)
            else:
                base_set.discard(x)
        best_set = set(base_set)

    # Local Search: Simulated Annealing / Random Swaps
    # Try to swap an element in the set with one not in the set to increase size or maintain size while allowing future additions.
    # Since we want to maximize cardinality, we primarily try to ADD elements.
    # If we can't add, we try to SWAP (remove one, add one) to reach a state where we CAN add more.
    
    # Let's try a simple randomized local search.
    current_set = set(best_set)
    
    # Pre-compute "conflicts" for faster checking?
    # _fast_check_validity is O(N^2) roughly. N=180, so 180*180 = 32400 ops. Very fast.
    
    # We will try to add any element from 1..180 not in current_set.
    # If stuck, try removing a random element and adding a random other.
    
    all_elements = set(range(1, 181))
    
    # Sort candidates by "difficulty" or just random
    # Let's try adding in descending order first (safer)
    # Then ascending.
    
    # Actually, the previous program did this. Let's do something slightly different.
    # Use a "repair" heuristic.
    
    # 1. Start with empty set? No, start with a good seed.
    # Let's try multiple random restarts with greedy construction.
    
    best_size = len(best_set)
    best_result = list(best_set)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random seed for this restart
        random.seed(100 + restart)
        
        # Greedy construction
        # Order: Random permutation of 1..180
        candidates = list(range(1, 181))
        random.shuffle(candidates)
        
        current = set()
        for c in candidates:
            if time.time() > deadline:
                break
            if _fast_check_validity(current | {c}):
                current.add(c)
                
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
            
        # Local search from this state
        # Try to add more
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any missing element
            missing = [x for x in range(1, 181) if x not in current]
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    break
                if _fast_check_validity(current | {m}):
                    current.add(m)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
                    break # Restart loop to check new additions
            
            if not improved:
                # Try swaps: remove one, add one (net 0, but might enable future adds)
                # Or remove one, add two (net +1)
                # Let's try removing a random element and seeing if we can add more.
                if len(current) > 0:
                    to_remove = random.choice(list(current))
                    temp_current = current - {to_remove}
                    
                    # Try to add as many as possible from missing
                    added_any = False
                    # Sort missing by some heuristic? Random for now.
                    missing_after_remove = [x for x in range(1, 181) if x not in temp_current]
                    random.shuffle(missing_after_remove)
                    
                    # Greedy add on temp_current
                    for m in missing_after_remove:
                        if time.time() > deadline:
                            break
                        if _fast_check_validity(temp_current | {m}):
                            temp_current.add(m)
                            added_any = True
                            
                    if len(temp_current) > len(current):
                        current = temp_current
                        improved = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_result = list(current)
                    # Even if size didn't increase, the state might be better for future adds?
                    # For simplicity, only accept if size increases or we are stuck and want to explore.
                    # Let's only accept if size increases to avoid drifting.
    
    return best_result

# EVOLVE-BLOCK-END
