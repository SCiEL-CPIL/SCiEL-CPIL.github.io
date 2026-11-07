"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def _is_valid(set_bits):
    """Check if the set represented by bits is valid."""
    # Get the list of elements
    elems = [i+1 for i in range(N) if (set_bits >> i) & 1]
    n = len(elems)
    if n < 3:
        return True
    
    # For each pair (a, b) in the set, check if a+b is in the set
    # We need to check: for distinct a, b, c in set, if a+b, a+c, b+c are all in set
    # Equivalently: there do NOT exist distinct a, b, c in S such that 
    # a+b in S, a+c in S, and b+c in S.
    
    # For each pair (a, b) with a < b, let s = a + b.
    # If s is in S, then we need to ensure there's no c in S (c != a, c != b) such that
    # a+c in S AND b+c in S.
    
    # More efficient: For each pair (a, b), a < b, both in S:
    #   s_ab = a + b
    #   if s_ab in S:
    #       Then for any c in S with c != a, c != b:
    #           if (a+c) in S and (b+c) in S: invalid
    #   
    # This is O(n^3) in worst case but n is at most ~64 for large sets.
    # Let's use a set for O(1) lookup.
    
    S = set(elems)
    
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            s_ab = a + b
            if s_ab in S:
                # Check if there exists c in S, c != a, c != b, such that a+c in S and b+c in S
                for k in range(n):
                    c = elems[k]
                    if c == a or c == b:
                        continue
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac in S and s_bc in S:
                        return False
    return True

def _greedy_construct():
    """Greedy construction: add elements in order, skip if it creates a violation."""
    result = []
    result_set = set()
    
    for x in range(1, N+1):
        # Try adding x
        candidate = result_set | {x}
        # Check validity
        if _is_valid_from_set(candidate):
            result.append(x)
            result_set.add(x)
    
    return result

def _is_valid_from_set(S):
    """Check validity of a set S (Python set of ints)."""
    n = len(S)
    if n < 3:
        return True
    
    elems = sorted(S)
    
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            s_ab = a + b
            if s_ab in S:
                for k in range(n):
                    c = elems[k]
                    if c == a or c == b:
                        continue
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac in S and s_bc in S:
                        return False
    return True

def _local_search(start_set, max_steps=5000, time_limit=50.0):
    """Local search to improve the set size."""
    start_time = time.time()
    current = set(start_set)
    current_size = len(current)
    best = set(current)
    best_size = current_size
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Try to add a random element not in current
        available = [x for x in range(1, N+1) if x not in current]
        if available:
            x = random.choice(available)
            candidate = current | {x}
            if _is_valid_from_set(candidate):
                current = candidate
                current_size += 1
                if current_size > best_size:
                    best = set(current)
                    best_size = current_size
        
        # Occasionally try to swap
        if current_size > 0 and step % 3 == 0:
            in_set = list(current)
            x = random.choice(in_set)
            available = [y for y in range(1, N+1) if y not in current]
            if available:
                y = random.choice(available)
                candidate = (current - {x}) | {y}
                if _is_valid_from_set(candidate):
                    current = candidate
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
    
    return list(best)

def construct():
    start_time = time.time()
    
    # Start with greedy construction
    greedy = _greedy_construct()
    
    # Try multiple random restarts with local search
    best_result = greedy
    best_size = len(greedy)
    
    # Try adding random elements in different orders
    for restart in range(50):
        if time.time() - start_time > 85.0:
            break
        
        # Random subset of size around best_size
        random.seed(restart)
        # Start from a random permutation and greedily add
        perm = list(range(1, N+1))
        random.shuffle(perm)
        
        current_set = set()
        for x in perm:
            candidate = current_set | {x}
            if _is_valid_from_set(candidate):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_result = list(current_set)
            best_size = len(current_set)
        
        # Local search from this result
        improved = _local_search(list(current_set), max_steps=2000, time_limit=1.0)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)
    
    # Final local search from best
    if time.time() - start_time < 85.0:
        final = _local_search(best_result, max_steps=5000, time_limit=5.0)
        if len(final) > best_size:
            best_result = final
    
    return list(best_result)
# EVOLVE-BLOCK-END
