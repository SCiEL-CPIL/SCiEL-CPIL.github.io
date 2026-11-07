import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    
    Strategy:
    1. Start with all integers in [129, 256] (size 128).
    2. Greedily remove the most-conflicted element until valid.
    3. Then perform local search with 2-for-1 swaps (remove one, try to add back
       removed elements) to escape local optima.
    
    This hybrid approach should yield a larger valid set than pure greedy removal.
    """
    deadline = time.time() + 50  # 50 second safety margin
    
    # Start with all numbers in [129, 256]
    S = list(range(129, 257))
    S_set = set(S)
    
    def count_violations(S, S_set):
        """Count violations and track which elements are involved."""
        conflict_count = {x: 0 for x in S}
        
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if a == b:
                    continue
                g = math.gcd(a, b)
                a_prime = a // g
                if a_prime == 1:
                    continue
                # Find c in S, c != a, c != b, c % a_prime == 0
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % a_prime == 0:
                        conflict_count[a] += 1
                        conflict_count[b] += 1
                        conflict_count[c] += 1
        
        return conflict_count
    
    def is_valid(S, S_set):
        """Check if S is valid (no violations)."""
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if a == b:
                    continue
                g = math.gcd(a, b)
                a_prime = a // g
                if a_prime == 1:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % a_prime == 0:
                        return False
        return True
    
    # Phase 1: Greedy removal
    while time.time() < deadline:
        conflict_count = count_violations(S, S_set)
        
        max_conflicts = max(conflict_count.values()) if conflict_count else 0
        
        if max_conflicts == 0:
            break
        
        # Remove the element with the most conflicts
        worst_element = None
        for x in S:
            if conflict_count[x] == max_conflicts:
                worst_element = x
                break
        
        if worst_element is not None:
            S.remove(worst_element)
            S_set.remove(worst_element)
        else:
            break
    
    # Phase 2: Local search with swaps
    removed = [x for x in range(129, 257) if x not in S_set]
    random.shuffle(removed)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for x in removed[:]:
            if x in S_set:
                continue
            
            # Try to add x back by removing one element from S
            for y in S[:]:
                if time.time() > deadline:
                    break
                
                # Check if removing y and adding x gives a valid set
                S_test = [z for z in S if z != y]
                S_test.append(x)
                S_test_set = set(S_test)
                
                if is_valid(S_test, S_test_set):
                    # Accept the swap
                    S = S_test
                    S_set = S_test_set
                    removed.remove(x)
                    removed.append(y)
                    random.shuffle(removed)
                    improved = True
                    break
    
    # Final verification
    if not is_valid(S, S_set):
        # If somehow invalid, fall back to greedy removal again
        while not is_valid(S, S_set) and time.time() < deadline:
            conflict_count = count_violations(S, S_set)
            max_conflicts = max(conflict_count.values()) if conflict_count else 0
            if max_conflicts == 0:
                break
            worst_element = None
            for x in S:
                if conflict_count[x] == max_conflicts:
                    worst_element = x
                    break
            if worst_element is not None:
                S.remove(worst_element)
                S_set.remove(worst_element)
            else:
                break
    
    return S
# EVOLVE-BLOCK-END
