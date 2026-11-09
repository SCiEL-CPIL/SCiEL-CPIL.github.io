"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    n = 256
    deadline = time.time() + 82.0  # Safety margin
    
    def is_valid_set(S):
        """Check if set S is valid: for all a<b<=c in S, a does not divide b+c."""
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def is_valid_addition(S, x):
        """Check if adding x to set S keeps the set valid."""
        # x must not be in S
        if x in S:
            return False
        
        S_list = sorted(S)
        
        # Check 1: x as 'a' - x must not divide b+c for any b,c in S with x < b <= c
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Check 2: x as 'b' - for any a in S with a < x, a must not divide x+c for any c in S with c >= x
        for i in range(len(S_list)):
            a = S_list[i]
            if a >= x:
                break
            for k in range(len(S_list)):
                c = S_list[k]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Check 3: x as 'c' - for any a,b in S with a < b <= x, a must not divide b+x
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True

    # Precompute divisor counts for prioritization
    def count_divisors(x):
        count = 0
        for d in range(1, int(math.isqrt(x)) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count

    # Start with the maximal safe tail
    start_threshold = 171
    current_set = set(range(start_threshold, n + 1))
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Phase 1: Greedy addition with divisor-count prioritization
    # Numbers with fewer divisors are less likely to create conflicts
    candidates = list(range(start_threshold - 1, 0, -1))
    # Sort by divisor count ascending (fewer divisors first), then by value descending
    candidates.sort(key=lambda x: (count_divisors(x), -x))
    
    for x in candidates:
        if time.time() > deadline - 10:
            break
        if is_valid_addition(current_set, x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
    
    # Phase 2: Multiple random restarts with different orderings
    random.seed(42)
    
    for restart in range(15):
        if time.time() > deadline - 10:
            break
            
        # Reset to initial tail
        current_set = set(range(start_threshold, n + 1))
        
        # Create shuffled candidates with some bias toward low-divisor numbers
        candidates = list(range(start_threshold - 1, 0, -1))
        
        # Mix: 50% random, 50% sorted by divisor count
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 10:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Phase 3: Local search - try to swap elements to improve
    if time.time() < deadline - 5:
        current_set = set(best_set)
        S_list = sorted(current_set)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i, elem in enumerate(S_list):
                if time.time() > deadline - 3:
                    break
                temp_set = set(current_set)
                temp_set.discard(elem)
                for x in range(1, n + 1):
                    if x in temp_set:
                        continue
                    if is_valid_addition(temp_set, x):
                        if x != elem:
                            current_set = temp_set
                            current_set.add(x)
                            S_list = sorted(current_set)
                            improved = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                            break
    
    # Final validation
    if not is_valid_set(best_set):
        # Fallback to known valid set
        best_set = set(range(start_threshold, n + 1))
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
