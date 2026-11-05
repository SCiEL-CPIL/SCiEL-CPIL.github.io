"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _check_valid(S):
    """Check if set S satisfies the property: for all a<b<=c in S, a does not divide b+c."""
    S = sorted(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            for k in range(j, n):
                c = S[k]
                if a <= b and b <= c:
                    if (b + c) % a == 0:
                        return False
    return True

def _count_violations(S):
    """Count number of violating triples (a,b,c) with a<b<=c in S where a|(b+c)."""
    S = sorted(S)
    n = len(S)
    violations = 0
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            for k in range(j, n):
                c = S[k]
                if (b + c) % a == 0:
                    violations += 1
    return violations

def _violations_with_element(S, new_elem):
    """Count violations that would be introduced by adding new_elem to set S."""
    S = sorted(S)
    new_violations = 0
    for i in range(len(S)):
        a = S[i]
        b = new_elem
        c = new_elem
        if a < b <= c and (b + c) % a == 0:
            new_violations += 1
        # a = new_elem as the smallest
        if new_elem < a:
            for j in range(len(S)):
                b = S[j]
                for k in range(j, len(S)):
                    c = S[k]
                    if new_elem < b <= c and (b + c) % new_elem == 0:
                        new_violations += 1
    return new_violations

def _fast_violation_check(S, candidate):
    """Check if adding candidate to S creates any violation. Returns True if valid."""
    S = sorted(S)
    # Check all triples involving candidate
    for i in range(len(S)):
        a = S[i]
        # Case 1: candidate is b or c, a is from S
        if a < candidate:
            # candidate as b, c >= b
            for j in range(len(S)):
                c = S[j]
                if candidate <= c and (candidate + c) % a == 0:
                    return False
                # candidate as c, b from S with a < b <= candidate
                if a < c <= candidate and (c + candidate) % a == 0:
                    return False
        # Case 2: candidate is a (smallest)
        if candidate < a:
            for j in range(len(S)):
                b = S[j]
                if candidate < b:
                    for k in range(j, len(S)):
                        c = S[k]
                        if b <= c and (b + c) % candidate == 0:
                            return False
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    best_set = list(range(192, 257))  # Initial: upper tail
    best_size = len(best_set)
    
    # Phase 1: Try to add elements from below the current minimum
    # Work with a set for O(1) membership
    current = set(best_set)
    
    # Try adding elements from 1 to 191
    for candidate in range(1, 192):
        if time.time() > deadline - 5.0:
            break
        if candidate not in current:
            # Check if adding this element creates violations
            if _fast_violation_check(current, candidate):
                current.add(candidate)
    
    best_set = sorted(current)
    best_size = len(best_set)
    
    # Phase 2: Local search - try swapping elements
    # Try to replace larger elements with smaller ones if it doesn't reduce size
    # Or try to add more elements by removing some that cause conflicts
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        current_list = sorted(current)
        
        # Try adding each missing element
        for candidate in range(1, 257):
            if time.time() > deadline - 5.0:
                break
            if candidate in current:
                continue
            
            # Check if we can add this element
            if _fast_violation_check(current, candidate):
                current.add(candidate)
                improved = True
                break  # Restart scan after modification
        
        # Try removing elements that cause many violations, then re-adding others
        # This is more complex; for now, just try additions
    
    # Phase 3: Random restarts with different initial sets
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Create a random initial set biased toward larger numbers
        init_set = set()
        for num in range(1, 257):
            # Higher probability for larger numbers
            prob = num / 256.0
            if random.random() < prob:
                init_set.add(num)
        
        # Try to add more elements greedily
        for candidate in range(1, 257):
            if time.time() > deadline - 5.0:
                break
            if candidate not in init_set:
                if _fast_violation_check(init_set, candidate):
                    init_set.add(candidate)
        
        if len(init_set) > best_size:
            best_set = sorted(init_set)
            best_size = len(best_set)
            current = init_set
    
    # Phase 4: Try to improve the best set found
    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for candidate in range(1, 257):
            if time.time() > deadline - 3.0:
                break
            if candidate in current:
                continue
            if _fast_violation_check(current, candidate):
                current.add(candidate)
                improved = True
    
    if len(current) > best_size:
        best_set = sorted(current)
        best_size = len(best_set)
    
    return best_set

# EVOLVE-BLOCK-END
