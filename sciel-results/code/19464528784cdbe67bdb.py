"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    def is_valid(set_list):
        """Check if the set satisfies the property."""
        s = sorted(set_list)
        for i in range(len(s)):
            a = s[i]
            for j in range(i+1, len(s)):
                b = s[j]
                for k in range(j, len(s)):
                    c = s[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(candidate, current_set):
        """Check if adding candidate to current_set maintains validity."""
        s = sorted(list(current_set) + [candidate])
        # Only need to check triples involving the new element
        # Case 1: candidate is 'a' (smallest in triple)
        # Case 2: candidate is 'b' 
        # Case 3: candidate is 'c'
        
        for i in range(len(s)):
            a = s[i]
            for j in range(i+1, len(s)):
                b = s[j]
                for k in range(j, len(s)):
                    c = s[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Start with upper tail
    # For tail [k, 256], we need to verify it's valid
    # Let's find the smallest k where [k, 256] is valid
    # A sufficient condition: for all a in [k,256], no multiple of a in [2a+2, 512] 
    # is expressible as b+c with b,c in [k,256], a<b<=c
    
    # Start with k=128 (size 129) and try to improve
    current_set = set(range(128, 257))
    
    # Verify initial set is valid
    if not is_valid(list(current_set)):
        # Find a valid tail
        for k in range(256, 0, -1):
            test_set = set(range(k, 257))
            if is_valid(list(test_set)):
                current_set = test_set
                break
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Try to add elements from 1 to 127
    candidates = list(range(1, 128))
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        if can_add(cand, current_set):
            current_set.add(cand)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Try some random restarts with different orderings
    import random
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        random.shuffle(candidates)
        current_set = set(best_set)
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            if cand not in current_set:
                if can_add(cand, current_set):
                    current_set.add(cand)
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
