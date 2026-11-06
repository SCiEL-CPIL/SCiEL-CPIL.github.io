"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid_set(S, n=256):
    """Check if S satisfies the property: for all a<b<=c in S, a does not divide b+c."""
    S_list = sorted(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            for k in range(j, m):
                c = S_list[k]
                if (b + c) % a == 0:
                    return False
    return True

def _score(S):
    """Return len(S) if valid, 0 otherwise."""
    if len(S) == 0:
        return 0
    if not _is_valid_set(S):
        return 0
    return len(S)

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin before 90s limit
    n = 256
    
    # Start with the known good interval {171, ..., 256}
    S = set(range(171, 257))
    best_S = set(S)
    best_score = _score(S)
    
    # Try to add smaller elements one by one, in greedy fashion
    # For each candidate a from 170 down to 1, check if we can add it
    # If adding a causes conflicts, try to remove conflicting elements if it's worth it
    
    # Greedy addition: try to add elements from 170 down to 1
    # For each a, check if S ∪ {a} is valid. If not, see if we can fix by removing some elements.
    # Simple approach: just try to add a if S ∪ {a} is valid.
    
    # But we can do better: for each a, find the set of conflicting pairs and see if removing them is net positive.
    
    # Let's do a more systematic greedy:
    # For a from 170 down to 1:
    #   Check if S ∪ {a} is valid
    #   If yes, add a
    #   If no, find all (b,c) in S such that a | (b+c), and check if removing them (and adding a) is beneficial
    
    # First pass: simple greedy add
    for a in range(170, 0, -1):
        if time.time() > deadline - 10.0:
            break
        S_test = S | {a}
        if _is_valid_set(S_test, n):
            S = S_test
            if _score(S) > best_score:
                best_score = _score(S)
                best_S = set(S)
    
    # Now try local search: for each element in S, try removing it and adding a smaller element
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        S_list = sorted(S)
        for i, elem in enumerate(S_list):
            if time.time() > deadline - 5.0:
                break
            # Try removing elem
            S_without = S - {elem}
            # Try adding each candidate from 1 to elem-1
            for a in range(1, elem):
                if time.time() > deadline - 5.0:
                    break
                S_new = S_without | {a}
                if _is_valid_set(S_new, n):
                    S = S_new
                    if _score(S) > best_score:
                        best_score = _score(S)
                        best_S = set(S)
                    improved = True
                    break  # Restart loop since S changed
    
    # Final verification
    if not _is_valid_set(best_S, n):
        best_S = set(range(171, 257))
        best_score = _score(best_S)
    
    return list(best_S)

# EVOLVE-BLOCK-END
