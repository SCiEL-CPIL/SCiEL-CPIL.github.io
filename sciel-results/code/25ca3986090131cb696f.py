import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    N = 256
    
    def is_ap(a, b, c):
        # Check if a, b, c form a non-constant 3-AP
        # a, b, c are distinct
        if a == b or b == c or a == c:
            return False
        return 2 * b == a + c or 2 * a == b + c or 2 * c == a + b

    def has_3ap(S):
        # S is a set of ints
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_ap(S_list[i], S_list[j], S_list[k]):
                        return True
        return False

    def can_add(S, x):
        # Check if S U {x} has a 3-AP
        # Only new APs involve x
        for a in S:
            for b in S:
                if a == b:
                    continue
                if is_ap(a, b, x):
                    return False
        return True

    def can_swap(S, i, j):
        # Check if (S \ {i}) U {j} has a 3-AP
        # Remove i, add j
        # New APs must involve j
        # Old APs not involving i are still valid (and were valid before)
        # APs involving i are broken
        # So we just need to check if j forms an AP with any pair in S \ {i}
        S_new = S - {i}
        for a in S_new:
            for b in S_new:
                if a == b:
                    continue
                if is_ap(a, b, j):
                    return False
        return True

    # 1. Greedy Initialization
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = S.copy()
    best_score = sum(1.0 / x for x in S)
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to Add
        # Iterate x from 1 to N, add if valid
        for x in range(1, N + 1):
            if x not in S:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    # Update best if better (it always is)
                    if sum(1.0 / y for y in S) > best_score:
                        best_score = sum(1.0 / y for y in S)
                        best_S = S.copy()
        
        # Try to Swap
        # We want to swap i in S with j not in S such that 1/j > 1/i (j < i)
        # and the new set is valid.
        # Iterate j from 1 to N (smallest first, highest value)
        for j in range(1, N + 1):
            if j in S:
                continue
            
            # Find i in S such that j < i and can_swap(S, i, j)
            # To be efficient, we can just check all i > j
            swapped = False
            for i in range(j + 1, N + 1):
                if i not in S:
                    continue
                if can_swap(S, i, j):
                    # Perform swap
                    S.remove(i)
                    S.add(j)
                    improved = True
                    swapped = True
                    # Update best
                    new_score = sum(1.0 / y for y in S)
                    if new_score > best_score:
                        best_score = new_score
                        best_S = S.copy()
                    break # Restart loop? Or continue? 
                    # Breaking inner loop, continuing outer j loop.
                    # Since S changed, other j's validity might change.
                    # But we are iterating j from small to large.
                    # If we swapped for j, we might have unblocked smaller j's? No, we added j.
                    # We removed i > j. This might allow adding even smaller k < j?
                    # But we already passed k < j in this pass.
                    # So we should probably restart the whole local search loop.
                    # Or just set improved=True and break out to re-evaluate.
                    break
            
            if swapped:
                # Break out of j loop to re-evaluate adds/swaps with new S
                # This is safer to ensure we don't miss interactions
                # But it might be slow.
                # Let's just break the j loop and let the while loop handle it.
                # Actually, if we swap, the set changes. The validity of other swaps/Adds changes.
                # It's best to break and restart the "improved" loop.
                break

    # 3. Final Polish: Try removing elements to allow adding multiple smaller ones?
    # This is complex. The swap logic should handle most cases.
    # Let's do one more pass of pure Add to catch any leftovers.
    for x in range(1, N + 1):
        if x not in S:
            if can_add(S, x):
                S.add(x)
                new_score = sum(1.0 / y for y in S)
                if new_score > best_score:
                    best_score = new_score
                    best_S = S.copy()

    return list(best_S)

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
