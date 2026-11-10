import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s safety margin for 60s limit

    # Precompute 3-APs
    # ap_pairs[x] = list of (u, v) such that {u, v, x} is a 3-AP
    # Actually, for checking if adding x creates a 3-AP, we need to know 
    # if there are existing pairs in S that complete a 3-AP with x.
    
    # Let's store for each x, the list of pairs (a, b) from S 
    # such that a, b, x form a 3-AP.
    # It's easier to precompute all 3-APs in [1, N].
    
    # ap_conflicts[x] = list of tuples (y, z) such that {x, y, z} is a 3-AP
    # If we have y, z in S, we cannot add x.
    # If we have x, y in S, we cannot add z.
    
    # Let's create a structure to quickly check if a set is 3-AP free.
    # Given a set S, and a candidate x not in S:
    # x creates a 3-AP if there exist y, z in S such that {x, y, z} is a 3-AP.
    
    # Precompute all 3-APs in [1, N]
    # ap_list is a list of sets of size 3
    # But we need to check against current S.
    
    # Better: For each x, store list of (y, z) where y < z and {x, y, z} is 3-AP.
    # Then when adding x, check if any (y, z) in ap_pairs[x] has both y, z in S.
    
    ap_pairs = [[] for _ in range(N + 1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Add to ap_pairs for a, b, c
            # {a, b, c} is the AP.
            # For a: pairs (b, c)
            # For b: pairs (a, c)
            # For c: pairs (a, b)
            ap_pairs[a].append((b, c))
            ap_pairs[b].append((a, c))
            ap_pairs[c].append((a, b))
            
    # Initial Greedy Construction
    S = [False] * (N + 1)
    S[1] = True
    for x in range(2, N + 1):
        # Check if adding x creates a 3-AP
        can_add = True
        for y, z in ap_pairs[x]:
            if S[y] and S[z]:
                can_add = False
                break
        if can_add:
            S[x] = True
            
    def get_score(S):
        return sum(1.0 / i for i in range(1, N + 1) if S[i])
    
    current_score = get_score(S)
    best_S = S[:]
    best_score = current_score
    
    def is_valid_addition(S, x):
        # S is boolean array, x not in S
        for y, z in ap_pairs[x]:
            if S[y] and S[z]:
                return False
        return True
        
    def is_valid_removal_addition(S, x, y):
        # Remove x, add y. x in S, y not in S.
        # We need to check if the new set is 3-AP free.
        # The only new 3-APs could involve y.
        # 3-APs involving y: (y, a, b) where {y, a, b} is 3-AP.
        # Since we are removing x, we must ensure that no 3-AP involving y
        # relies on x being present? No, if a 3-AP involves y and two other elements,
        # those two other elements must be in the new set.
        # The new set is (S \ {x}) U {y}.
        # So for any (a, b) in ap_pairs[y], if a != x and b != x, we need NOT (S[a] and S[b]).
        # If a == x or b == x, then that pair is broken by removing x.
        
        for a, b in ap_pairs[y]:
            if a == x or b == x:
                continue
            if S[a] and S[b]:
                return False
        return True

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding any element
        for x in range(1, N + 1):
            if not S[x]:
                if is_valid_addition(S, x):
                    S[x] = True
                    current_score += 1.0 / x
                    improved = True
                    # Restart loop as set changed significantly? 
                    # Or just continue? Adding one might block others.
                    # Let's break and restart the "try adding" phase to be safe.
                    break
        
        if improved:
            continue
            
        # 2. Try swapping one element
        # To speed up, we can iterate through elements in S and potential additions
        # But N is small (256), so O(N^2) is 65k checks. 
        # Each check is O(degree) in AP graph. Degree is small.
        
        best_swap = None
        best_gain = 0
        
        for x in range(1, N + 1):
            if not S[x]:
                continue
            # Try removing x
            for y in range(1, N + 1):
                if S[y]:
                    continue
                # Check if removing x and adding y is valid
                if is_valid_removal_addition(S, x, y):
                    gain = (1.0 / y) - (1.0 / x)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
        
        if best_swap:
            x, y = best_swap
            S[x] = False
            S[y] = True
            current_score += best_gain
            improved = True
            
        if current_score > best_score:
            best_score = current_score
            best_S = S[:]
            
        # If no improvement, we are stuck in a local optimum.
        # Apply a random perturbation to escape.
        if not improved:
            # Random restart or perturbation
            # Option 1: Randomly remove a few elements and re-greedy
            # Option 2: Simulated annealing step
            
            # Let's try a simple perturbation: 
            # Pick a random element in S, remove it.
            # Then try to add elements greedily.
            
            # Actually, a better perturbation for "max weight independent set" like problems:
            # Remove a random element, and then run the "add if possible" loop.
            
            elements_in_S = [i for i in range(1, N + 1) if S[i]]
            if elements_in_S:
                rem = random.choice(elements_in_S)
                S[rem] = False
                current_score -= 1.0 / rem
                
                # Now try to add elements
                for x in range(1, N + 1):
                    if not S[x]:
                        if is_valid_addition(S, x):
                            S[x] = True
                            current_score += 1.0 / x
                            
                # Then try swaps again
                # ... (This effectively restarts the local search from a perturbed state)
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]

    # Final check and return
    # Ensure best_S is a list of ints, not boolean array, as per "subset of [1,n]"
    # The seed program returns a list.
    return [i for i in range(1, N + 1) if best_S[i]]

# EVOLVE-BLOCK-END
