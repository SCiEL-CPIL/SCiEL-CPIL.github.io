"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    n = 256
    
    # Precompute nothing heavy, just use the set logic.
    
    def get_sums(S):
        """Compute the set of all sums b+c for b,c in S with b<=c."""
        S_list = sorted(S)
        sums = set()
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                sums.add(b + c)
        return sums

    def can_add_small(x, S, sums_S):
        """
        Check if x can be added to S, assuming x < min(S).
        Condition: x must not divide any sum in sums_S.
        """
        # Optimization: if x is large, fewer sums are divisible.
        # Just check all sums.
        for s in sums_S:
            if s % x == 0:
                return False
        return True

    def is_valid_set(S):
        """Full validity check for arbitrary S."""
        S_list = sorted(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j, L):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Phase 1: Find the best starting interval [L, 256]
    best_start_L = 256
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S_interval = set(range(L, n + 1))
        if is_valid_set(S_interval):
            best_start_L = L
            break

    # Start with the largest valid interval
    current_set = set(range(best_start_L, n + 1))
    
    # Phase 2: Greedy Extension
    # Add elements from best_start_L - 1 down to 1
    # Using sums optimization
    sums_S = get_sums(current_set)
    
    for x in range(best_start_L - 1, 0, -1):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if can_add_small(x, current_set, sums_S):
            current_set.add(x)
            # Recompute sums
            sums_S = get_sums(current_set)

    best_set = set(current_set)
    best_size = len(best_set)

    # Phase 3: Local Search (MaxSAT-inspired WalkSAT/Local Improvement)
    # Strategy: Try to increase size by "Add 1" or "Add 2, Remove 1".
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_set = set(best_set)
        sums_S = get_sums(current_set)
        
        # Step 3a: Try to Add 1
        added_any = False
        for x in range(1, n + 1):
            if x in current_set:
                continue
            # If x < min(current_set), use fast check
            if x < min(current_set):
                if can_add_small(x, current_set, sums_S):
                    current_set.add(x)
                    added_any = True
                    break
            else:
                # If x >= min(current_set), do full check (rare in our structure)
                test_set = current_set | {x}
                if is_valid_set(test_set):
                    current_set = test_set
                    added_any = True
                    break
        
        if added_any:
            best_set = current_set
            best_size = len(best_set)
            improved = True
            continue

        # Step 3b: Try to Add 2, Remove 1
        # If we couldn't add 1, we try to swap to enable an addition.
        # Find x not in S such that removing some y allows x to be added.
        # Then see if we can add z.
        
        # To speed up, we only try a subset of x and y
        # Sort candidates x by "potential" (e.g., smaller x might be easier?)
        # Actually, let's just iterate.
        
        candidates_x = [x for x in range(1, n + 1) if x not in current_set]
        random.shuffle(candidates_x) # Add randomness to escape local minima
        
        for x in candidates_x:
            if time.time() > deadline - 5:
                break
            
            # Find y in S such that x can be added to S \ {y}
            # If x < min(S), we need to check sums of S \ {y}
            
            # Optimization: If x < min(S), the sums of S \ {y} are sums_S minus sums involving y.
            # sums_S_y = {b+c | b,c in S, b<=c, b!=y, c!=y}
            # This is expensive to compute for each y.
            
            # Alternative: Just check validity of (S \ {y}) U {x} directly?
            # That's O(N^3) per x. Too slow.
            
            # Let's use the sums_S approximation.
            # If we remove y, the new sums are sums_S minus {y+b | b in S, b>=y} and {y+c | c in S, c>=y} and {2y}.
            # Actually, it's easier to just recompute sums for the candidate set.
            
            # To limit search space, only try y that are "large" or "critical"?
            # Let's try all y but break early if we find a valid z.
            
            # Heuristic: Try removing y that is largest in S? Or smallest?
            # Let's try a few random y's first, then all.
            
            y_candidates = list(current_set)
            random.shuffle(y_candidates)
            
            for y in y_candidates:
                if time.time() > deadline - 5:
                    break
                
                # Construct S_temp = S \ {y} U {x}
                # Check if x can be added to S \ {y}
                S_minus_y = current_set - {y}
                
                # Check if x is valid in S_minus_y
                # If x < min(S_minus_y), use sums
                # Note: min(S_minus_y) might change if y was min(S)
                
                if x < min(S_minus_y):
                    sums_minus_y = get_sums(S_minus_y)
                    if can_add_small(x, S_minus_y, sums_minus_y):
                        # x is valid in S_minus_y
                        S_temp = S_minus_y | {x}
                        # Now try to add z to S_temp
                        sums_temp = get_sums(S_temp)
                        
                        for z in range(1, n + 1):
                            if z in S_temp:
                                continue
                            if z < min(S_temp):
                                if can_add_small(z, S_temp, sums_temp):
                                    # Success!
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                            else:
                                test_set = S_temp | {z}
                                if is_valid_set(test_set):
                                    best_set = test_set
                                    best_size = len(best_set)
                                    improved = True
                                    break
                        
                        if improved:
                            break
                else:
                    # x >= min(S_minus_y), full check
                    test_set = S_minus_y | {x}
                    if is_valid_set(test_set):
                        S_temp = test_set
                        sums_temp = get_sums(S_temp)
                        
                        for z in range(1, n + 1):
                            if z in S_temp:
                                continue
                            if z < min(S_temp):
                                if can_add_small(z, S_temp, sums_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                            else:
                                test_set_z = S_temp | {z}
                                if is_valid_set(test_set_z):
                                    best_set = test_set_z
                                    best_size = len(best_set)
                                    improved = True
                                    break
                        
                        if improved:
                            break

    if best_set is None:
        best_set = list(range(192, 257))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
