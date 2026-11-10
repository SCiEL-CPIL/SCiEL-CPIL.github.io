"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit
    n = 512
    
    # Precompute all 4-APs to speed up checking? 
    # Actually, checking on the fly is O(n^2) per move if done naively.
    # With n=512, max d is ~170. 
    # Number of 4-APs is roughly sum_{d=1}^{n/3} (n-3d) ~ n^2/6 ~ 40k.
    # We can precompute for each index x, the list of other indices that form a 4-AP with x.
    # Or better, precompute for each pair (a, b) if they are part of a 4-AP.
    # Let's stick to on-the-fly checking but optimized.
    
    # Optimization: Precompute "conflict lists" is hard because it's 4-wise.
    # Instead, we check if adding x creates a 4-AP.
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # If we add x, we check if there exist d such that:
    # 1. x is 4th: x-3d, x-2d, x-d in set
    # 2. x is 3rd: x-2d, x-d, x+d in set
    # 3. x is 2nd: x-d, x, x+d, x+2d in set
    # 4. x is 1st: x, x+d, x+2d, x+3d in set
    
    def is_valid_4ap_free(in_set):
        """Check if the current set is 4-AP free. Returns True if valid."""
        # Iterate through all possible 4-APs
        # a, a+d, a+2d, a+3d <= n
        for a in range(1, n + 1):
            if not in_set[a]:
                continue
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if in_set[b] and in_set[c] and in_set[e]:
                    return False
        return True

    def can_add(x, in_set):
        """Check if x can be added to in_set without creating a 4-AP."""
        # x as 4th term: x-3d, x-2d, x-d
        for d in range(1, (x // 3) + 1):
            if in_set[x-d] and in_set[x-2*d] and in_set[x-3*d]:
                return False
        # x as 1st term: x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                return False
        # x as 2nd term: x-d, x+d, x+2d
        max_d = min(x, (n - x) // 2)
        for d in range(1, max_d + 1):
            if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                return False
        # x as 3rd term: x-2d, x-d, x+d
        max_d = min(x // 2, n - x)
        for d in range(1, max_d + 1):
            if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                return False
        return True

    # 1. Initialize with a greedy construction to get a good starting point
    # Use base-3 digit sum ordering which tends to avoid small 3-APs, hence 4-APs
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s

    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    in_set = [False] * (n + 1)
    current_set = []
    
    for x in order:
        if time.time() > deadline - 10:
            break
        if can_add(x, in_set):
            in_set[x] = True
            current_set.append(x)
    
    best_size = len(current_set)
    best_set = list(current_set)
    
    # 2. Simulated Annealing
    # State: in_set (list of bools), current_set (list of ints)
    # Move 1: Add a random element not in set (if valid)
    # Move 2: Remove a random element from set, then try to add a different one?
    # Actually, standard SA on this problem is tricky because the constraint is global.
    # Let's use a "repair" based local search or simple swaps.
    
    # Better approach: Local Search with Random Restarts
    # Since greedy gets stuck, let's try to "shake" the solution.
    # We will try to improve the greedy result by trying to add elements that were rejected
    # by removing conflicting elements.
    
    # Let's implement a simple "kick" and "recover" strategy.
    # Kick: Remove K random elements from the set.
    # Recover: Greedily add elements back (including the removed ones) in a random order.
    
    random.seed(42)
    current_set = list(best_set)
    in_set = [False] * (n + 1)
    for x in current_set:
        in_set[x] = True
        
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Create a perturbed set
        perturbed_set = list(current_set)
        perturbed_in_set = [False] * (n + 1)
        for x in perturbed_set:
            perturbed_in_set[x] = True
            
        # Remove 5-10 random elements
        num_remove = random.randint(5, 10)
        removed = []
        for _ in range(num_remove):
            if perturbed_set:
                idx = random.randint(0, len(perturbed_set) - 1)
                x = perturbed_set.pop(idx)
                perturbed_in_set[x] = False
                removed.append(x)
        
        # Now try to add elements back and new elements
        # Combine removed and all missing elements
        missing = [x for x in range(1, n + 1) if not perturbed_in_set[x]]
        random.shuffle(missing)
        
        # Greedily add
        for x in missing:
            if time.time() > deadline - 5:
                break
            if can_add(x, perturbed_in_set):
                perturbed_in_set[x] = True
                perturbed_set.append(x)
        
        if len(perturbed_set) > best_size:
            best_size = len(perturbed_set)
            best_set = list(perturbed_set)
            current_set = list(perturbed_set)
            in_set = list(perturbed_in_set)
            
    # Final cleanup: ensure best_set is sorted and valid
    # The can_add checks during construction guarantee validity for the added elements,
    # but we should verify the final set just in case of any logic errors.
    # However, verification is O(n^2) which is fast enough for n=512.
    
    # Re-verify best_set
    final_in_set = [False] * (n + 1)
    for x in best_set:
        final_in_set[x] = True
        
    # Quick check
    valid = True
    for a in range(1, n + 1):
        if not final_in_set[a]:
            continue
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            if final_in_set[a+d] and final_in_set[a+2*d] and final_in_set[a+3*d]:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a simple valid set if something went wrong (should not happen)
        # Just return the base-3 greedy result which is known to be valid
        fallback_in_set = [False] * (n + 1)
        fallback_set = []
        for x in order:
            if can_add(x, fallback_in_set):
                fallback_in_set[x] = True
                fallback_set.append(x)
        return sorted(fallback_set)
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
