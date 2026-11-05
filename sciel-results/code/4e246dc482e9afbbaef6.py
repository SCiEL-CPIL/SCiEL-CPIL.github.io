import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Bitset representation: bit i is 1 if i is in A (using 1-indexed, so bit 0 is unused or represents 1? 
    # Let's use bit i for number i. So range(1, N+1).
    # Python int can handle this easily.
    A_mask = 0
    A_set = set()
    
    # Helper to check if adding x is valid (no 4-AP)
    def is_valid_add(x, mask, aset):
        # Check 4 conditions for 4-APs containing x
        # 1. x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (mask >> (x + d)) & 1 and (mask >> (x + 2*d)) & 1 and (mask >> (x + 3*d)) & 1:
                return False
        # 2. x-d, x, x+d, x+2d
        for d in range(1, min(x, (N - x) // 2 + 1) + 1):
            if (mask >> (x - d)) & 1 and (mask >> (x + d)) & 1 and (mask >> (x + 2*d)) & 1:
                return False
        # 3. x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, (N - x) + 1) + 1):
            if (mask >> (x - 2*d)) & 1 and (mask >> (x - d)) & 1 and (mask >> (x + d)) & 1:
                return False
        # 4. x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (mask >> (x - 3*d)) & 1 and (mask >> (x - 2*d)) & 1 and (mask >> (x - d)) & 1:
                return False
        return True

    # Helper to calculate gain in 3-APs if x is added
    def gain_3aps(x, mask, aset):
        gain = 0
        # 1. x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if (mask >> (x + d)) & 1 and (mask >> (x + 2*d)) & 1:
                gain += 1
        # 2. x-d, x, x+d
        for d in range(1, min(x, N - x) + 1):
            if (mask >> (x - d)) & 1 and (mask >> (x + d)) & 1:
                gain += 1
        # 3. x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if (mask >> (x - 2*d)) & 1 and (mask >> (x - d)) & 1:
                gain += 1
        return gain

    # Initialize with greedy
    for x in range(1, N + 1):
        if is_valid_add(x, A_mask, A_set):
            A_mask |= (1 << x)
            A_set.add(x)
    
    current_score = 0
    # Calculate initial score
    # We can just count all 3-APs in the initial set
    def count_3aps(mask):
        count = 0
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2*d + 1):
                if (mask >> a) & 1 and (mask >> (a+d)) & 1 and (mask >> (a+2*d)) & 1:
                    count += 1
        return count

    current_score = count_3aps(A_mask)
    best_score = current_score
    best_mask = A_mask
    best_set = A_set

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if time.time() > deadline: break
            if (A_mask >> x) & 1:
                continue
            if is_valid_add(x, A_mask, A_set):
                g = gain_3aps(x, A_mask, A_set)
                if g > 0:
                    # Add x
                    A_mask |= (1 << x)
                    A_set.add(x)
                    current_score += g
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = A_mask
                        best_set = set(A_set)
                    # Restart scan? Or continue? 
                    # Adding x might enable other additions. 
                    # Let's restart the inner loop to find the next best addition.
                    break
        
        if improved:
            continue
            
        # Try swaps: remove y, add x
        # This is O(N^2) per iteration. N=256, so ~65k checks.
        # Each check involves is_valid_add for the new set.
        # We can optimize by only trying swaps if we are stuck.
        
        # Simpler: Random restarts or Simulated Annealing?
        # Let's try a few random swaps if stuck.
        swaps_made = 0
        for _ in range(1000): # Limit swap attempts per "stuck" phase
            if time.time() > deadline: break
            if not A_set: break
            y = random.choice(list(A_set))
            # Find a candidate x not in A
            candidates = [x for x in range(1, N+1) if x not in A_set]
            if not candidates: break
            x = random.choice(candidates)
            
            # Check if A \ {y} U {x} is valid
            # We need to check 4-APs in the new set.
            # This is expensive. 
            # Alternative: Check if removing y makes x valid, and adding x doesn't create 4-APs with remaining.
            
            # Let's implement a specific check for swap validity
            # Remove y temporarily
            A_mask_tmp = A_mask & ~(1 << y)
            A_set_tmp = set(A_set)
            A_set_tmp.remove(y)
            
            if is_valid_add(x, A_mask_tmp, A_set_tmp):
                # Calculate score change
                # Remove y: lose 3-APs containing y
                # Add x: gain 3-APs containing x in new set
                
                # This is complex to calculate delta quickly without recounting.
                # Given N=256, recounting is O(N^2) ~ 65k ops. 
                # 1000 swaps * 65k = 65M ops. Might be too slow in Python.
                # Let's limit swaps or optimize.
                
                # Actually, let's just do a simple local search with "add" only first.
                # If "add" only gets stuck, we might miss better configurations.
                # But "add" only is monotonic in score? No, adding x increases score by g>0.
                # So "add" only will always increase score until no valid x with g>0 exists.
                # This finds a local maximum where no single addition is possible or beneficial.
                
                pass # Skip complex swap logic for now to ensure speed and correctness.
                # We can add a simple "remove and re-add" heuristic later if needed.

        # If we couldn't add anything, try a random restart?
        # Or accept the result.
        
    return sorted(best_set)
