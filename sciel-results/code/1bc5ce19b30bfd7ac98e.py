import itertools
import time
import random

def is_sidon(subset):
    """
    Check if a subset is a Sidon set (all pair sums a+b with a<=b are unique).
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """
    Compute the size of the largest Sidon subset of S.
    S is a list of distinct integers.
    """
    # Check from largest possible size down to 2
    # Size 1 is always Sidon. Size 2 is always Sidon for distinct ints.
    # We know for N=12, min max size is likely 3 or 4.
    for k in range(len(S), 1, -1):
        for combo in itertools.combinations(S, k):
            if is_sidon(combo):
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    # Initial candidate: The seed
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    # If we already found 3, it's likely the global minimum.
    # But let's run a bit to see if we can confirm or find 2 (impossible) or just verify 3 is stable.
    # Actually, if best_val is 3, we can return immediately if we are confident.
    # Let's assume 3 is the target.
    
    # Search Loop
    # We will use a mix of local search from the current best and random restarts.
    
    while time.time() < deadline:
        # Choose a starting point for local search
        # 50% chance to start from best, 50% from a new random dense set
        if random.random() < 0.5 and best_set:
            current_set = best_set.copy()
        else:
            # Generate a random dense set
            # Pick a base and offset
            base = random.randint(0, 20)
            current_set = []
            while len(current_set) < 12:
                # Pick numbers in a small window to ensure density
                val = base + random.randint(0, 40)
                if val <= 128 and val not in current_set:
                    current_set.append(val)
        
        curr_val = max_sidon_size(current_set)
        
        # Local Search (Hill Climbing)
        # Try to reduce curr_val
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try multiple random moves per iteration
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                idx = random.randint(0, 11)
                # Bias new value to be close to current value to maintain density
                # Or just random in [0, 128]
                # Let's try a mix: 50% close, 50% random
                if random.random() < 0.5:
                    new_val_int = max(0, min(128, current_set[idx] + random.randint(-10, 10)))
                else:
                    new_val_int = random.randint(0, 128)
                
                if new_val_int in current_set:
                    continue
                
                old_val_int = current_set[idx]
                current_set[idx] = new_val_int
                
                new_val = max_sidon_size(current_set)
                
                if new_val < curr_val:
                    curr_val = new_val
                    improved = True
                    # If we hit 3, we found the theoretical min (likely)
                    # We can update best and break local search
                elif new_val == curr_val:
                    # Accept move to explore plateau
                    pass
                else:
                    # Revert
                    current_set[idx] = old_val_int
                
                # Update global best if improved
                if curr_val < best_val:
                    best_val = curr_val
                    best_set = current_set.copy()
                    # If we found 3, it's the best possible for distinct integers > 2
                    # We can exit early to save time, or continue to be sure.
                    # Given the budget, continuing is safe but unnecessary if 3 is min.
                    # Let's break out of local search if we hit 3.
                    if best_val == 3:
                        improved = False
                        break
            
            # If we found 3, we can break the outer loop too?
            # It's highly unlikely to find 2.
            # Let's allow a few more iterations to confirm stability, but 3 is the target.
            if best_val == 3:
                # We can return 3 immediately if we are confident.
                # But let's just break the outer loop to avoid wasting time.
                break

    return best_set
