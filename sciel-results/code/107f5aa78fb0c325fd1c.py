"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def is_valid(S, target=200):
        """Check if no subset of S sums to target using bitset DP."""
        if not S:
            return True
        bits = 1  # bit i set means sum i is achievable
        for x in S:
            if x <= target:
                bits |= (bits << x)
        return (bits >> target) & 1 == 0

    # Start with known valid set: {66, 67, ..., 100}, size 35
    S = set(range(66, 101))
    best = S.copy()
    best_size = len(S)
    
    # To allow randomization in swaps, we might shuffle candidates
    # But deterministic is safer for reproducibility if we don't find better.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        outside = set(range(1, 101)) - S
        outside_list = list(outside)
        inside_list = list(S)
        
        # 1. Try adding a single element (increases size by 1)
        added = False
        for x in outside_list:
            if time.time() > deadline:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
                if len(S) > best_size:
                    best = S.copy()
                    best_size = len(S)
                improved = True
                added = True
                break # Restart loop with new S
        
        if added:
            continue
            
        # 2. Try swapping one element for two (increases size by 1)
        # Remove y, add x1, x2
        # Complexity: |S| * |outside|^2. 35 * 65^2 ~ 150k checks.
        # Each check is bitset DP (fast).
        swapped = False
        # To avoid bias, we can iterate through all y
        for y in inside_list:
            if time.time() > deadline:
                break
            S_minus_y = S - {y}
            # Try pairs from outside
            for i in range(len(outside_list)):
                if time.time() > deadline:
                    break
                x1 = outside_list[i]
                for j in range(i+1, len(outside_list)):
                    x2 = outside_list[j]
                    new_S = S_minus_y | {x1, x2}
                    if len(new_S) > len(S):
                        if is_valid(new_S):
                            S = new_S
                            if len(S) > best_size:
                                best = S.copy()
                                best_size = len(S)
                            improved = True
                            swapped = True
                            break
                if swapped:
                    break
            if swapped:
                break
        
        if swapped:
            continue
            
        # 3. Try swapping one element for one element (keeps size, explores space)
        # This is crucial to escape local optima if we are stuck at 35.
        # We perform this only if we haven't improved size recently? 
        # Or just do it occasionally.
        # To be safe and fast, we limit this phase.
        
        # 4. Try adding one element and removing one (size constant)
        # This is the same as swap 1-for-1.
        
        # If we are stuck, we might try a randomized local search for a few steps
        # to see if we can find a path to a larger set.
        
        # To prevent infinite loops of 1-for-1 swaps, we can track visited states
        # or just limit the number of swap iterations.
        # Given the small domain, 1-for-1 swaps might cycle.
        # We'll do a limited number of random swaps if stuck.
        
        # Actually, if 35 is the max, we will loop forever on swaps.
        # We need a termination condition for the "no size improvement" case.
        # Let's say if we do 1000 swaps without size increase, we stop.
        
        # For simplicity and speed, if we didn't increase size in steps 1 and 2,
        # we assume we are at a local maximum.
        # We can try a few random swaps to escape, but it's risky.
        
        # Let's try one more phase: Remove one, add one (randomized)
        # If we find a valid swap, we do it.
        # But to avoid cycles, we should probably stop if no size increase.
        
        # However, it's possible that a sequence of size-preserving swaps leads to a size-increasing move.
        # Example: S -> S' (same size) -> S' + x (size+1).
        
        # Let's do a limited number of such explorations.
        
        # To keep it simple and fast, we will stop if no size increase.
        # This might miss solutions requiring swaps, but 35 is a strong baseline.
        
        # Let's add a random restart or perturbation if stuck?
        # No, let's just break.
        
        break

    result = list(best)
    assert len(result) > 0
    assert is_valid(set(result))
    return result
# EVOLVE-BLOCK-END
