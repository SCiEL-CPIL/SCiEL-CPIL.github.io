import time
import random
from fractions import Fraction

# Precompute 3-AP conflicts
# For each x in 1..256, store list of (y, z) such that {x, y, z} is a 3-AP
# y < z, and x, y, z are distinct.
# We need to check if adding x to S creates a 3-AP.
# This happens if there exist y, z in S such that {x, y, z} is a 3-AP.
# So for each x, we precompute all pairs (y, z) in [1, 256] such that {x, y, z} is a 3-AP.

def _precompute_conflicts(n=256):
    conflicts = {i: [] for i in range(1, n + 1)}
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            # Triple is (a, b, c)
            # For a, conflicts are (b, c)
            conflicts[a].append((b, c))
            # For b, conflicts are (a, c)
            conflicts[b].append((a, c))
            # For c, conflicts are (a, b)
            conflicts[c].append((a, b))
    return conflicts

CONFLICTS = _precompute_conflicts(256)
N = 256

def _can_add(S_bitmask, x):
    """Check if x can be added to S (represented as bitmask) without creating a 3-AP."""
    # x is not in S
    # Check all pairs (y, z) in CONFLICTS[x]
    # If y and z are both in S, then adding x creates a 3-AP.
    for y, z in CONFLICTS[x]:
        if (S_bitmask & (1 << (y - 1))) and (S_bitmask & (1 << (z - 1))):
            return False
    return True

def _score_float(S_list):
    return sum(1.0 / x for x in S_list)

def _bitmask_to_list(S_bitmask):
    return [i + 1 for i in range(N) if (S_bitmask & (1 << i))]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Greedy Initialization
    # Sort candidates by reciprocal weight (smallest x first)
    candidates = list(range(1, N + 1))
    
    S_bitmask = 0
    S_list = []
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if _can_add(S_bitmask, x):
            S_bitmask |= (1 << (x - 1))
            S_list.append(x)
    
    if not S_list:
        return [1]
    
    current_score = _score_float(S_list)
    best_S_bitmask = S_bitmask
    best_score = current_score
    
    # Local Search
    # We will alternate between:
    # 1. Try to add an element
    # 2. Try to swap an element
    # 3. If stuck, kick (remove some, re-add greedily)
    
    no_improve_count = 0
    max_no_improve = 20 # Threshold for kick
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Add
        best_add = None
        best_add_score = current_score
        for x in range(1, N + 1):
            if S_bitmask & (1 << (x - 1)):
                continue
            if _can_add(S_bitmask, x):
                new_score = current_score + 1.0 / x
                if new_score > best_add_score:
                    best_add_score = new_score
                    best_add = x
        
        if best_add is not None:
            S_bitmask |= (1 << (best_add - 1))
            S_list.append(best_add)
            current_score = best_add_score
            improved = True
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_S_bitmask = S_bitmask
            continue
        
        # 2. Try Swap
        best_swap_remove = None
        best_swap_add = None
        best_swap_score = current_score
        
        # Optimize: iterate over elements in S and candidates not in S
        # This is O(|S| * (N-|S|)) which is ~ 128 * 128 = 16k checks per pass.
        # Each check is O(conflicts[x]) ~ 100-200 ops.
        # Total ~ 2-3 million ops per pass. Very fast.
        
        S_indices = [i for i in range(N) if (S_bitmask & (1 << i))]
        NotS_indices = [i for i in range(N) if not (S_bitmask & (1 << i))]
        
        for i in S_indices:
            x_remove = i + 1
            # Temporarily remove x_remove
            S_bitmask_no_x = S_bitmask & ~(1 << i)
            score_no_x = current_score - 1.0 / x_remove
            
            for j in NotS_indices:
                x_add = j + 1
                if _can_add(S_bitmask_no_x, x_add):
                    new_score = score_no_x + 1.0 / x_add
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_remove = x_remove
                        best_swap_add = x_add
        
        if best_swap_remove is not None and best_swap_add is not None:
            # Perform swap
            S_bitmask &= ~(1 << (best_swap_remove - 1))
            S_bitmask |= (1 << (best_swap_add - 1))
            S_list.remove(best_swap_remove)
            S_list.append(best_swap_add)
            current_score = best_swap_score
            improved = True
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_S_bitmask = S_bitmask
            continue
        
        # 3. No improvement, try Kick
        no_improve_count += 1
        if no_improve_count >= max_no_improve:
            # Kick: Remove 2-3 random elements, then greedily add best
            num_remove = 2
            if len(S_list) > num_remove:
                to_remove = random.sample(S_list, num_remove)
                for x in to_remove:
                    S_bitmask &= ~(1 << (x - 1))
                    S_list.remove(x)
                current_score = _score_float(S_list)
                
                # Greedily add back
                for x in range(1, N + 1):
                    if S_bitmask & (1 << (x - 1)):
                        continue
                    if _can_add(S_bitmask, x):
                        # Check if adding improves score (it always does if x>0)
                        # But we might want to be selective? No, just fill.
                        S_bitmask |= (1 << (x - 1))
                        S_list.append(x)
                        current_score += 1.0 / x
                
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_S_bitmask = S_bitmask

    # Final validation and return
    final_list = _bitmask_to_list(best_S_bitmask)
    if not final_list:
        final_list = [1]
        
    # Verify validity just in case
    s_set = set(final_list)
    valid = True
    for a in s_set:
        for b in s_set:
            if a >= b:
                continue
            c = 2 * b - a
            if c in s_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to simple greedy if something went wrong
        S_bitmask = 0
        S_list = []
        for x in range(1, N + 1):
            if _can_add(S_bitmask, x):
                S_bitmask |= (1 << (x - 1))
                S_list.append(x)
        final_list = S_list

    return sorted(final_list)
