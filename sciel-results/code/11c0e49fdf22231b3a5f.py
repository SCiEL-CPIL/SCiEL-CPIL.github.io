import time
import random

N = 256
DEADLINE_MARGIN = 5.0

def _precompute_aps():
    """
    Precompute:
    - all_4aps: list of tuples (a, a+d, a+2d, a+3d)
    - all_3aps: list of tuples (a, a+d, a+2d)
    - ap3_membership: dict {x: set of indices of 3-APs containing x}
    - ap4_membership: dict {x: set of indices of 4-APs containing x}
    """
    all_4aps = []
    all_3aps = []
    
    # 4-APs
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                all_4aps.append((a, b, c, e))
                
    # 3-APs
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b, c))

    ap3_membership = {i: set() for i in range(1, N + 1)}
    ap4_membership = {i: set() for i in range(1, N + 1)}
    
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            ap3_membership[x].add(idx)
            
    for idx, ap in enumerate(all_4aps):
        for x in ap:
            ap4_membership[x].add(idx)
            
    return all_4aps, all_3aps, ap3_membership, ap4_membership

def _greedy_hitting_set():
    all_4aps, all_3aps, ap3_membership, ap4_membership = _precompute_aps()
    
    # active_3aps[i] = True if 3-AP i is active (all elements present)
    # active_4aps[i] = True if 4-AP i is active
    active_3aps = [True] * len(all_3aps)
    active_4aps = [True] * len(all_4aps)
    
    A = set(range(1, N + 1))
    
    # Count active 3-APs
    count_3 = len(all_3aps)
    count_4 = len(all_4aps)
    
    # We need to efficiently find the "best" element to remove.
    # An element x is "bad" if it is in any active 4-AP.
    # We want to remove x that minimizes the number of active 3-APs it is in.
    
    # To speed this up, we can maintain counts.
    # active_3_count[x] = number of active 3-APs containing x
    # active_4_count[x] = number of active 4-APs containing x
    
    active_3_count = [0] * (N + 1)
    active_4_count = [0] * (N + 1)
    
    for x in range(1, N + 1):
        active_3_count[x] = len(ap3_membership[x])
        active_4_count[x] = len(ap4_membership[x])
        
    while count_4 > 0:
        # Find bad elements
        bad_elements = [x for x in A if active_4_count[x] > 0]
        if not bad_elements:
            break
            
        # Pick x with min active_3_count[x], tie-break by max active_4_count[x]
        # Note: active_3_count[x] might be 0 if all 3-APs containing x are already broken.
        best_x = None
        best_score = None
        
        for x in bad_elements:
            # Score: lower is better.
            # Primary: active_3_count[x]
            # Secondary: -active_4_count[x] (higher is better, so negate)
            score = (active_3_count[x], -active_4_count[x])
            if best_score is None or score < best_score:
                best_score = score
                best_x = x
                
        if best_x is None:
            break
            
        # Remove best_x
        A.remove(best_x)
        
        # Update counts for all 3-APs and 4-APs containing best_x
        # For each 3-AP index in ap3_membership[best_x]:
        for idx in ap3_membership[best_x]:
            if active_3aps[idx]:
                # This 3-AP is broken
                active_3aps[idx] = False
                count_3 -= 1
                # Decrement counts for other elements in this 3-AP
                for x in all_3aps[idx]:
                    if x != best_x:
                        active_3_count[x] -= 1
                        
        # For each 4-AP index in ap4_membership[best_x]:
        for idx in ap4_membership[best_x]:
            if active_4aps[idx]:
                # This 4-AP is broken
                active_4aps[idx] = False
                count_4 -= 1
                # Decrement counts for other elements in this 4-AP
                for x in all_4aps[idx]:
                    if x != best_x:
                        active_4_count[x] -= 1
                        
    return sorted(A), count_3

def _count_3aps(A):
    A_set = set(A)
    count = 0
    for c in A:
        for d in range(1, N + 1):
            if (c - d) in A_set and (c + d) in A_set:
                count += 1
    return count

def _has_4ap(A):
    A_set = set(A)
    for a in A:
        for d in range(1, N + 1):
            if (a + d) in A_set and (a + 2*d) in A_set and (a + 3*d) in A_set:
                return True
    return False

def _local_search(A):
    """
    Try to add elements back if they don't create 4-APs.
    """
    A_set = set(A)
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if x in A_set:
                continue
            # Check if adding x creates a 4-AP
            creates_4 = False
            for d in range(1, N + 1):
                # x as 1st
                if (x+d) in A_set and (x+2*d) in A_set and (x+3*d) in A_set:
                    creates_4 = True
                    break
                # x as 2nd
                if (x-d) > 0 and (x-d) in A_set and (x+d) in A_set and (x+2*d) in A_set:
                    creates_4 = True
                    break
                # x as 3rd
                if (x-2*d) > 0 and (x-2*d) in A_set and (x-d) in A_set and (x+d) in A_set:
                    creates_4 = True
                    break
                # x as 4th
                if (x-3*d) > 0 and (x-3*d) in A_set and (x-2*d) in A_set and (x-d) in A_set:
                    creates_4 = True
                    break
            if not creates_4:
                A_set.add(x)
                improved = True
                
    return sorted(A_set)

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 - DEADLINE_MARGIN
    
    # 1. Greedy Hitting Set
    A_greedy, count_greedy = _greedy_hitting_set()
    
    # 2. Local Search (Add back safe elements)
    A_ls = _local_search(A_greedy)
    count_ls = _count_3aps(A_ls)
    
    best_A = A_ls
    best_count = count_ls
    
    # 3. Randomized Restart of Greedy Hitting Set
    # We can randomize the tie-breaking in the greedy step.
    # Or just run the greedy step with different random seeds for tie-breaking.
    
    # Given the time limit, we might not have time for many restarts if _greedy_hitting_set is slow.
    # _greedy_hitting_set is O(N * APs).
    # Let's check the time.
    
    if time.time() < deadline:
        # Try a few randomized variants
        # We can modify the greedy step to pick randomly among the top K best candidates.
        pass # For now, rely on the deterministic greedy + local search.
        
    # Final check
    if _has_4ap(best_A):
        # Fallback
        best_A = [1]
        
    return best_A
