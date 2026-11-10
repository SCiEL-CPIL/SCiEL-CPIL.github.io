import time
import random

def _can_add(S_bool, x, n):
    """
    Check if x can be added to S (represented by S_bool) without forming a 4-AP.
    S_bool is a list where S_bool[i] is True if i is in S.
    """
    # x is 1st term: x, x+d, x+2d, x+3d
    # Need x+d, x+2d, x+3d in S
    # d >= 1
    # x+3d <= n => d <= (n-x)//3
    for d in range(1, (n - x) // 3 + 1):
        if S_bool[x + d] and S_bool[x + 2 * d] and S_bool[x + 3 * d]:
            return False

    # x is 2nd term: x-d, x, x+d, x+2d
    # Need x-d, x+d, x+2d in S
    # x-d >= 1 => d <= x-1
    # x+2d <= n => d <= (n-x)//2
    max_d = min(x - 1, (n - x) // 2)
    for d in range(1, max_d + 1):
        if S_bool[x - d] and S_bool[x + d] and S_bool[x + 2 * d]:
            return False

    # x is 3rd term: x-2d, x-d, x, x+d
    # Need x-2d, x-d, x+d in S
    # x-2d >= 1 => d <= (x-1)//2
    # x+d <= n => d <= n-x
    max_d = min((x - 1) // 2, n - x)
    for d in range(1, max_d + 1):
        if S_bool[x - 2 * d] and S_bool[x - d] and S_bool[x + d]:
            return False

    # x is 4th term: x-3d, x-2d, x-d, x
    # Need x-3d, x-2d, x-d in S
    # x-3d >= 1 => d <= (x-1)//3
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if S_bool[x - 3 * d] and S_bool[x - 2 * d] and S_bool[x - d]:
            return False

    return True

def _greedy_fill(S_set, S_bool, n, deadline):
    """
    Greedily add elements from 1 to n to S.
    """
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if not S_bool[x]:
            if _can_add(S_bool, x, n):
                S_set.add(x)
                S_bool[x] = True

def _local_search(S_set, S_bool, n, deadline):
    """
    Local search: try removing one element and greedily refilling.
    Returns updated S_set and S_bool.
    """
    improved = True
    # Limit passes to prevent excessive time if stuck in a loop of non-improvements
    # Though 'improved' flag handles this, random shuffling might cause oscillation.
    # We use a max_passes limit.
    max_passes = 100 
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline - 5:
        improved = False
        passes += 1
        # Shuffle order of removals to explore different neighborhoods
        elements = list(S_set)
        random.shuffle(elements)
        
        for u in elements:
            if time.time() > deadline - 2:
                break
            
            # Copy state
            S_copy = set(S_set)
            S_bool_copy = S_bool[:]  # Shallow copy of list
            
            # Remove u
            S_copy.remove(u)
            S_bool_copy[u] = False
            
            # Greedy refill
            for x in range(1, n + 1):
                if time.time() > deadline - 1:
                    break
                if not S_bool_copy[x]:
                    if _can_add(S_bool_copy, x, n):
                        S_copy.add(x)
                        S_bool_copy[x] = True
            
            if len(S_copy) > len(S_set):
                S_set = S_copy
                S_bool = S_bool_copy
                improved = True
                # Break to restart the loop with the new set and new shuffle
                break
                
    return S_set, S_bool

def construct():
    n = 512
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We stop at 85s.
    deadline = start_time + 85
    
    # 1. Base-3 Construction
    S_set = set()
    S_bool = [False] * (n + 1)
    
    for x in range(1, n + 1):
        t = x
        ok = True
        while t > 0:
            if t % 3 == 2:
                ok = False
                break
            t //= 3
        if ok:
            S_set.add(x)
            S_bool[x] = True
            
    # 2. Greedy Fill
    _greedy_fill(S_set, S_bool, n, deadline)
    
    # 3. Local Search
    S_set, S_bool = _local_search(S_set, S_bool, n, deadline)
    
    # 4. Randomized Perturbation and Refinement
    best_size = len(S_set)
    best_set = set(S_set)
    best_bool = S_bool[:]
    
    # If time remains, try perturbing the best set found so far
    # We do this in a loop, attempting to improve further.
    
    # To avoid getting stuck, we can try different perturbation strengths.
    
    # We reserve 10 seconds for the final phase
    final_deadline = deadline - 5
    
    while time.time() < final_deadline:
        # Create a perturbed version
        S_pert = set(best_set)
        S_bool_pert = best_bool[:]
        
        # Remove 2-5 random elements
        k = random.randint(2, 5)
        if len(S_pert) >= k:
            to_remove = random.sample(list(S_pert), k)
            for u in to_remove:
                S_pert.remove(u)
                S_bool_pert[u] = False
        
        # Greedy fill
        _greedy_fill(S_pert, S_bool_pert, n, final_deadline)
        
        # Short local search
        # We pass a tighter deadline to ensure we don't overshoot
        sub_deadline = min(time.time() + 5, final_deadline)
        S_pert, S_bool_pert = _local_search(S_pert, S_bool_pert, n, sub_deadline)
        
        if len(S_pert) > best_size:
            best_size = len(S_pert)
            best_set = S_pert
            best_bool = S_bool_pert
            # If we found a better set, we can continue.
            # The loop condition will check time again.
        else:
            # If not improved, we can just continue the loop to try another perturbation.
            pass

    return sorted(best_set)
