import time
import random

def _can_add_with_dp(dp, current_set, cand, n):
    """
    Check if cand can be added to current_set given the DP table for current_set.
    Returns True if safe. n = len(current_set)
    """
    if n == 0:
        return True
    
    # 1. cand is not mean of any subset of current_set (size >= 2)
    for k in range(2, n + 1):
        target = k * cand
        if (dp[k] >> target) & 1:
            return False
            
    # 2. No a in current_set is mean of subset of (current_set U {cand}) including cand.
    for a in current_set:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            if 0 <= target and (dp[k] >> target) & 1:
                return False
    return True

def _update_dp(dp, cand, k_prev):
    """
    Update DP table in-place to include cand.
    k_prev = size of set before adding cand.
    """
    for k in range(k_prev + 1, 0, -1):
        dp[k] |= (dp[k - 1] << cand)

def _build_dp(elements):
    """Build DP table from scratch for a given list of elements."""
    max_dp_size = 50
    dp = [0] * (max_dp_size + 1)
    dp[0] = 1
    cnt = 0
    for elem in elements:
        _update_dp(dp, elem, cnt)
        cnt += 1
    return dp

def _count_violations_for_element(dp, current_set, elem, n):
    """
    Count how many subset-mean violations involve elem.
    A violation exists if elem is the mean of some subset of other elements,
    OR if some other element is the mean of a subset that includes elem.
    """
    count = 0
    others = [x for x in current_set if x != elem]
    
    # Check if elem is mean of any subset of others (size >= 2)
    # We need DP for 'others', but we have DP for full set.
    # This is tricky - let's just do a simpler heuristic:
    # Check if elem is mean of any pair, triple, etc. from others.
    # For speed, only check small subsets.
    
    # Actually, let's use a different approach:
    # Try removing elem and see if the remaining set is valid.
    # If removing elem fixes violations, then elem was causing them.
    # But that's expensive.
    
    # Simpler: just return 0 and rely on the swap test.
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_len = 0
    
    nums = list(range(1, 513))
    
    # Strategy 1: Greedy with multiple orderings
    orders = []
    
    # Edge first
    edge_order = []
    l, r = 1, 512
    while l <= r:
        edge_order.append(l)
        if l != r:
            edge_order.append(r)
        l += 1
        r -= 1
    orders.append(edge_order)
    
    # Ascending, descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    # Random orders with different seeds
    for seed in range(50):
        rnd = random.Random(seed)
        rnd_order = list(range(1, 513))
        rnd.shuffle(rnd_order)
        orders.append(rnd_order)
    
    for order in orders:
        if time.time() > deadline - 15.0:
            break
            
        current_set_list = []
        max_dp_size = 50
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        
        for cand in order:
            if time.time() > deadline - 10.0:
                break
            
            n = len(current_set_list)
            if _can_add_with_dp(dp, current_set_list, cand, n):
                current_set_list.append(cand)
                _update_dp(dp, cand, n)
                
        if len(current_set_list) > best_len:
            best_set = current_set_list[:]
            best_len = len(best_set)
            
            if best_len >= 48:
                return best_set
    
    # Strategy 2: Local search - conflict-driven swap heuristic
    if best_len > 0 and time.time() < deadline - 10.0:
        current_set = best_set[:]
        current_set_set = set(current_set)
        
        # Build DP for current_set
        dp = _build_dp(current_set)
        
        improved = True
        max_iterations = 1000
        iteration = 0
        
        while improved and time.time() < deadline - 5.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(current_set)
            
            # Try each element as a potential removal
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                
                elem_to_remove = current_set[i]
                remaining = [x for x in current_set if x != elem_to_remove]
                remaining_set = set(remaining)
                
                # Rebuild DP for remaining
                dp_rem = _build_dp(remaining)
                
                # Try adding each number not in remaining
                best_cand = None
                best_cand_idx = -1
                
                # Prioritize candidates near edges (they tend to have fewer conflicts)
                candidates = [x for x in nums if x not in remaining_set]
                # Sort by distance from center (edges first)
                candidates.sort(key=lambda x: min(abs(x - 1), abs(x - 512)))
                
                for cand in candidates:
                    if time.time() > deadline - 5.0:
                        break
                    
                    if _can_add_with_dp(dp_rem, remaining, cand, len(remaining)):
                        # This swap is valid
                        new_set = remaining + [cand]
                        
                        # Check if this improves anything (e.g., allows future additions)
                        # For now, just accept the first valid swap
                        best_cand = cand
                        break
                
                if best_cand is not None:
                    # Perform the swap
                    current_set = [x for x in current_set if x != elem_to_remove]
                    current_set.append(best_cand)
                    current_set_set = set(current_set)
                    dp = _build_dp(current_set)
                    improved = True
                    break  # Restart the outer loop after a swap
    
    if len(current_set) > best_len:
        best_set = current_set
        best_len = len(best_set)
    
    return best_set
