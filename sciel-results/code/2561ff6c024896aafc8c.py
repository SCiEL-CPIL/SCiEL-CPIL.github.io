"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(set_arr):
    """Check no element is the mean of any subset of >=2 OTHER elements.
    Uses DP: for each target a, check if any subset of others (size>=2) sums to len(a)*a.
    Returns True if valid.
    """
    n = len(set_arr)
    if n < 3:
        return True
    arr = np.sort(set_arr)
    for i in range(n):
        a = int(arr[i])
        others = np.concatenate([arr[:i], arr[i+1:]])
        m = n - 1
        if m < 2:
            continue
        # dp[k] = bitset (Python int) of sums achievable with exactly k elements
        dp = [0] * (m + 1)
        dp[0] = 1
        for e in others:
            e = int(e)
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << e)
        for k in range(2, m + 1):
            target = k * a
            if target > 512 * m:
                break
            if (dp[k] >> target) & 1:
                return False
    return True

def _is_valid_fast(set_arr):
    """Faster validity check using numpy bitsets with early termination."""
    n = len(set_arr)
    if n < 3:
        return True
    arr = np.sort(set_arr)
    for i in range(n):
        a = int(arr[i])
        others = np.concatenate([arr[:i], arr[i+1:]])
        m = n - 1
        if m < 2:
            continue
        dp = [0] * (m + 1)
        dp[0] = 1
        for e in others:
            e = int(e)
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << e)
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * m:
                    break
                if (dp[k] >> target) & 1:
                    return False
    return True

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s limit
    best = []
    best_size = 0
    
    N = 512
    MAX_SIZE = 48
    
    def check_validity_with_dp(S_list):
        """Full DP validity check. Returns True if valid."""
        m = len(S_list)
        if m < 3:
            return True
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            for k in range(2, m_others + 1):
                target = k * a
                if target > N * m_others:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True
    
    def incremental_greedy(order):
        """
        Greedily build a valid set by adding elements in 'order'.
        Uses incremental DP for speed.
        Returns the set built.
        """
        selected = []
        n = 0
        
        # dp_all[k]: bitset of sums achievable with k elements from selected
        dp_all = [1]  # dp_all[0] = 1 (sum 0)
        
        # For each element a in selected, dp_without[a][k]: bitset of sums with k elements from selected\{a}
        dp_without = {}
        
        for x in order:
            if time.time() > deadline:
                break
            if n >= MAX_SIZE:
                break
            
            # Check 1: Is x the mean of any subset of selected (size >= 2)?
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > N * n:
                    break
                if k < len(dp_all) and (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: For each a in selected, is a the mean of a subset involving x?
            for a in selected:
                if not valid:
                    break
                if a not in dp_without:
                    continue
                dp_w = dp_without[a]
                for k in range(1, n):  # k = |U|, U has k elements from selected\{a}
                    target = (k + 1) * a - x
                    if target < 0 or target > N * n:
                        continue
                    if k < len(dp_w) and (dp_w[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add x to selected
            selected.append(x)
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = dp_all[k] if k < len(dp_all) else 0
                if k - 1 >= 0 and k - 1 < len(dp_all):
                    val |= (dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                if a not in dp_without:
                    continue
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k < len(old_dp_w) else 0
                    if k - 1 >= 0 and k - 1 < len(old_dp_w):
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x, dp_without[x] is the old dp_all (selected before adding x)
            new_dp_without[x] = dp_all[:]
            
            dp_without = new_dp_without
            n += 1
        
        return selected
    
    def local_search_improve(S, max_iters=100):
        """
        Try to improve set S by adding elements or swapping.
        Uses full DP check for simplicity.
        """
        S = set(S)
        all_vals = set(range(1, N + 1))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add an element
            added = False
            candidates_to_add = [x for x in all_vals if x not in S]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add[:50]:
                new_S = list(S | {x})
                if check_validity_with_dp(new_S):
                    S = S | {x}
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap
            if S:
                a = random.choice(list(S))
                candidates_to_swap = [x for x in all_vals if x not in S]
                random.shuffle(candidates_to_swap)
                for x in candidates_to_swap[:50]:
                    new_S = list((S - {a}) | {x})
                    if check_validity_with_dp(new_S):
                        S = (S - {a}) | {x}
                        break
        
        return list(S)
    
    # Generate multiple candidate orders and try each
    random.seed(42)
    
    # Order 1: natural ascending
    order1 = list(range(1, N + 1))
    
    # Order 2: descending
    order2 = list(range(N, 0, -1))
    
    # Order 3: random shuffle
    order3 = list(range(1, N + 1))
    random.shuffle(order3)
    
    # Order 4: spread out (interleaved)
    order4 = []
    for i in range(1, N + 1, 4):
        for j in [i, i+1, i+2, i+3]:
            if j <= N:
                order4.append(j)
    
    # Order 5: powers of 2 first, then fill
    powers = [2**i for i in range(10) if 2**i <= N]
    rest = [x for x in range(1, N + 1) if x not in set(powers)]
    random.shuffle(rest)
    order5 = powers + rest
    
    # Order 6: another random
    order6 = list(range(1, N + 1))
    random.shuffle(order6)
    
    orders = [order1, order2, order3, order4, order5, order6]
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        
        result = incremental_greedy(order)
        
        if len(result) > best_size:
            best_size = len(result)
            best = result[:]
        
        # Try to improve with local search if time permits
        if time.time() < deadline - 15 and len(result) > 0:
            improved = local_search_improve(result, max_iters=50)
            if len(improved) > best_size:
                best_size = len(improved)
                best = improved[:]
    
    # Final verification and return
    if best and not _is_valid(np.array(best)):
        # Fallback: use a known small valid set
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best
# EVOLVE-BLOCK-END
