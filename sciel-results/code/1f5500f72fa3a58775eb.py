import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses incremental DP for fast validity checks and local search.
    """
    N = 512
    MAX_SIZE = 48
    MAX_SUM = N * MAX_SIZE  # 24576
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Helper to rebuild DP structures for a given set
    def build_dps(S_list):
        """
        Builds dp_all and dp_without for a set S_list.
        dp_all[k]: bitset of sums of k elements from S_list.
        dp_without[a][k]: bitset of sums of k elements from S_list \ {a}.
        """
        m = len(S_list)
        if m == 0:
            return [1], {}
        
        # dp_all[k] for k=0..m
        dp_all = [0] * (m + 1)
        dp_all[0] = 1
        for x in S_list:
            # Update in reverse to avoid using x multiple times
            for k in range(m, 0, -1):
                dp_all[k] |= (dp_all[k - 1] << x)
        
        # dp_without[a] for each a in S_list
        dp_without = {}
        for idx, a in enumerate(S_list):
            others = [x for i, x in enumerate(S_list) if i != idx]
            m_o = len(others)
            dp_w = [0] * (m_o + 1)
            dp_w[0] = 1
            for x in others:
                for k in range(m_o, 0, -1):
                    dp_w[k] |= (dp_w[k - 1] << x)
            dp_without[a] = dp_w
            
        return dp_all, dp_without

    def is_valid_with_dps(a, dp_all, dp_without, S_list):
        """
        Checks if adding 'a' to the current set (represented by dp_all, dp_without, S_list)
        would maintain validity.
        Actually, this function is better used to check if the CURRENT set is valid,
        or to check a candidate for addition.
        
        Let's define:
        check_candidate(x, S_list, dp_all):
           1. Is x the mean of any subset of S_list (size >= 2)?
              Check if k*x is in dp_all[k] for k>=2.
           2. For each a in S_list, is a the mean of any subset of S_list U {x} 
              that includes x?
              Subset T includes x. T \ {x} is a subset U of S_list \ {a}.
              sum(T) = |T| * a.
              sum(U) + x = (|U| + 1) * a.
              sum(U) = (|U| + 1) * a - x.
              U has size k. So we check if ((k+1)*a - x) is in dp_without[a][k].
              k ranges from 1 to len(S_list)-1 (since U subset of S_list\{a}).
              Note: |T| = k+1. We require |T| >= 2 => k+1 >= 2 => k >= 1.
        """
        m = len(S_list)
        
        # Check 1: x is mean of subset of S_list
        for k in range(2, m + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if (dp_all[k] >> target) & 1:
                return False

        # Check 2: a in S_list is mean of subset involving x
        for a in S_list:
            dp_w = dp_without[a]
            # U is subset of S_list \ {a}, size k.
            # k can be 1 to m-1.
            for k in range(1, m):
                target = (k + 1) * a - x
                if target < 0 or target > MAX_SUM:
                    continue
                if (dp_w[k] >> target) & 1:
                    return False
        return True

    def check_full_validity(S_list):
        """Full check for a set, used for final verification."""
        m = len(S_list)
        if m < 3:
            return True
        for idx, a in enumerate(S_list):
            others = [x for i, x in enumerate(S_list) if i != idx]
            m_o = len(others)
            dp = [0] * (m_o + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_o, 0, -1):
                    dp[k] |= (dp[k - 1] << x)
            for k in range(2, m_o + 1):
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def incremental_greedy(order):
        """
        Greedily build a valid set by adding elements in 'order'.
        Uses incremental DP for speed.
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
            
            # Check if x can be added
            valid = is_valid_with_dps(x, selected, dp_all, dp_without)
            
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
                    val |= (dp_all[k - 1] << x)
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
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x, dp_without[x] is the old dp_all (selected before adding x)
            new_dp_without[x] = dp_all[:]
            
            dp_without = new_dp_without
            n += 1
        
        return selected

    def local_search_improve(S_initial, max_iters=300):
        """
        Improve set S by trying to add elements or swap.
        Rebuilds DPs when necessary (removal/swap).
        """
        S = set(S_initial)
        S_list = list(S)
        
        # Initial build of DPs
        dp_all, dp_without = build_dps(S_list)
        n = len(S_list)
        
        all_vals = set(range(1, N + 1))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add an element
            added = False
            candidates_to_add = [x for x in all_vals if x not in S]
            random.shuffle(candidates_to_add)
            
            # Limit candidates to check to keep time bounded
            for x in candidates_to_add[:50]:
                if is_valid_with_dps(x, S_list, dp_all, dp_without):
                    # Add x
                    S.add(x)
                    S_list.append(x)
                    n += 1
                    
                    # Update DPs incrementally
                    new_dp_all = [0] * (n + 1)
                    new_dp_all[0] = 1
                    for k in range(1, n + 1):
                        val = dp_all[k] if k < len(dp_all) else 0
                        if k - 1 >= 0 and k - 1 < len(dp_all):
                            val |= (dp_all[k - 1] << x)
                        new_dp_all[k] = val
                    dp_all = new_dp_all
                    
                    new_dp_without = {}
                    for a in S_list[:-1]:
                        old_dp_w = dp_without[a]
                        new_list = [0] * (n + 1)
                        new_list[0] = 1
                        for k in range(1, n + 1):
                            val = old_dp_w[k] if k < len(old_dp_w) else 0
                            if k - 1 >= 0 and k - 1 < len(old_dp_w):
                                val |= (old_dp_w[k - 1] << x)
                            new_list[k] = val
                        new_dp_without[a] = new_list
                    
                    new_dp_without[x] = dp_all[:]
                    dp_without = new_dp_without
                    
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap if we haven't reached max size or want to try different elements
            if S:
                a = random.choice(list(S))
                candidates_to_swap = [x for x in all_vals if x not in S]
                random.shuffle(candidates_to_swap)
                
                swapped = False
                for x in candidates_to_swap[:50]:
                    # Temporarily remove a
                    S_temp = S - {a}
                    S_list_temp = [y for y in S_list if y != a]
                    dp_all_temp, dp_without_temp = build_dps(S_list_temp)
                    
                    if is_valid_with_dps(x, S_list_temp, dp_all_temp, dp_without_temp):
                        # Perform swap
                        S = S_temp | {x}
                        S_list = S_list_temp + [x]
                        n = len(S_list)
                        dp_all, dp_without = build_dps(S_list) # Rebuild to be safe/simple
                        swapped = True
                        break
                
                if not swapped:
                    # If no swap found, we might be stuck. 
                    # To escape local minima, we can try removing a random element 
                    # and then trying to add something else, but that reduces size.
                    # Since we want to MAXIMIZE size, we only swap if it keeps size same.
                    # If we can't add and can't swap, we are at a local maximum for this seed.
                    pass

        return S

    best_set = set()
    best_size = 0
    candidates = list(range(1, N + 1))

    # Strategy 1: Greedy independent set on conflict graph with various orderings
    # We use incremental greedy with various orderings
    
    orderings = []
    # Large-first
    orderings.append(list(range(N, 0, -1)))
    # Small-first
    orderings.append(list(range(1, N + 1)))
    # Random
    for _ in range(10):
        o = candidates[:]
        random.shuffle(o)
        orderings.append(o)
    # Mixed
    mixed = []
    for i in range(N // 2):
        mixed.append(N - i)
        mixed.append(i + 1)
    orderings.append(mixed)
    # Powers of 2 first
    powers = [2 ** i for i in range(1, 10) if 2 ** i <= N]
    rest = [x for x in candidates if x not in set(powers)]
    random.shuffle(rest)
    orderings.append(powers + rest)
    
    # Run greedy for each ordering
    for order in orderings:
        if time.time() > deadline:
            break

        S_list = incremental_greedy(order)
        S = set(S_list)

        # Verify
        if S:
            if check_full_validity(list(S)):
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()

        # Local search to improve
        if len(S) < MAX_SIZE:
            S_improved = local_search_improve(S, max_iters=100)
            S_improved = set(S_improved)
            if S_improved:
                if check_full_validity(list(S_improved)):
                    if len(S_improved) > best_size:
                        best_size = len(S_improved)
                        best_set = S_improved.copy()

        if best_size == MAX_SIZE:
            break

    # If we have a good set, try more local search
    if best_set and len(best_set) < MAX_SIZE:
        best_set_improved = local_search_improve(best_set, max_iters=200)
        best_set_improved = set(best_set_improved)
        if best_set_improved:
            if check_full_validity(list(best_set_improved)):
                best_size = len(best_set_improved)
                best_set = best_set_improved

    # If still not at MAX_SIZE, try repair-based approach or more restarts
    if best_size < MAX_SIZE and time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline:
                break

            # Random start
            S = set(random.sample(candidates, min(30, MAX_SIZE)))
            S_list = list(S)
            
            # Try to expand this random set
            S_expanded = local_search_improve(S, max_iters=100)
            S_expanded = set(S_expanded)
            
            if S_expanded:
                if check_full_validity(list(S_expanded)):
                    if len(S_expanded) > best_size:
                        best_size = len(S_expanded)
                        best_set = S_expanded.copy()

            if best_size == MAX_SIZE:
                break

    # Final verification
    if best_set:
        if check_full_validity(list(best_set)):
            return list(best_set)

    # Fallback
    return [1, 3, 7, 15, 31, 63, 127, 255, 511]

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
