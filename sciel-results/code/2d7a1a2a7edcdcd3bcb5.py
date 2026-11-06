import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # Safety margin before 90s kill

    def time_up():
        return time.time() > DEADLINE

    # The problem: find a strictly increasing sequence a_1 < a_2 < ... < a_n in [1, 256]
    # such that all sums of non-empty consecutive blocks are distinct.
    # Let P_i = a_1 + ... + a_i (prefix sums), with P_0 = 0.
    # The sum of consecutive block a_i+1..a_j is P_j - P_{i-1}.
    # So we need all differences P_j - P_i for 0 <= i < j <= n to be distinct.
    # This is exactly a Sidon sequence (or B_2 sequence) on the prefix sums,
    # where the prefix sums are 0 = P_0 < P_1 < ... < P_n, and P_i - P_{i-1} = a_i >= 1,
    # and a_i < a_{i+1} (strictly increasing gaps).
    # Also a_n <= 256, so P_n - P_{n-1} <= 256, and all a_i <= 256.
    # Wait, the constraint is that the sequence elements are in [1, 256], i.e., 1 <= a_i <= 256.
    # And the sequence is strictly increasing: a_1 < a_2 < ... < a_n.
    # So the gaps g_i = a_i are strictly increasing positive integers, each <= 256.
    # The prefix sums P_i must form a Sidon sequence: all P_j - P_i (0 <= i < j <= n) are distinct.

    # Strategy: Use BFS/DFS with strong pruning.
    # State: (current_prefix_list, set_of_differences, last_gap)
    # We want to maximize the length.

    # First, get a good baseline with a greedy approach.
    # Greedy: try to add the smallest possible next gap that doesn't create a duplicate difference.
    
    def greedy_construct():
        prefix = [0]
        diffs = set()
        last_gap = 0
        gaps = []
        while True:
            found = False
            for g in range(last_gap + 1, 257):
                pnew = prefix[-1] + g
                ok = True
                for p in prefix:
                    d = pnew - p
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    prefix.append(pnew)
                    for p in prefix[:-1]:
                        diffs.add(pnew - p)
                    last_gap = g
                    gaps.append(g)
                    found = True
                    break
            if not found:
                break
        return gaps

    baseline_seq = greedy_construct()
    best_seq = baseline_seq
    best_len = len(baseline_seq)
    
    # Now try to find a longer sequence using DFS with pruning.
    # We'll use iterative deepening: try to find a sequence of length best_len + 1, then + 2, etc.
    
    # To make DFS faster, we use forward checking: when we add a new element, we check if
    # the remaining search space is still feasible.
    
    def sidon_feasible(prefix, diffs, last_gap, current_len, target_len):
        # Check if it's possible to reach target_len from the current state.
        # 1. We need to add k = target_len - current_len more gaps.
        # 2. The next gaps must be > last_gap, and each <= 256.
        #    So the minimum next gaps are last_gap+1, last_gap+2, ..., last_gap+k.
        #    The maximum value of the last gap is last_gap + k, which must be <= 256.
        if last_gap + (target_len - current_len) > 256:
            return False
        
        # 3. Sidon bound: The total number of differences in a complete sequence of length target_len
        #    is target_len * (target_len - 1) / 2.
        #    The maximum difference is P_target = prefix[-1] + sum of remaining gaps.
        #    At minimum, the remaining gaps are last_gap+1, ..., last_gap+k.
        #    So P_target >= prefix[-1] + sum_{i=1}^{k} (last_gap + i)
        #    We need P_target >= target_len * (target_len - 1) / 2 (since all differences are distinct positive integers,
        #    the largest difference is at least the count of differences).
        k = target_len - current_len
        min_extra = k * last_gap + k * (k + 1) // 2
        min_p_target = prefix[-1] + min_extra
        if min_p_target < target_len * (target_len - 1) // 2:
            return False
        
        # 4. Forward checking: For each potential next gap g, check if adding it would make
        #    the remaining problem infeasible. We don't do full forward checking here for speed,
        #    but we can do a quick check: the new last_gap would be g, and we need to add k-1 more gaps.
        #    This is expensive, so we skip it and rely on the above bounds.
        
        return True

    def dfs(prefix, diffs, last_gap, current_len, target_len, gaps_so_far):
        nonlocal best_seq, best_len
        
        if time_up():
            return False
        
        if current_len == target_len:
            if current_len > best_len:
                best_len = current_len
                best_seq = list(gaps_so_far)
            return True
        
        if not sidon_feasible(prefix, diffs, last_gap, current_len, target_len):
            return False
        
        # Try next gaps. Heuristic: try smaller gaps first to keep P_n small,
        # which makes it easier to satisfy the Sidon condition for future elements.
        # But we also need to ensure we can still reach target_len.
        
        k = target_len - current_len
        # The smallest possible next gap is last_gap + 1.
        # The largest possible next gap is 256 - (k - 1) (to leave room for k-1 more strictly larger gaps).
        max_g = min(256, 256 - (k - 1))
        
        for g in range(last_gap + 1, max_g + 1):
            if time_up():
                return False
            
            pnew = prefix[-1] + g
            ok = True
            for p in prefix:
                d = pnew - p
                if d in diffs:
                    ok = False
                    break
            
            if not ok:
                continue
            
            # Add new point
            prefix.append(pnew)
            new_diffs = []
            for p in prefix[:-1]:
                d = pnew - p
                new_diffs.append(d)
                diffs.add(d)
            gaps_so_far.append(g)
            
            if dfs(prefix, diffs, g, current_len + 1, target_len, gaps_so_far):
                return True
            
            # Backtrack
            for d in new_diffs:
                diffs.discard(d)
            prefix.pop()
            gaps_so_far.pop()
        
        return False

    # Try to extend the length
    for target_len in range(best_len + 1, 30):
        if time_up():
            break
        
        # Quick feasibility check
        k = target_len
        min_p = k * (k + 1) // 2  # minimal P_n with gaps 1, 2, ..., k
        if min_p < k * (k - 1) // 2:
            continue
        if k > 256:  # can't have more than 256 gaps each >= 1 and strictly increasing
            break
        
        prefix = [0]
        diffs = set()
        gaps_so_far = []
        
        if dfs(prefix, diffs, 0, 0, target_len, gaps_so_far):
            continue
        else:
            if time_up():
                break
            # If no solution found for this target_len, a longer one is also impossible
            break

    if not best_seq:
        best_seq = [1]
        
    return best_seq
