import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin

    def time_up():
        return time.time() > DEADLINE

    # We want to find the maximum length n.
    # We use iterative deepening: try to find a sequence of length 1, then 2, etc.
    # But we can start from a reasonable lower bound to save time.
    
    # A known good lower bound for M=256 can be found by greedy.
    # Let's run a fast greedy first to get a baseline.
    
    def greedy_construct():
        prefix = [0]
        diffs = set()
        last_gap = 0
        while True:
            found = False
            # Try smallest valid gap first for greedy baseline?
            # Or largest? Smallest keeps P_n small, which is good for Sidon bound P_n >= n(n-1)/2.
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
                    found = True
                    break
            if not found:
                break
        return [prefix[i+1] - prefix[i] for i in range(len(prefix)-1)]

    baseline_seq = greedy_construct()
    best_seq = baseline_seq
    best_len = len(baseline_seq)
    
    # Now try to extend beyond baseline_len using DFS.
    # We will try to find a sequence of length L for L = best_len + 1, best_len + 2, ...
    
    # To speed up DFS, we can start from the greedy prefix if it's valid for the target length's prefix.
    # But the greedy prefix might not be extendable.
    # A better strategy: Run DFS from scratch for each target length, but with strong pruning.
    # Or, run DFS starting from the greedy prefix and try to replace/extend.
    
    # Let's use a DFS that tries to build the sequence from scratch, but prunes aggressively.
    # State: (prefix, diffs, last_gap, current_len)
    # We want to reach target_len.
    
    def dfs(prefix, diffs, last_gap, current_len, target_len):
        nonlocal best_seq, best_len, time_up
        if time_up():
            return False
        
        if current_len == target_len:
            # Found a valid sequence of target_len
            seq = [prefix[i+1] - prefix[i] for i in range(current_len)]
            if current_len > best_len:
                best_len = current_len
                best_seq = seq
            return True
        
        # Pruning: Check if it's possible to reach target_len
        # 1. Sidon bound: The number of differences so far is current_len*(current_len-1)/2.
        #    We need to add (target_len - current_len) more elements.
        #    The new prefix sum P_new must be at least P_current + (last_gap+1) + ...
        #    Also, the total number of differences will be target_len*(target_len-1)/2.
        #    The maximum difference is P_target.
        #    So we need P_target >= target_len*(target_len-1)/2.
        
        # Estimate minimal P_target:
        # We need to add k = target_len - current_len more gaps.
        # The next gap must be > last_gap. Minimal next gaps: last_gap+1, last_gap+2, ...
        min_extra_sum = 0
        g = last_gap
        for _ in range(target_len - current_len):
            g += 1
            min_extra_sum += g
        min_p_target = prefix[-1] + min_extra_sum
        
        if min_p_target < target_len * (target_len - 1) // 2:
            return False
            
        # 2. Gap bound: The gaps must be <= 256.
        # If last_gap is already close to 256, we might not be able to add enough gaps.
        # Max possible gaps we can add: 256 - last_gap.
        if 256 - last_gap < target_len - current_len:
            return False
            
        # Try next gaps
        # Heuristic: Try larger gaps first to spread out differences.
        # But we must ensure we don't exceed 256.
        # Also, to find *a* solution, larger gaps are often better for Sidon properties.
        
        # Optimization: Skip gaps that are "too small" if we are in a tight spot?
        # Let's just iterate from 256 down to last_gap + 1.
        
        for g in range(256, last_gap, -1):
            if time_up():
                return False
                
            pnew = prefix[-1] + g
            
            # Check Sidon condition for new point
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
                
            if dfs(prefix, diffs, g, current_len + 1, target_len):
                return True
                
            # Backtrack
            for d in new_diffs:
                diffs.discard(d)
            prefix.pop()
            
        return False

    # Try to extend length
    # Start from best_len + 1
    # We can also try to restart DFS from scratch for each target length to avoid getting stuck in a bad local area of the greedy prefix.
    # However, starting from scratch is expensive.
    # Let's try starting from the greedy prefix first. If it fails, try from scratch.
    
    # Actually, a pure DFS from scratch with iterative deepening is more robust.
    # Let's do that.
    
    # To make it faster, we can use a heuristic initial ordering.
    # We'll try target lengths from best_len + 1 upwards.
    
    for target_len in range(best_len + 1, 30):  # Upper bound 30 is safe
        if time_up():
            break
            
        # Check feasibility for target_len
        # Minimal P_n for target_len with strictly increasing gaps starting from 1:
        # Gaps: 1, 2, ..., target_len. Sum = target_len(target_len+1)/2.
        # Sidon bound: P_n >= target_len(target_len-1)/2.
        # So we need target_len(target_len+1)/2 >= target_len(target_len-1)/2, which is always true.
        # But we also have the gap limit 256.
        # Max length is limited by 256.
        
        # Run DFS from scratch
        prefix = [0]
        diffs = set()
        last_gap = 0
        
        if dfs(prefix, diffs, last_gap, 0, target_len):
            # Found a sequence of this length
            continue
        else:
            # If DFS fails, it might be because we hit the time limit or truly no solution.
            # If time limit, break.
            if time_up():
                break
            # If no solution found for this target_len, break?
            # Not necessarily, a longer sequence might exist? No, if length L is impossible, L+1 is impossible.
            # So we can break.
            break

    if not best_seq:
        best_seq = [1]
        
    return best_seq
