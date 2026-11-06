import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality."""
    n = len(elements)
    if n == 0:
        return [0] * 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << e)
    return dp

def _is_valid_mean_free_descending(S, x):
    """
    S: current set (all elements > x)
    x: new candidate (smallest)
    Returns True if S U {x} is valid.
    """
    new_set = S + [x]
    n = len(new_set)
    
    # Build DP for the entire new set
    dp_total = _build_dp(new_set)
    
    # Check 1: Is x the mean of any subset of S?
    # Since all s in S are > x, sum of k elements > k*x.
    # So x cannot be the mean. Skip.
    
    # Check 2: Is any s in S the mean of any subset of new_set \ {s}?
    for s in S:
        # Check if s is the mean of any subset of new_set \ {s}
        # Target sum for k elements is k * s
        # We need to check if k * s is achievable by k elements from new_set \ {s}
        
        # Optimization: Check if k * s is in dp_total[k]
        # If not, s is safe for this k.
        # If yes, we must verify if s is used in that subset.
        # If s is NOT used, then s is the mean -> Invalid.
        # If s IS used, we need to check if there's another subset not using s.
        
        # Simple approach: Build DP for new_set \ {s} and check.
        # Only do this if dp_total[k] contains k*s.
        
        potential_violation = False
        for k in range(2, n):
            target = k * s
            if target > 512 * n: # Max possible sum
                break
            if (dp_total[k] >> target) & 1:
                # Potential match in total set.
                # Check if it can be formed without s.
                # Build DP for new_set without s
                others = [e for e in new_set if e != s]
                dp_others = _build_dp(others)
                if (dp_others[k] >> target) & 1:
                    return False
                # If not in dp_others, the match in dp_total MUST use s.
                # So s is not the mean of OTHERS.
                # Continue to next k.
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = []
    best_len = 0
    
    # Strategy: Descending Greedy
    current_set = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if _is_valid_mean_free_descending(current_set, x):
            current_set.append(x)
            
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
        
    if best_len >= 48:
        return best_set

    # If greedy didn't reach 48, try a few random restarts or local search?
    # Given the time, maybe just return the greedy result.
    # The descending greedy is very strong for this problem.
    
    return best_set
