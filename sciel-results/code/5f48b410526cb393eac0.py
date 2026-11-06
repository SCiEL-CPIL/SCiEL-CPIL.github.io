"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    """
    Find the minimum number of distinct divisors of 5040 that sum to 4979.
    Returns the list of such divisors.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # Sort descending for potential greedy fallback
    
    # DP: dp[s] = (min_count, last_divisor_index, prev_sum)
    # We want to minimize count.
    # Since target is 4979 and max divisor is 5040, the sum is bounded.
    # However, we only care about sums up to target.
    
    # Initialize DP table
    # dp[s] stores the tuple (count, last_idx, prev_sum) for sum s
    # last_idx is the index in the divisors list used to reach this state
    # We iterate through divisors one by one.
    # To reconstruct, we need to store the choice at each step.
    # Standard 0/1 knapsack for min count.
    
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_last = [-1] * (target + 1)  # Index of the last divisor added
    dp_prev = [-1] * (target + 1)  # Previous sum before adding the last divisor
    
    dp_count[0] = 0
    
    # We need to process divisors in a specific order to ensure distinctness.
    # Standard 0/1 knapsack: iterate items (divisors), update dp from high sum to low sum.
    # But we also need to track which item was used.
    # Let's use a 2D DP or store backpointers carefully.
    # Since target is ~5000 and num divisors is ~60, a 2D DP [num_divs][target] is small.
    # num_divs = len(divisors)
    
    m = len(divisors)
    # dp[i][s] = min count using first i divisors to sum to s
    # To save memory and time, we can use 1D if we are careful, but backtracking is easier with 2D or explicit storage.
    # Let's use 2D for clarity and speed (60 * 5000 = 300k states, very fast).
    
    # dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    # prev[i][s] = index of the last divisor used (in the original list) or -1 if not reached
    # Actually, for 0/1 knapsack, if we use divisor i-1 to reach sum s, then previous sum was s - divisors[i-1].
    
    # Let's optimize:
    # dp[i][s] depends on dp[i-1][s] and dp[i-1][s - divisors[i-1]] + 1
    
    # We can store just the count and a boolean 'used' or the previous sum.
    # Let's store:
    # min_count[i][s]
    # used[i][s] -> bool, whether divisors[i-1] was used to achieve min_count[i][s]
    
    # Memory: 60 * 5000 * 2 bytes/bool? Python lists of ints/bools are heavy.
    # Let's use arrays or lists of tuples?
    # Better: just store min_count[i][s]. To backtrack, we check if min_count[i][s] == min_count[i-1][s - divisors[i-1]] + 1.
    # If yes, divisors[i-1] was used.
    # This works if we store the full 2D table of counts.
    
    # 60 * 5000 = 300,000 integers. This is fine.
    
    counts = [[INF] * (target + 1) for _ in range(m + 1)]
    counts[0][0] = 0
    
    for i in range(1, m + 1):
        d = divisors[i - 1]
        # Copy previous row
        # counts[i][s] = counts[i-1][s]
        # Then try adding d
        for s in range(target + 1):
            # Option 1: Don't use d
            best = counts[i - 1][s]
            # Option 2: Use d
            if s >= d:
                prev_val = counts[i - 1][s - d]
                if prev_val + 1 < best:
                    best = prev_val + 1
            counts[i][s] = best
            
            # Deadline check (optional, but good practice)
            # if time.time() > deadline:
            #     break
        
        # if time.time() > deadline:
        #     break
    
    # Check if solution exists
    if counts[m][target] == INF:
        # Fallback: Greedy (should not happen for this specific instance)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        # If greedy fails, raise or return empty? Task implies solution exists.
        # The seed program asserts remainder==0, so it assumes greedy works.
        # But we are doing exact DP, so it should find a solution if one exists.
        # If DP says INF, no solution exists with distinct divisors.
        # However, the problem states "Represent 4979 as a sum...", implying existence.
        # Let's try to return the greedy result if DP fails, or raise.
        # Given the nature of 5040 (practical number), a solution likely exists.
        # Let's assume counts[m][target] < INF.
        pass
    
    # Backtrack to find the divisors
    chosen = []
    s = target
    for i in range(m, 0, -1):
        d = divisors[i - 1]
        # Check if d was used
        # If counts[i][s] == counts[i-1][s - d] + 1, then d was used.
        # Note: It's possible that counts[i][s] == counts[i-1][s] (d not used)
        # And also counts[i][s] == counts[i-1][s-d] + 1 (d used).
        # We need to pick the one that led to the minimum count.
        # If both are equal, we can pick either. Let's pick 'used' if it doesn't increase count?
        # No, we want min count.
        # If counts[i-1][s] < counts[i-1][s-d] + 1, then d was NOT used.
        # If counts[i-1][s-d] + 1 < counts[i-1][s], then d WAS used.
        # If equal, we can choose either. Let's choose 'not used' to keep indices smaller? Or 'used'?
        # It doesn't matter for the count.
        
        if s >= d and counts[i - 1][s - d] + 1 == counts[i][s]:
            # d was used (or could have been)
            # We must ensure this choice is consistent with the minimum.
            # If counts[i-1][s] is also equal, we have a tie.
            # Let's check if not using d is better or equal.
            if counts[i - 1][s] < counts[i - 1][s - d] + 1:
                # Definitely not used
                pass
            else:
                # Used is optimal (or tied)
                chosen.append(d)
                s -= d
        # Else, d was not used
    
    # The chosen list is in reverse order of the divisors list (which was sorted descending).
    # So chosen contains divisors in descending order of their index in the sorted list.
    # The order in the return list doesn't matter for the sum, but let's return it.
    
    return chosen

# EVOLVE-BLOCK-END
