# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0

    # Precompute APs
    # To save memory and time, we only store APs of length >= 2? 
    # Singletons always have sum +/-1. If B < 1, impossible. So B >= 1.
    # We can ignore singletons in the constraint check if B >= 1, as they are always satisfied.
    # However, they affect the "remaining" count? No, they are independent.
    # So we can just ensure B >= 1.
    
    ap_list = []
    index_aps = [[] for _ in range(n)]
    
    # Generate APs with length >= 2
    # d is step, L is length
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        if max_len < 2:
            continue
        for L in range(2, max_len + 1):
            max_start = n - (L - 1) * d
            for a in range(max_start):
                indices = [a + k * d for k in range(L)]
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    index_aps[idx].append(ap_id)

    num_aps = len(ap_list)
    ap_lengths = [len(ap) for ap in ap_list]
    
    best_s = None
    best_B = float('inf')
    
    # We know B must be at least 1.
    # Let's try to find the minimum B.
    # Based on previous runs, B is likely around 5-10.
    # We will try B = 1, 2, ... up to a limit.
    # To speed up, we can use a randomized restart strategy or just tight backtracking.
    
    def solve(B):
        s = [-1] * n
        ap_sums = [0] * num_aps
        unassigned_counts = ap_lengths[:]
        
        # Variable ordering: 
        # Indices with higher degree (more APs) are more constrained.
        # However, for APs, index order is often sufficient and faster due to cache locality.
        # Let's try index order 0..n-1.
        
        # Value ordering: try the sign that keeps sums closer to 0.
        
        def is_consistent(k, sigma):
            # Check if assigning s[k]=sigma violates any AP constraint
            for ap_id in index_aps[k]:
                new_sum = ap_sums[ap_id] + sigma
                remaining = unassigned_counts[ap_id] - 1
                # The sum can change by at most 'remaining' in the future
                # So final sum is in [new_sum - remaining, new_sum + remaining]
                # We need |final_sum| <= B
                # So we need:
                # new_sum - remaining <= B  AND  new_sum + remaining >= -B
                # i.e., new_sum <= B + remaining AND new_sum >= -B - remaining
                if new_sum > B + remaining or new_sum < -B - remaining:
                    return False
            return True

        def bt(k):
            nonlocal s, ap_sums, unassigned_counts
            if time.time() - start_time > time_limit - 1.0:
                return False
            if k == n:
                return True
            
            # Heuristic: Try the value that minimizes the maximum absolute sum change
            # or simply alternate based on current sums.
            
            # For speed, we can just try 1 then -1, or order by heuristic.
            # Heuristic: calculate impact on APs.
            # Let's just try both, but maybe order them.
            
            # To improve performance, we can use "fail-first" or similar.
            # But simple forward checking is a good start.
            
            for sigma in [1, -1]:
                if is_consistent(k, sigma):
                    s[k] = sigma
                    for ap_id in index_aps[k]:
                        ap_sums[ap_id] += sigma
                        unassigned_counts[ap_id] -= 1
                    if bt(k + 1):
                        return True
                    for ap_id in index_aps[k]:
                        ap_sums[ap_id] -= sigma
                        unassigned_counts[ap_id] += 1
                    s[k] = -1
            return False

        return bt(0)

    # Try increasing B
    # Start from B=1. If it fails quickly, try B=2, etc.
    # We can limit the search time for each B.
    
    B = 1
    while B <= 15 and time.time() - start_time < time_limit:
        # Reset time check for this B? No, global limit.
        # We can give each B a small time slice? 
        # Backtracking can be exponential.
        # If B is too small, it will fail fast? Not necessarily.
        # Let's just run it.
        
        # Optimization: If B=1 fails, it's likely impossible for N=256.
        # Actually, for N=3, min B=2.
        # For N=256, min B is definitely > 1.
        # We can skip B=1, 2, 3?
        # Let's just try them, they should fail fast if we have good pruning.
        
        if solve(B):
            best_s = s[:]
            best_B = B
            break
        else:
            B += 1

    if best_s is None:
        # Fallback to a greedy or random solution if CSP fails
        # Greedy: assign s[i] to minimize max sum so far
        s = [0] * n
        ap_sums = [0] * num_aps
        # This is complex to do greedily correctly.
        # Just use random with local search?
        rng = random.Random(42)
        best_s = [rng.choice([-1, 1]) for _ in range(n)]
        # Improve with local search
        # Calculate current max discrepancy
        def calc_max_disc(s):
            max_d = 0
            # Iterate over all APs
            for ap in ap_list:
                sum_val = 0
                for idx in ap:
                    sum_val += s[idx]
                if abs(sum_val) > max_d:
                    max_d = abs(sum_val)
            return max_d

        current_disc = calc_max_disc(best_s)
        # Simple local search: flip a random index if it reduces max disc
        # This is slow if done naively.
        # Given time constraints, just return random if CSP fails.
        # But CSP should work.

    return best_s

# EVOLVE-BLOCK-END
