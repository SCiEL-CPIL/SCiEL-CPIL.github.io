"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors ascending for binary search
    divisors.sort()
    N = len(divisors)
    
    # Helper to check if a value exists in divisors[start_idx:]
    def exists_in_range(val, start_idx):
        # Binary search in divisors[start_idx:]
        # We need val to be in divisors and its index > start_idx
        # Since divisors is sorted, we can bisect in the whole list and check index
        idx = bisect.bisect_left(divisors, val)
        if idx < N and divisors[idx] == val and idx > start_idx:
            return True
        return False

    # Try k=1
    if exists_in_range(target, -1):
        return [target]

    # Try k=2
    # d1 + d2 = target
    # Iterate d1
    for i in range(N):
        d1 = divisors[i]
        if d1 >= target:
            break
        d2 = target - d1
        if d2 > d1 and exists_in_range(d2, i):
            return [d1, d2]

    # Try k=3
    # d1 + d2 + d3 = target
    for i in range(N):
        d1 = divisors[i]
        if d1 > target // 3:
            break
        for j in range(i + 1, N):
            d2 = divisors[j]
            if d1 + d2 >= target:
                break
            d3 = target - d1 - d2
            if d3 > d2 and exists_in_range(d3, j):
                return [d1, d2, d3]

    # Try k=4
    # d1 + d2 + d3 + d4 = target
    for i in range(N):
        d1 = divisors[i]
        if d1 > target // 4:
            break
        for j in range(i + 1, N):
            d2 = divisors[j]
            if d1 + d2 > target // 2:
                break
            for k in range(j + 1, N):
                d3 = divisors[k]
                if d1 + d2 + d3 >= target:
                    break
                d4 = target - d1 - d2 - d3
                if d4 > d3 and exists_in_range(d4, k):
                    return [d1, d2, d3, d4]

    # If k < 5 not found, use DFS for k=5..N or fallback
    # We know greedy finds a valid solution of size 5.
    # We can just return the greedy solution if we are confident it's minimal or just valid.
    # But to be safe and potentially find a better k=5 solution (though count is same), 
    # or if greedy fails (it shouldn't), we can run a DFS for k=5.
    # Given time constraints, and that greedy worked in analysis, we can return greedy.
    # However, to ensure we don't miss a k=5 solution that is "better" in some other metric? 
    # No, metric is count.
    
    # Let's try to find a k=5 solution with DFS to be sure, but with a tight time limit.
    # Or just return the greedy solution which is valid.
    
    # Greedy solution
    divisors_desc = divisors[::-1]
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
    
    # If greedy failed (should not happen), fallback to DFS for larger k
    # But for this specific instance, greedy works.
    
    # To be absolutely safe against "greedy not minimal" (which it isn't, count is 5),
    # and if there was a k=5 solution, greedy finds one.
    # If there was no k=5 solution, greedy would fail to reach 0.
    # Since 4979 is representable, and greedy is a valid heuristic, it likely works.
    
    # If we want to be rigorous, we can run a DFS for k=5.
    # But 5 is small enough that the previous DFS would have found it if it existed.
    # The previous DFS iterates k=1, 2, 3, 4, 5...
    # If my explicit checks for 1-4 fail, then k_min >= 5.
    # The seed program would have found k=5 via DFS.
    # So returning a valid k=5 solution is optimal.
    
    return chosen
# EVOLVE-BLOCK-END
