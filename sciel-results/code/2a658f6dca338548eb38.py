import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # 1. Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth.sort() # Ascending order for pruning ease
    n_smooth = len(smooth)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum(smooth[i:])
    suffix_sum = [0] * (n_smooth + 1)
    for i in range(n_smooth - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + smooth[i]
        
    answers = [None] * 128
    
    # Helper to check if x can be added to selected (antichain constraint)
    def can_add(x, selected):
        for y in selected:
            if x % y == 0 or y % x == 0:
                return False
        return True

    def find_min_antichain(target):
        # Filter smooth numbers <= target
        # Since smooth is sorted, find the rightmost index <= target
        # Binary search or simple loop
        max_idx = 0
        for i in range(n_smooth):
            if smooth[i] <= target:
                max_idx = i
            else:
                break
        
        if max_idx < 0:
            return None
            
        # We want the minimum number of terms.
        # We will use DFS. To find the MINIMUM, we can try to find a solution of size 1, then 2, etc.
        # However, DFS naturally finds *a* solution. To ensure it's minimal, 
        # we can prune branches where len(selected) >= current_best.
        
        best_sol = None
        
        def dfs(current_sum, start_idx, selected):
            nonlocal best_sol
            
            # Pruning: If we already have a solution, and current length is not better, stop
            if best_sol is not None and len(selected) >= len(best_sol):
                return
                
            if current_sum == target:
                # Found a solution
                if best_sol is None or len(selected) < len(best_sol):
                    best_sol = list(selected)
                return
                
            if current_sum > target:
                return
                
            # Pruning: If even taking all remaining smooth numbers won't reach target
            if current_sum + suffix_sum[start_idx] < target:
                return
                
            # Iterate through candidates
            for i in range(start_idx, max_idx + 1):
                x = smooth[i]
                
                # If adding x exceeds target, and we are iterating in ascending order, 
                # we can't break here because smaller numbers were already skipped? 
                # No, we are at index i. If x > target - current_sum, then x and all larger 
                # numbers (i+1...) will also be too large? 
                # Wait, smooth is ascending. So if x is too big, all subsequent are too big.
                if current_sum + x > target:
                    break
                
                # Check antichain constraint
                if can_add(x, selected):
                    selected.append(x)
                    dfs(current_sum + x, i + 1, selected)
                    selected.pop()
                    
        dfs(0, 0, [])
        return best_sol

    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: return what we have? Or raise? 
            # The task says "All targets require valid representations".
            # If we run out of time, we should try to be faster or use a heuristic.
            # Given the small size, this should not happen.
            pass
            
        sol = find_min_antichain(n)
        if sol is None:
            # This should not happen for 1..128 with 2,3,5 smooth numbers.
            # 1 is smooth. 1 is an antichain of size 1.
            # Every number n can be represented by n itself if n is smooth.
            # If n is not smooth, it must be a sum.
            # Since 1 is smooth and 1 divides everything, we can't use 1 with anything else.
            # But we can use n if n is smooth.
            # If n is not smooth, e.g., 7. 7 is not smooth.
            # 7 = ? 
            # Smooth numbers <= 7: 1, 2, 3, 4, 5, 6.
            # Antichain sums to 7:
            # 3+4=7. 3 and 4: 4%3!=0, 3%4!=0. Valid. Size 2.
            # So it should always find one.
            pass
        answers[n-1] = sol

    return answers
