import time
from collections import defaultdict

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    timeout = 85
    
    # Generate multiples
    # Limit: The optimal range is small. Values should be small.
    # 10000 is safe.
    limit = 10000
    all_mults = []
    for i, d in enumerate(divisors):
        for m in range(d, limit + 1, d):
            all_mults.append((m, i))
            
    # Sort by value
    all_mults.sort(key=lambda x: x[0])
    
    # Sliding Window to find top K smallest spans covering all divisors
    # We want the minimum max-min.
    # The span is determined by the min and max value in the window.
    
    best_windows = [] # List of (span, L_val, R_val)
    
    counts = [0] * n
    active = 0
    left = 0
    
    for right in range(len(all_mults)):
        val_r, idx_r = all_mults[right]
        counts[idx_r] += 1
        if counts[idx_r] == 1:
            active += 1
            
        while active == n:
            val_l, idx_l = all_mults[left]
            span = val_r - val_l
            
            # We can optimize: if we already have a span smaller than this, 
            # and we are shrinking left, the span will only increase.
            # But we want to record the *best* windows.
            # Actually, for a fixed right, as left increases, span increases.
            # So the smallest span for this right is when left is as large as possible.
            # But we also need to ensure we don't miss windows where left is smaller?
            # No, if left is smaller, span is larger.
            # So for each valid right, the "best" left is the maximum left such that active==n.
            # Wait, the while loop shrinks left until active < n.
            # So the last valid state before active < n is the one with max left (min span) for this right.
            
            # So we only need to record the state when we are about to break the while loop?
            # Or just record every valid state?
            # Recording every valid state is O(N*K) roughly.
            # Let's just record the span when active == n.
            
            # To avoid duplicates and keep list small, we can just track the global minimum span found.
            # But we might need backup if assignment fails.
            # Let's store the best 10 spans found.
            
            if len(best_windows) < 10 or span < best_windows[-1][0]:
                best_windows.append((span, val_l, val_r))
                best_windows.sort(key=lambda x: x[0])
                if len(best_windows) > 10:
                    best_windows.pop()
            
            # Shrink left
            counts[idx_l] -= 1
            if counts[idx_l] == 0:
                active -= 1
            left += 1
            
            if time.time() - start_time > timeout:
                break
        
        if time.time() - start_time > timeout:
            break
            
    # If no window found, fallback
    if not best_windows:
        return divisors[:]
        
    # Try to solve assignment for the best windows
    for span, L, R in best_windows:
        # Collect candidates
        candidates = [[] for _ in range(n)]
        # We need to find all multiples in [L, R] for each divisor.
        # Since all_mults is sorted, we can slice it.
        # But we need to filter by divisor index.
        
        # Efficient way: iterate all_mults in [L, R] range?
        # We can find indices using bisect on values, but all_mults is (val, idx).
        # Let's just iterate the slice of all_mults that falls in [L, R].
        
        # Find start index in all_mults for L
        # Since L is from all_mults, we can find it? 
        # Better: just iterate all_mults and filter? No, O(N).
        # N is ~5000. 10 windows * 5000 = 50k ops. Fast.
        
        for val, idx in all_mults:
            if L <= val <= R:
                candidates[idx].append(val)
        
        # Check if all have candidates
        if any(len(c) == 0 for c in candidates):
            continue
            
        # Solve Assignment
        # Sort indices by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [None] * n
        
        def backtrack(k):
            if k == n:
                return True
            idx = order[k]
            for val in candidates[idx]:
                if val not in used:
                    used.add(val)
                    assignment[idx] = val
                    if backtrack(k + 1):
                        return True
                    used.remove(val)
            return False
            
        if backtrack(0):
            result = [assignment[i] for i in range(n)]
            # Final verification
            if len(set(result)) == n:
                for i in range(n):
                    if result[i] % divisors[i] != 0:
                        continue # Should not happen
                return result
    
    # Fallback if all windows failed assignment (very unlikely)
    # Just return the first window's greedy assignment or something?
    # Or just return divisors?
    return divisors[:]
