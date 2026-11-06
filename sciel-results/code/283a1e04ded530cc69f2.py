import sys
import time

def construct():
    start_time = time.time()
    deadline = 10.0 # seconds
    
    # We know the answer is likely 36.
    # Let's search for a set of size 9 with max element <= 36.
    
    def is_valid_sidon(a):
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i+1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
        
    def covers_1_to_30(a):
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(a[j] - a[i])
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    # Backtracking search for a set of size 9 in [0, 36]
    # We must include 0.
    # We want to minimize the max element.
    # Since 36 is the lower bound, we try to find a set with max 36.
    
    target_size = 9
    max_val = 36
    
    # We can use a simple recursive search.
    # To prune, we check if the current set of differences already has duplicates.
    # And we check if it's possible to cover 1..30 with the remaining elements.
    
    best_set = None
    
    def dfs(current_set, current_diffs, next_val):
        nonlocal best_set
        if time.time() - start_time > deadline:
            return
        
        if len(current_set) == target_size:
            if covers_1_to_30(current_set):
                best_set = current_set[:]
                return
            return
        
        # Pruning:
        # If we have k elements, we have k*(k-1)/2 differences.
        # We need to cover 1..30.
        # If the number of missing differences from 1..30 is greater than the number of new differences we can add, prune.
        # New elements: target_size - len(current_set)
        # New differences added by m new elements to a set of size k is m*k + m*(m-1)/2.
        # Actually, it's simpler to just check if we can still reach 30 covered differences.
        missing = [d for d in range(1, 31) if d not in current_diffs]
        if not missing:
             # We already cover 1..30. We just need to fill up to size 9 without breaking Sidon.
             # But we must maintain Sidon.
             pass
        
        # Heuristic: The set must be "dense" in differences.
        
        # Try adding next_val
        # We can iterate from next_val to max_val
        
        for v in range(next_val, max_val + 1):
            # Check Sidon property with new v
            valid = True
            new_diffs = set(current_diffs)
            for x in current_set:
                d = v - x
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                dfs(current_set + [v], new_diffs, v + 1)
                if best_set is not None:
                    return

    # Start with 0
    # To make it faster, we can fix a few more elements if we know a pattern,
    # but 9 elements in 0..36 is small enough for naive DFS with pruning.
    # However, naive DFS might be slow if it explores many invalid branches.
    # Let's add a stronger prune: 
    # The largest difference in the final set must be at least 30.
    # Also, the sum of the smallest k differences must be <= max_val? No.
    
    # A known good starting point for Sidon sets is to start with small gaps.
    # Let's just run it.
    
    dfs([0], set(), 1)
    
    if best_set:
        return best_set
    
    # If not found in 10s, fallback to a greedy construction which might have larger diameter
    # But 36 is the theoretical min, so it should find it.
    
    # Fallback: Greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
