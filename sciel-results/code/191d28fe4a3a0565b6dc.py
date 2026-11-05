import sys
import time

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with the minimum possible largest element.
    """
    required_diffs_mask = 0
    for d in range(1, 31):
        required_diffs_mask |= (1 << d)
    
    def find_set_for_D(D, k):
        """
        Tries to find a Sidon set of size k in [0, D] covering 1..30.
        Returns the set if found, else None.
        Uses bitmasks for fast difference checking.
        """
        A = [0]
        
        def solve(idx, current_max, diffs_mask, covered_mask):
            if idx == k:
                if (covered_mask & required_diffs_mask) == required_diffs_mask:
                    return A[:]
                return None
            
            # Pruning: Can we fit remaining elements?
            m = k - idx
            if D - current_max < m:
                return None

            # Try each possible next element x
            for x in range(current_max + 1, D + 1):
                valid = True
                new_diffs_mask = 0
                for a in A:
                    d = x - a
                    bit = 1 << d
                    if (diffs_mask & bit) != 0:
                        valid = False
                        break
                    new_diffs_mask |= bit
                
                if not valid:
                    continue
                
                # Update state
                A.append(x)
                next_diffs_mask = diffs_mask | new_diffs_mask
                next_covered_mask = covered_mask | (new_diffs_mask & required_diffs_mask)
                
                res = solve(idx + 1, x, next_diffs_mask, next_covered_mask)
                if res is not None:
                    return res
                
                # Backtrack
                A.pop()
            
            return None
        
        return solve(1, 0, 0, 0)

    # Iterate D upwards
    start_time = time.time()
    timeout = 110
    
    for D in range(30, 500):
        if time.time() - start_time > timeout:
            return greedy_construct()
        
        max_k = int((1 + (1 + 8 * D) ** 0.5) / 2)
        
        found = None
        # Prioritize smaller k? Or larger?
        # Smaller k is more constrained, might fail faster if impossible.
        # But we want to find *a* solution.
        # Usually smaller k has smaller diameter for the same coverage?
        # Let's try k=9 first.
        for k in range(9, min(max_k, 12) + 1):
            result = find_set_for_D(D, k)
            if result is not None:
                found = result
                break
        
        if found is not None:
            return sorted(found)

    return greedy_construct()

def greedy_construct():
    """Fallback greedy constructor."""
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

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)
