"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    best_result = None
    best_max = float('inf')
    
    # We need a Sidon set containing 0, with all differences 1..30 present.
    # Minimum size: C(n,2) >= 30 => n >= 9
    # We'll search for the minimum diameter (max element).
    
    def is_sidon_partial(arr):
        """Check if arr (sorted) has all pairwise differences distinct."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def can_complete(arr, remaining_needed, max_val):
        """Check if it's possible to complete arr to cover all remaining_needed diffs within max_val."""
        n = len(arr)
        # We can add at most (max_val - arr[-1]) more elements, but bounded by what's left
        # Remaining pairs we can form: if we add k more elements, total pairs = C(n+k, 2) - C(n,2)
        # We need at least len(remaining_needed) more distinct differences
        # Simple lower bound: need at least len(remaining_needed) more pairs
        # Each new element adds (n + i) new differences for the i-th new element (0-indexed from current)
        # So adding k elements adds k*n + C(k,2) new differences
        # We need k*n + C(k,2) >= len(remaining_needed)
        # Also, all new elements must be <= max_val
        needed = len(remaining_needed)
        if needed == 0:
            return True
        
        # Try to see if we can fit enough elements
        # The new elements must be > arr[-1] and <= max_val
        # Max number of new elements: limited by max_val
        # For simplicity, check if there are enough candidate positions
        # A new element x > arr[-1] creates differences x - arr[i] for all i
        # The smallest new difference from x is x - arr[-1]
        
        # Lower bound on k: solve k*n + k*(k-1)/2 >= needed
        k = 0
        while k * n + k * (k - 1) // 2 < needed:
            k += 1
            if k > 50:  # sanity check
                return False
        
        # Can we place k distinct integers in (arr[-1], max_val]?
        available_space = max_val - arr[-1]
        if available_space < k:
            return False
        
        return True
    
    def backtrack(arr, used_diffs, remaining_needed, max_val):
        """Recursively build the Sidon set."""
        nonlocal best_result, best_max, deadline
        
        if time.time() > deadline:
            return False
        
        if not remaining_needed:
            # All differences covered
            current_max = arr[-1]
            if current_max < best_max:
                best_max = current_max
                best_result = arr[:]
            return True
        
        # Pruning: can we complete?
        if not can_complete(arr, remaining_needed, max_val):
            return False
        
        # If we have enough elements and all needed diffs are covered, we're done
        # Try to add the next smallest possible value
        # The next element must be > arr[-1]
        # It must create at least one new difference in remaining_needed
        
        last = arr[-1]
        # Try values from last+1 upward
        # But we can prune: the new value x must satisfy that at least one of (x - arr[i]) is in remaining_needed
        # Also, all (x - arr[i]) must not be in used_diffs
        
        # Find the minimum possible x that creates a useful difference
        # For each i, x = arr[i] + d for d in remaining_needed
        # x must be > last
        candidates = set()
        for i in range(len(arr)):
            for d in remaining_needed:
                x = arr[i] + d
                if x > last:
                    candidates.add(x)
        
        if not candidates:
            return False
        
        # Try candidates in increasing order
        for x in sorted(candidates):
            if x > max_val:
                break
            
            # Check Sidon property: all (x - arr[i]) must be distinct and not in used_diffs
            new_diffs = []
            valid = True
            for i in range(len(arr)):
                d = x - arr[i]
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check which remaining_needed are satisfied
            new_remaining = set(remaining_needed)
            for d in new_diffs:
                new_remaining.discard(d)
            
            # Recurse
            if backtrack(arr + [x], used_diffs | set(new_diffs), new_remaining, max_val):
                return True
            
            # Deadline check
            if time.time() > deadline:
                return False
        
        return False
    
    # Search strategy: try increasing max_val until we find a solution
    # Start with a reasonable lower bound
    # With 9 elements, the differences go up to at least 30, so max >= 30
    # But we need all diffs 1..30, so the set must span at least 30
    # Lower bound: max element >= 30
    
    # Let's try max_val from 30 upward
    for max_val in range(30, 1000000):
        if time.time() > deadline:
            break
        
        # Reset for this max_val
        # Start with [0]
        initial_arr = [0]
        initial_used = set()
        initial_remaining = set(range(1, 31))
        
        result = backtrack(initial_arr, initial_used, initial_remaining, max_val)
        
        if result:
            if best_result is not None:
                return best_result
            else:
                return [0]  # fallback
    
    # If we didn't find a solution within time, return best found or fallback
    if best_result is not None:
        return best_result
    
    # Fallback: use the original greedy approach
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
# EVOLVE-BLOCK-END
