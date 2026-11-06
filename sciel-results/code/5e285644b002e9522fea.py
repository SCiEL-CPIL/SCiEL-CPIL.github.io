"""Rewritten program for Sidon set with differences 1..30."""
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30.
    Minimize the largest element.
    
    Strategy: Deterministic Greedy Construction followed by Deterministic Local Search.
    1. Greedy: Build a set of 32 elements by always picking the smallest integer that
       maintains the Sidon property and covers the most missing differences (prioritizing small ones).
    2. Local Search: Systematically try to move each element to a smaller position
       to reduce the maximum element, maintaining Sidon and difference coverage.
    """
    
    def get_diffs(arr):
        """Get all positive differences as a set."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        return diffs

    def is_sidon(arr):
        """Check if arr is a Sidon set."""
        diffs = []
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.append(d)
        return True

    def has_all_diffs(arr, target_diffs):
        """Check if all target differences are present."""
        diff_set = get_diffs(arr)
        return all(d in diff_set for d in target_diffs)

    def greedy_construct():
        """
        Deterministic greedy construction.
        Prioritizes covering small missing differences.
        """
        a = [0]
        target_diffs = set(range(1, 31))
        
        for step in range(31):
            # Find the best candidate x
            best_x = None
            best_score = float('inf')
            
            # We want to keep the set small, so search from 1 upwards
            # Upper bound: current max + 100 is usually enough for greedy
            # But to be safe, we might need to go higher if stuck.
            # Let's search up to max(a) + 500 initially.
            upper_bound = max(a) + 500 if a else 500
            
            # If we are stuck (no candidate found in range), expand range
            # But for now, assume 500 is enough for the first few steps.
            
            candidates = range(1, upper_bound + 1)
            
            for x in candidates:
                if x in a:
                    continue
                
                trial = sorted(a + [x])
                if not is_sidon(trial):
                    continue
                
                trial_diffs = get_diffs(trial)
                # Count how many target diffs are covered
                covered = len(target_diffs & trial_diffs)
                
                # Score: minimize x, but strongly prefer covering more diffs
                # If we cover a new diff, it's very good.
                # We want to cover 1..30.
                # Score = x - 1000 * covered
                score = x - 1000 * covered
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # If no candidate found, expand search range
                # This should rarely happen if we start small
                for x in range(upper_bound + 1, upper_bound + 1000):
                    if x in a:
                        continue
                    trial = sorted(a + [x])
                    if not is_sidon(trial):
                        continue
                    trial_diffs = get_diffs(trial)
                    covered = len(target_diffs & trial_diffs)
                    score = x - 1000 * covered
                    if score < best_score:
                        best_score = score
                        best_x = x
            
            if best_x is None:
                # Fallback: just pick a large number that works
                x = max(a) + 1
                while True:
                    trial = sorted(a + [x])
                    if is_sidon(trial):
                        break
                    x += 1
                best_x = x
            
            a = sorted(a + [best_x])
        
        return a

    def local_search(arr):
        """
        Deterministic local search to minimize max element.
        Iterates through elements and tries to move them to smaller positions.
        """
        best = sorted(arr)
        improved = True
        
        while improved:
            improved = False
            # Sort by value descending to shrink the largest elements first
            indices = sorted(range(len(best)), key=lambda i: best[i], reverse=True)
            
            for idx in indices:
                if idx == 0:
                    continue # Don't move 0
                
                old_val = best[idx]
                # Try to move to any smaller position
                # We can try positions from 1 to old_val - 1
                # To speed up, we can check if a smaller valid position exists
                
                # Optimization: Check if we can move to a specific smaller spot
                # We want the smallest possible value for this element that keeps validity
                # But moving one element might break others.
                
                # Try all positions from 1 to old_val - 1
                # This is O(N^2) per move, which is fine for N=32.
                found_better = False
                for new_val in range(1, old_val):
                    if new_val in best:
                        continue
                    
                    trial = best[:]
                    trial[idx] = new_val
                    trial = sorted(trial)
                    
                    if not is_sidon(trial):
                        continue
                    
                    if not has_all_diffs(trial, range(1, 31)):
                        continue
                    
                    # If valid, accept the first (smallest) new_val found?
                    # No, we want to minimize the GLOBAL max.
                    # Moving this element to a smaller value reduces max if it was the max.
                    # If it's not the max, it might not reduce max, but it frees up space.
                    # However, we are iterating largest to smallest.
                    # So if we move the largest element to a smaller value, max decreases.
                    # We should accept the move if it reduces the max element of the set.
                    
                    new_max = max(trial)
                    old_max = max(best)
                    
                    if new_max < old_max:
                        best = trial
                        improved = True
                        found_better = True
                        break # Restart the loop as the set changed
                
                # If we didn't reduce max, should we still move to a smaller value?
                # It might help future moves.
                # But for simplicity and speed, let's only accept moves that reduce the current max.
                # This is a greedy hill-climbing on the max element.
        
        return best

    # 1. Greedy Construction
    initial = greedy_construct()
    
    # Ensure it's valid
    if not is_sidon(initial) or not has_all_diffs(initial, range(1, 31)):
        # If greedy failed (unlikely), try a simpler fallback
        # But greedy should work.
        pass

    # 2. Local Search
    result = local_search(initial)
    
    # Final check
    if not is_sidon(result) or not has_all_diffs(result, range(1, 31)):
        # Fallback to initial
        result = initial

    return sorted(result)
# EVOLVE-BLOCK-END
