import time
import random
import sys

def _is_valid_set(S):
    """
    Checks if no element in S is the mean of a subset of at least 2 other elements.
    Uses bitset DP for subset sums with exact count.
    """
    n = len(S)
    if n < 3:
        return True
        
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        
        # dp[j] is a bitset where bit s is set if sum s is achievable with j elements
        dp = [0] * (len(others) + 1)
        dp[0] = 1
        
        for x in others:
            for j in range(len(others) - 1, -1, -1):
                if dp[j]:
                    dp[j+1] |= (dp[j] << x)
                    
        for k in range(2, len(others) + 1):
            target_sum = k * a
            if (dp[k] >> target_sum) & 1:
                return False
                
    return True

def _count_violations(S):
    """
    For each element in S, count how many other elements it violates (i.e., is the mean of a subset of them).
    Returns a list of violation counts corresponding to indices in S.
    """
    n = len(S)
    if n < 3:
        return [0] * n
        
    violation_counts = [0] * n
    
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        # Map value to index in S for quick lookup if needed, but here we just check sums
        
        dp = [0] * (len(others) + 1)
        dp[0] = 1
        
        for x in others:
            for j in range(len(others) - 1, -1, -1):
                if dp[j]:
                    dp[j+1] |= (dp[j] << x)
                    
        count = 0
        for k in range(2, len(others) + 1):
            target_sum = k * a
            if (dp[k] >> target_sum) & 1:
                count += 1
        violation_counts[i] = count
        
    return violation_counts

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    best_set = []
    
    # Strategy: Randomized Local Search
    # 1. Start with a random subset of [1, 512]
    # 2. While invalid, remove the element with the most violations
    # 3. Repeat with different random seeds
    
    rng = random.Random(42)
    
    # Try multiple random initializations
    for seed in range(1000):
        if time.time() > deadline:
            break
            
        if seed == 0:
            rng = random.Random(42)
        else:
            rng = random.Random(seed)
            
        # Start with a fairly large random set
        # Size 100-200 is a good start, as we expect to prune down to ~48
        initial_size = rng.randint(80, 150)
        S = rng.sample(range(1, 513), initial_size)
        
        # Local search loop
        # We want to maximize cardinality. 
        # Removing an element might allow keeping others? 
        # No, the property is monotonic: if S is valid, any subset is valid. 
        # If S is invalid, removing an element makes it "less invalid" (in terms of constraints), 
        # but we want to keep as many as possible.
        # Greedy removal of the "most problematic" element is a standard heuristic for hitting set / independent set type problems.
        
        current_S = list(S)
        
        while len(current_S) > 0:
            if time.time() > deadline:
                break
                
            # Check if valid
            if _is_valid_set(current_S):
                break
                
            # Find element with max violations
            violations = _count_violations(current_S)
            max_violations = max(violations)
            
            # If max violations is 0, it means it's valid? 
            # Wait, _is_valid_set returns False if ANY element is mean of subset.
            # _count_violations counts how many OTHER elements are means of subsets of the rest?
            # Let's re-read _is_valid_set logic.
            # "No element may be the exact arithmetic mean of any subset of at least two OTHER elements."
            # My _count_violations implementation:
            # For element a (index i), it checks if a is mean of subset of others.
            # It increments count if such a subset exists.
            # So count > 0 means a is invalid in the context of current_S.
            
            if max_violations == 0:
                # Should be valid, but _is_valid_set said False? 
                # Maybe _is_valid_set logic and _count_violations logic differ slightly?
                # _is_valid_set returns False on first violation found.
                # _count_violations counts all.
                # If max is 0, no element is a mean of any subset. So it should be valid.
                # If this happens, break.
                break
                
            # Remove one of the elements with max violations
            # Pick randomly among those with max violations to escape local minima
            candidates = [j for j, v in enumerate(violations) if v == max_violations]
            remove_idx = rng.choice(candidates)
            current_S.pop(remove_idx)
            
        if len(current_S) > len(best_set):
            best_set = current_S
            
    # If we haven't found a good set, try a greedy fill on the best set found so far?
    # Or just return what we have.
    # The local search usually finds a maximal set quickly.
    
    # One final polish: try to add elements from [1, 512] \ best_set to best_set
    if len(best_set) < 48:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        # Sort remaining by some heuristic? Maybe just random or increasing
        # Let's try increasing order first for determinism in the polish
        for x in remaining:
            if time.time() > deadline:
                break
            new_S = best_set + [x]
            if _is_valid_set(new_S):
                best_set = new_S
            if len(best_set) == 48:
                break

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    if len(s) > 0:
        print(f"Valid: {_is_valid_set(s)}")
