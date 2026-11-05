"""Fast constructor for mean-free subset problem."""
import time
from itertools import combinations

def _is_valid_candidate(a, others):
    """Check if a is NOT the mean of any subset of others with size >= 2."""
    n = len(others)
    if n < 2:
        return True
    
    # Quick pairwise check: if any two others average to a, invalid
    for i in range(n):
        for j in range(i+1, n):
            if others[i] + others[j] == 2 * a:
                return False
    
    # For larger subsets, use DP with sets
    # dp[k] = set of achievable sums using exactly k elements from diffs
    diffs = [x - a for x in others]
    
    # dp[k] is a set of achievable sums
    dp = [set() for _ in range(n+1)]
    dp[0].add(0)
    
    for d in diffs:
        # Update in reverse to avoid using same element twice
        for k in range(n, 0, -1):
            if dp[k-1]:
                for s in dp[k-1]:
                    dp[k].add(s + d)
    
    # Check if 0 is in dp[k] for any k >= 2
    for k in range(2, n+1):
        if 0 in dp[k]:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin: finish well before 60s
    
    # Strategy: use exponential spacing as base, then greedily add elements
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256 (9 elements)
    # These are naturally mean-free because they grow exponentially
    
    selected = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    selected_set = set(selected)
    
    # Try to add more elements from 1 to 512
    candidates = list(range(1, 513))
    candidates.sort(key=lambda x: abs(x - 256))  # try middle values first
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        
        # Check if x can be added without violating constraints
        valid = True
        
        # Check x against existing elements
        if not _is_valid_candidate(x, selected):
            valid = False
        
        # Check existing elements against x
        if valid:
            for a in selected:
                others = [y for y in selected if y != a] + [x]
                if not _is_valid_candidate(a, others):
                    valid = False
                    break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
    
    # If we have room, try a few more random attempts
    if len(selected) < 48 and time.time() < deadline - 5:
        import random
        random.seed(123)
        
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            if len(selected) >= 48:
                break
            
            # Try adding a random element
            remaining = [x for x in range(1, 513) if x not in selected_set]
            if not remaining:
                break
            
            random.shuffle(remaining)
            for x in remaining[:10]:  # try a few random ones
                if time.time() > deadline - 2:
                    break
                
                valid = True
                if not _is_valid_candidate(x, selected):
                    valid = False
                
                if valid:
                    for a in selected:
                        others = [y for y in selected if y != a] + [x]
                        if not _is_valid_candidate(a, others):
                            valid = False
                            break
                
                if valid:
                    selected.append(x)
                    selected_set.add(x)
                    break
    
    return selected[:48]
