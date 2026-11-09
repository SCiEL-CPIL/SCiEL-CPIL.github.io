import time
from math import lcm
from functools import reduce
from itertools import combinations
import random

# Universe is [1, 24]
UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

# Precompute weights
WEIGHTS = {x: L // x for x in UNIVERSE}

def get_subset_sums(S):
    """
    Return a set of all possible sums of subsets of S.
    S is a list of integers (weights).
    """
    sums = {0}
    for w in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        sums.update(new_sums)
    return sums

def greedy_construct(order):
    """
    Greedily add elements in the given order if they don't conflict.
    Returns the set of values.
    """
    current_values = []
    current_weights = []
    current_sums = {0}
    
    for x in order:
        w = WEIGHTS[x]
        # Check if w is in current_sums
        if w not in current_sums:
            current_values.append(x)
            current_weights.append(w)
            # Update sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + w)
            current_sums = new_sums
            
    return current_values

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Greedy Phase to find a good initial solution
    best_size = 0
    best_set = []
    
    # Try several orders
    orders = []
    orders.append(sorted(UNIVERSE, reverse=True))
    orders.append(sorted(UNIVERSE))
    
    # Random orders
    random.seed(42)
    for _ in range(20):
        o = UNIVERSE[:]
        random.shuffle(o)
        orders.append(o)
        
    for o in orders:
        if time.time() > deadline:
            break
        s = greedy_construct(o)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
    # If greedy didn't find anything (shouldn't happen), fallback
    if not best_set:
        best_set = [1]
        best_size = 1

    # 2. DFS Phase to find optimal or larger solution
    # We use the best_size from greedy as the initial bound
    
    # Sort candidates. 
    # Heuristic: Elements that are "harder" to place might be better placed earlier?
    # Or "easier" ones?
    # Let's stick to descending value for now, as it seemed to work in the seed.
    candidates = sorted(UNIVERSE, reverse=True)
    
    def time_exceeded():
        return time.time() > deadline

    def dfs(idx, current_values, current_weights, current_sums):
        nonlocal best_size, best_set
        
        if time_exceeded():
            return
        
        # Update best
        if len(current_values) > best_size:
            best_size = len(current_values)
            best_set = current_values[:]
            
        # Pruning: Upper bound on size
        if len(current_values) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            return
        
        # Option 1: Skip candidates[idx]
        dfs(idx + 1, current_values, current_weights, current_sums)
        
        if time_exceeded():
            return
            
        # Option 2: Include candidates[idx]
        x = candidates[idx]
        w = WEIGHTS[x]
        
        if w in current_sums:
            return
        
        new_values = current_values + [x]
        new_weights = current_weights + [w]
        
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
            
        dfs(idx + 1, new_values, new_weights, new_sums)

    # Initialize DFS
    # We can also start DFS from the greedy solution? 
    # No, DFS explores from scratch. But the bound is tight.
    # To speed up DFS, we can also try to "fix" the greedy solution?
    # i.e., start DFS with the elements of best_set already included?
    # This is tricky because the order in DFS is fixed by `candidates`.
    # If we force include some elements, we must ensure they are in the correct order in the candidate list.
    
    # Let's just run DFS from scratch with the improved bound.
    dfs(0, [], [], {0})
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Best set size: {len(s)}")
    print(f"Set: {s}")
