import math
import sys

def construct():
    # Calculate LCM of 1 to 22
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    D = 1
    for i in range(1, 23):
        D = lcm(D, i)
        
    vals = [D // i for i in range(1, 23)]
    
    # Greedy initialization to set a good lower bound for pruning
    current_sums = {0}
    current_set = []
    for x in range(22, 0, -1):
        inc = D // x
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + inc)
        if new_sums.isdisjoint(current_sums):
            current_sums.update(new_sums)
            current_set.append(x)
            
    best_set = current_set[:]
    best_size = len(best_set)
    
    sys.setrecursionlimit(10000)
    
    def dfs(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        # Pruning: if we can't beat the best size, return
        if len(current_set) + (22 - idx) <= best_size:
            return
            
        if idx == 22:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Option 1: Include vals[idx]
        # Generate new sums
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + vals[idx])
            
        if new_sums.isdisjoint(current_sums):
            new_set = current_set + [idx + 1]
            new_sums_updated = current_sums.union(new_sums)
            dfs(idx + 1, new_set, new_sums_updated)
            
        # Option 2: Exclude vals[idx]
        dfs(idx + 1, current_set, current_sums)

    dfs(0, [], {0})
    
    return best_set
