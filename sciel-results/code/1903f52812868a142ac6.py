from fractions import Fraction
import sys

sys.setrecursionlimit(10000)

def construct():
    candidates = list(range(24, 0, -1))
    n = len(candidates)
    recips = [Fraction(1, k) for k in candidates]
    
    best_size = 0
    best_set = []
    
    def dfs(idx, current_set, current_recips, current_sums, current_size):
        nonlocal best_size, best_set
        
        if current_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        k = candidates[idx]
        r_k = recips[idx]
        
        # Exclude k
        dfs(idx + 1, current_set, current_recips, current_sums, current_size)
        
        # Include k
        if r_k in current_sums:
            return
            
        valid = True
        # Check conflicts with existing elements
        # We need 1/x - 1/k not in current_sums for all x in current_set
        for r_x in current_recips:
            if r_x > r_k:
                diff = r_x - r_k
                if diff in current_sums:
                    valid = False
                    break
        
        if not valid:
            return
            
        new_set = current_set + [k]
        new_recips = current_recips + [r_k]
        
        new_sums = set(current_sums)
        # Add new sums
        # We can iterate over a list to avoid "set changed size" issues if we were modifying in place, 
        # but we are creating a new set.
        for s in current_sums:
            new_sums.add(s + r_k)
            
        dfs(idx + 1, new_set, new_recips, new_sums, current_size + 1)
        
    # Greedy initial solution
    g_set = []
    g_recips = []
    g_sums = {Fraction(0)}
    for i in range(n):
        k = candidates[i]
        r_k = recips[i]
        if r_k in g_sums:
            continue
        valid = True
        for r_x in g_recips:
            if r_x > r_k:
                if (r_x - r_k) in g_sums:
                    valid = False
                    break
        if valid:
            g_set.append(k)
            g_recips.append(r_k)
            new_sums = set(g_sums)
            for s in g_sums:
                new_sums.add(s + r_k)
            g_sums = new_sums
            
    if g_set:
        best_size = len(g_set)
        best_set = g_set[:]
        
    # DFS
    dfs(0, [], [], {Fraction(0)}, 0)
    
    return best_set

