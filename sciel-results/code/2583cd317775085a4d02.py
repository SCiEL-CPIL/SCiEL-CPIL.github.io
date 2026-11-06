import time
import sys

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s for overhead
    
    # Global variables for B&B
    best_size = 0
    best_set = set()
    
    # Greedy initialization to get a good lower bound
    # Try multiple greedy strategies
    def greedy_heuristic(order):
        S = set()
        for v in order:
            can_add = True
            # Check validity
            for a in S:
                for b in S:
                    if a >= b: continue
                    # Check triple (a, b, v)
                    # Sort a, b, v
                    triple = sorted([a, b, v])
                    x, y, z = triple
                    # Condition: x does NOT divide y+z
                    if (y + z) % x == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.add(v)
        return S

    # Try descending and ascending
    S1 = greedy_heuristic(range(n, 0, -1))
    S2 = greedy_heuristic(range(1, n + 1))
    
    if len(S1) > len(S2):
        best_set = S1
        best_size = len(S1)
    else:
        best_set = S2
        best_size = len(S2)

    # B&B
    # Use list for set for faster iteration? Or set.
    # We'll use a list of elements in the current set for O(1) access? No, set is fine.
    # To speed up, we can precompute divisors?
    
    # Optimization: Check if v can be added to S
    def can_add(v, S):
        # S is a set
        # We need to check all pairs (a, b) in S
        # And the triple formed by a, b, v
        # To avoid sorting every time, we can categorize
        # 1. v is smallest: v < a <= b. Check v | (a+b)
        # 2. v is middle: a < v <= b. Check a | (v+b)
        # 3. v is largest: a < b <= v. Check a | (b+v)
        
        # Efficient check:
        # Iterate through a in S
        # For each a, iterate through b in S where b > a
        # This is O(|S|^2)
        
        # We can optimize by early breaking
        for a in S:
            for b in S:
                if a >= b:
                    continue
                
                # Determine the sorted order
                if v < a:
                    # v, a, b
                    if (a + b) % v == 0:
                        return False
                elif a < v <= b:
                    # a, v, b
                    if (v + b) % a == 0:
                        return False
                else: # a < b <= v
                    # a, b, v
                    if (b + v) % a == 0:
                        return False
        return True

    # Recursive B&B
    # To prevent stack overflow, we might need to limit depth or use iterative, but 256 is fine.
    # We need to track the current set.
    
    # We'll use a list to store the current path/set
    current_set = set()
    
    # Order: Descending seems to find large sets faster? Or Ascending?
    # Let's try Descending first.
    vertices = list(range(n, 0, -1))
    
    nodes_visited = 0
    
    def search(idx, current_set):
        nonlocal best_size, best_set, nodes_visited
        
        if time.time() - start_time > time_limit:
            return
        
        nodes_visited += 1
        
        # Pruning 1: Upper bound
        # Max possible size = current_size + (n - idx)
        if len(current_set) + (n - idx) <= best_size:
            return
            
        # Base case
        if idx == n:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        v = vertices[idx]
        
        # Option 1: Include v
        if can_add(v, current_set):
            current_set.add(v)
            search(idx + 1, current_set)
            current_set.remove(v)
            
        # Option 2: Exclude v
        search(idx + 1, current_set)

    # Run B&B
    # To make it faster, we can add a heuristic: if including v leads to a dead end quickly, exclude it?
    # The standard B&B above explores Include first.
    
    # One issue: `can_add` is O(|S|^2). 
    # If |S| is 100, 10k ops.
    # 2^256 is huge. We rely on pruning.
    
    # Let's add a stronger pruning:
    # If we include v, we might block many future vertices.
    # We can compute a "greedy completion" upper bound.
    # But for now, let's run the basic B&B with time limit.
    
    # To improve performance, we can skip `can_add` if we know it will fail?
    # No, we need to check.
    
    # Let's run it.
    search(0, current_set)
    
    return list(best_set)
