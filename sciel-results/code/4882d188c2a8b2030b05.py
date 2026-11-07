"""Rewritten program with DFS search for maximum distinct subset product set."""

def construct():
    import time
    
    # Deadline guard: 60s limit, keep 5s margin
    start_time = time.time()
    deadline = start_time + 55.0
    
    candidates = list(range(2, 41))
    best_set = []
    best_products = {1}
    
    # Pre-compute nothing, just DFS
    
    def dfs(idx, current_set, current_products):
        nonlocal best_set, best_products, deadline
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: if we can't beat the current best
        if len(current_set) + (len(candidates) - idx) <= len(best_set):
            return
            
        if idx == len(candidates):
            if len(current_set) > len(best_set):
                best_set = list(current_set)
                best_products = set(current_products)
            return
        
        # Option 1: Exclude candidates[idx]
        dfs(idx + 1, current_set, current_products)
        
        # Option 2: Include candidates[idx]
        x = candidates[idx]
        
        # Check if adding x creates a collision
        # New products are x * p for all p in current_products
        # Collision if any new product is already in current_products
        
        # Optimization: Check if x is in current_products first (fast fail)
        if x in current_products:
            return
            
        # Generate new products
        # If current_products is large, this might be slow, but max size is small.
        # We can use a set intersection check.
        new_products = set()
        for p in current_products:
            np = x * p
            if np in current_products:
                # Collision found
                return
            new_products.add(np)
        
        # If no collision, recurse
        # We pass a new set to avoid mutation issues in backtracking
        # Copying the set is O(N), which is fine.
        next_products = current_products.union(new_products)
        
        current_set.append(x)
        dfs(idx + 1, current_set, next_products)
        current_set.pop()
        
    dfs(0, [], {1})
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    print(construct())
