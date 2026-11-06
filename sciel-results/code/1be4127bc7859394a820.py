import time

def construct():
    # Target sum
    T = 200
    N = 100
    
    # Bitset representation of reachable sums
    # Bit i is 1 if sum i is reachable
    # We care about sums up to T
    
    best_set = []
    best_size = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = 50.0  # Leave 10s margin
    
    # Elements to consider
    elements = list(range(N, 0, -1))
    
    # DFS Branch and Bound
    # state: index in elements, current reachable sums (int), current selected list
    def dfs(idx, reachable, selected):
        nonlocal best_set, best_size, deadline
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: if current size + remaining elements <= best_size, prune
        if len(selected) + (N - idx) <= best_size:
            return
            
        # If we've processed all elements
        if idx == N:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
        
        x = elements[idx]
        
        # Option 1: Include x
        # Check if adding x creates sum T
        # new_reachable = reachable | (reachable << x)
        # We only care if bit T is set in new_reachable
        # But we also need to update reachable for further steps
        
        # Optimization: Only keep bits up to T
        mask = (1 << (T + 1)) - 1
        
        new_reachable = (reachable | (reachable << x)) & mask
        
        if not (new_reachable & (1 << T)):
            # Valid to include
            selected.append(x)
            dfs(idx + 1, new_reachable, selected)
            selected.pop()
            
        # Option 2: Exclude x
        # Only explore this if we haven't found a solution with size N (impossible) 
        # or if including x didn't lead to a better solution?
        # Actually, we always explore exclude to see if we can get a larger set
        # by skipping a "bad" element and taking more "good" ones later?
        # Since we go 100->1, excluding 100 allows us to potentially take 99, 98...
        # But 99, 98 are "smaller" (in value) but "larger" in index?
        # No, index 0 is 100, index 1 is 99.
        # If we exclude 100, we go to idx 1 (99).
        # This is standard B&B.
        
        dfs(idx + 1, reachable, selected)

    dfs(0, 1, [])
    
    # If B&B timed out, best_set might be empty or suboptimal.
    # Fallback to simple greedy if best_size is 0 or very low
    if best_size < 1:
        # Simple greedy fallback
        selected, reachable = [], 1
        for x in range(100, 0, -1):
            new = (reachable | (reachable << x)) & ((1 << 201) - 1)
            if not (new & (1 << 200)):
                selected.append(x)
                reachable = new
        best_set = selected
        best_size = len(selected)

    return best_set
