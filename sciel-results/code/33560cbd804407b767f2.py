import time
import random
import sys

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Precompute neighbors for all vertices? 
    # 512 * 9 = 4608 neighbors. Very small.
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    best_path = []
    best_len = 0
    
    # Heuristic: We will use backtracking.
    # To avoid deep recursion stack limits, we can increase recursion limit or use iterative.
    # Depth 190 is fine for recursion in Python if limit is raised.
    sys.setrecursionlimit(5000)
    
    def backtrack(path, used, block_count):
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        cur = path[-1]
        
        # Generate candidates
        candidates = []
        for n in neighbors[cur]:
            if n not in used and block_count[n] == 1:
                candidates.append(n)
                
        if not candidates:
            return
            
        # Heuristic: Sort candidates to explore promising branches first.
        # A good heuristic is to prefer neighbors that have fewer future options (Warnsdorff).
        # We can approximate this by looking at the number of valid neighbors of n.
        # But calculating that is O(9) per candidate.
        # Let's just shuffle or use a static order with random restarts.
        # Random restarts with random bit orderings is often effective.
        
        random.shuffle(candidates)
        
        for n in candidates:
            if time.time() > deadline:
                return
            
            # Add n
            path.append(n)
            used.add(n)
            
            # Update block counts for neighbors of n
            for nn in neighbors[n]:
                block_count[nn] += 1
                
            backtrack(path, used, block_count)
            
            # Remove n
            path.pop()
            used.remove(n)
            
            for nn in neighbors[n]:
                block_count[nn] -= 1
                
            if time.time() > deadline:
                return

    # We will run multiple restarts with different random seeds/orders if possible.
    # However, the backtracking itself explores the space.
    # To ensure we find a good solution, we can bias the initial steps.
    
    # Strategy: 
    # 1. Run one main backtracking search.
    # 2. If it finishes or times out, return best.
    # Given the speed of O(1) checks, it might run fast.
    # If it runs too fast and doesn't find the max, we can restart with a different random seed for shuffling.
    
    # Let's try a few restarts if the first one doesn't find a very long snake quickly.
    # But backtracking is depth-first. It might get stuck in a local area.
    # Randomizing the candidate order helps.
    
    # We can also try to force a specific structure in the first few steps to guide the search.
    # For now, let's just run the backtracker.
    
    # To improve chances, we can try a few different random seeds for the shuffle.
    # But the shuffle is inside the recursion.
    
    # Let's just run it. The O(1) check is the key.
    
    path = [0]
    used = {0}
    block_count = [0] * MAX_V
    
    # Initialize block counts for start vertex 0
    for n in neighbors[0]:
        block_count[n] += 1
        
    backtrack(path, used, block_count)
    
    # If the backtracker didn't find anything (shouldn't happen, at least length 1), 
    # or if it finished quickly and we have time, we could try more restarts.
    # But usually, one deep search with good heuristics (or randomization) is enough for N=9.
    
    # Ensure we return a valid path
    if not best_path:
        return [0]
        
    return best_path
