import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 580

    # 1. Compute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # 2. Build adjacency list
    # adj[i] is the set of vertices j such that (i-j) % m is in forbidden
    adj = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            adj[i].add(j1)
            adj[i].add(j2)
        # Note: adj is symmetric. We don't need to add both directions in the loop if we do it for all i,
        # but doing it for all i ensures adj[i] contains all neighbors.
        
    # 3. Greedy heuristic to find a good initial solution
    # Try several random orderings
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # Deterministic greedy
    selected = set()
    for v in range(m):
        if all(v not in adj[u] for u in selected):
            selected.add(v)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = list(selected)

    # Randomized greedy
    for _ in range(100):
        order = list(range(m))
        rng.shuffle(order)
        selected = set()
        for v in order:
            if all(v not in adj[u] for u in selected):
                selected.add(v)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            if best_size > 20: # Unlikely, but safe check
                break

    # 4. Exact Branch and Bound
    # We want to find if there is an independent set larger than best_size
    
    # Precompute neighbors as lists for faster access? Sets are fine for N=205.
    
    def branch_and_bound(candidates, selected_size):
        nonlocal best_size, best_set
        
        # Upper bound: selected_size + len(candidates)
        if selected_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if selected_size > best_size:
                best_size = selected_size
                # We need to reconstruct the set. 
                # To avoid passing the set around, we can just store the size 
                # and then do a second pass or store the path.
                # For efficiency, let's pass the set.
            return

        # Heuristic: Pick a vertex to branch on.
        # Choosing a vertex with high degree in the candidate subgraph 
        # tends to reduce the search space more in the "include" branch.
        
        # Find vertex with max degree in candidates
        max_deg = -1
        pivot = None
        for v in candidates:
            deg = len(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return

        # Branch 1: Include pivot
        # New candidates: candidates minus pivot and its neighbors
        new_candidates = candidates - {pivot} - (adj[pivot] & candidates)
        branch_and_bound(new_candidates, selected_size + 1)
        
        # Branch 2: Exclude pivot
        new_candidates = candidates - {pivot}
        branch_and_bound(new_candidates, selected_size)

    # To get the actual set, we need to modify the function to track the set
    # Or, since we just need the size for the score, and then return a valid set,
    # we can just run the search to find the max size, then run a greedy 
    # constrained to find a set of that size? 
    # Actually, the task requires returning the set.
    
    # Let's modify branch_and_bound to store the best set.
    
    def search(candidates, current_set):
        nonlocal best_size, best_set
        
        if len(current_set) + len(candidates) <= best_size:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Pivot selection
        max_deg = -1
        pivot = None
        for v in candidates:
            deg = len(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        if pivot is None:
            return

        # Include pivot
        new_candidates = candidates - {pivot} - (adj[pivot] & candidates)
        new_set = set(current_set)
        new_set.add(pivot)
        search(new_candidates, new_set)
        
        # Exclude pivot
        new_candidates = candidates - {pivot}
        search(new_candidates, current_set)

    # Initial call
    # To speed up, we can order vertices or use the greedy result as a starting point?
    # The B&B will explore the space.
    # We can limit the time.
    
    # Optimization: Use a greedy coloring bound for tighter pruning.
    # For now, let's try the simple bound. If it's too slow, we'll see.
    # With N=205, and dense graph, it should be fast.
    
    # One issue: Python recursion limit. 205 is fine.
    
    # Let's add a time check inside the loop if it's too deep, but recursion makes it hard.
    # Given the small N, it should finish quickly.
    
    # Start search
    # We can start with all vertices as candidates.
    # To improve pruning, we can sort candidates?
    
    # Let's run the search
    search(set(range(m)), set())
    
    # Safety: if search didn't improve, return the greedy best
    if best_size == 0:
         best_set = list(range(m)) # Fallback, shouldn't happen

    return np.array(best_set, dtype=np.int64)
