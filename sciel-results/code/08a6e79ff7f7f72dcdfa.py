import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 256
    squares = {k*k for k in range(2, int(math.isqrt(2*N)) + 2)}
    
    # Filter diagonal bad
    valid_nodes = []
    for i in range(1, N + 1):
        if 2 * i not in squares:
            valid_nodes.append(i)
            
    # Map valid nodes to indices 0..M-1 for compact representation?
    # Or just use 1-based indices and sets.
    
    # Build conflicts among valid_nodes
    # conflicts[u] = set of v such that u+v is square
    conflicts = {u: set() for u in valid_nodes}
    for u in valid_nodes:
        for s in squares:
            v = s - u
            if v in conflicts: # v is valid
                if u != v: # u!=v is guaranteed by diagonal filter? No, if 2u=s, u is filtered. So u!=v always in valid set.
                    conflicts[u].add(v)
                    conflicts[v].add(u) # Symmetric

    M = len(valid_nodes)
    if M == 0:
        return [1] # Fallback, though 1+1=2 not square, so 1 is valid.

    # Greedy Initial Solution
    # Sort by degree descending?
    degs = {u: len(conflicts[u]) for u in valid_nodes}
    sorted_nodes = sorted(valid_nodes, key=lambda x: degs[x]) # Ascending degree (low degree first for greedy IS)
    
    current_set = set()
    for u in sorted_nodes:
        if all(v not in current_set for v in conflicts[u]):
            current_set.add(u)
            
    best_set = list(current_set)
    best_size = len(best_set)
    
    # Branch and Bound
    # We work on the list of valid_nodes
    # State: remaining nodes (list), current chosen (set)
    
    def greedy_matching_size(nodes):
        # Returns size of a greedy matching in the induced subgraph of 'nodes'
        # nodes is a list of node ids
        node_set = set(nodes)
        used = set()
        match_count = 0
        # Sort by degree in induced subgraph?
        # For speed, just iterate
        # This is O(E) roughly
        for u in nodes:
            if u in used:
                continue
            # Find a neighbor not used
            # We need to check neighbors in the induced subgraph
            # conflicts[u] contains all neighbors. Filter by node_set and not used.
            # To speed up, we can just pick the first one found.
            for v in conflicts[u]:
                if v in node_set and v not in used:
                    match_count += 1
                    used.add(u)
                    used.add(v)
                    break
        return match_count

    def solve_bnb(nodes, current):
        nonlocal best_set, best_size, deadline, start_time
        
        if time.time() > deadline:
            return
            
        if not nodes:
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
            return
            
        # Upper Bound
        # UB = len(current) + len(nodes) - greedy_matching_size(nodes)
        # This calculation is expensive.
        # Optimization: Only calculate bound if len(current) + len(nodes) > best_size
        if len(current) + len(nodes) <= best_size:
            return
            
        # Calculate matching bound
        # To save time, we can use a simpler bound first: len(nodes)
        # If len(current) + len(nodes) <= best_size, prune. (Already done)
        # Let's try the matching bound.
        # If M is small, it's fast.
        # For M=256, it might be slow.
        # Let's use a heuristic: if len(nodes) < 50, use matching. Else use len(nodes).
        # Actually, let's just use len(nodes) as UB to ensure speed, and rely on good branching.
        # The matching bound is better. Let's try it but with a time check inside?
        # No, let's just use len(nodes) for the bound in the first few levels, or always?
        # The prompt asks for the Lovasz/Matching bound. I will implement it.
        
        # To make matching fast, we can cache? No, subgraphs are different.
        
        # Let's try the bound.
        m_size = greedy_matching_size(nodes)
        ub = len(current) + len(nodes) - m_size
        
        if ub <= best_size:
            return
            
        # Branching Variable Selection
        # Pick node with highest degree in induced subgraph
        # degrees = {u: len([v for v in conflicts[u] if v in set(nodes)]) for u in nodes}
        # This is also expensive.
        # Heuristic: Pick the first node? Or random?
        # Let's pick the node with the most neighbors in 'nodes'.
        # Approximation: Use global degree as proxy?
        # Or just pick nodes[0].
        
        # Let's use a simple heuristic: pick the node with max global degree among remaining.
        # Precompute global degrees.
        # global_deg = {u: len(conflicts[u]) for u in valid_nodes}
        
        # For now, pick nodes[0] to keep code simple and fast.
        # Better: Sort nodes by degree once? No, induced degree changes.
        
        # Let's just pick a random node to avoid bias?
        # Or the one with max degree in the *original* graph.
        
        # Let's stick to nodes[0] for now.
        
        v = nodes[0]
        remaining = nodes[1:]
        
        # Branch 1: Include v
        # New nodes: remaining minus neighbors of v
        # neighbors_in_remaining = [u for u in remaining if u in conflicts[v]]
        # new_nodes = [u for u in remaining if u not in conflicts[v]]
        
        # Optimization: Use sets for fast intersection?
        # conflicts[v] is a set.
        # new_nodes = [u for u in remaining if u not in conflicts[v]]
        
        new_current = current | {v}
        
        # Recurse Include
        solve_bnb(new_nodes, new_current)
        
        # Branch 2: Exclude v
        # new_nodes = remaining
        solve_bnb(remaining, current)

    # Run B&B
    # Initial nodes: all valid_nodes
    # To improve B&B, sort initial nodes by degree descending?
    # The branching picks nodes[0]. If we sort by degree descending, we branch on high degree nodes first.
    initial_nodes = sorted(valid_nodes, key=lambda x: len(conflicts[x]), reverse=True)
    
    solve_bnb(initial_nodes, set())
    
    # Ensure non-empty
    if not best_set:
        # Fallback to greedy
        best_set = list(current_set)
        
    return best_set
