import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    node_set = set(valid_nodes)
    
    # Build adjacency list
    # adj[v] = list of u in valid_nodes such that u != v and u+v in squares
    adj = {v: [] for v in valid_nodes}
    for i, v in enumerate(valid_nodes):
        for u in valid_nodes[i+1:]:
            if v + u in squares:
                adj[v].append(u)
                adj[u].append(v)
                
    best_set = set()
    best_size = 0
    
    # Multiple restarts with random tie-breaking
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Copy degrees for dynamic update? 
        # For 256 nodes, recalculating degree each step is O(N^2) per step, O(N^3) total.
        # 256^3 ~ 16 million ops. Very fast in Python if optimized, or even with lists.
        # But let's just use a simple greedy with random shuffle of candidates with min degree.
        
        current_nodes = set(valid_nodes)
        s = set()
        
        # Precompute static degrees for initial sort? 
        # Or just pick min degree dynamically.
        # Dynamic min degree is better.
        
        while current_nodes:
            # Find node with minimum degree in current subgraph
            min_deg = float('inf')
            candidates = []
            for v in current_nodes:
                deg = sum(1 for u in adj[v] if u in current_nodes)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            # Randomly pick from candidates
            v = random.choice(candidates)
            s.add(v)
            
            # Remove v and its neighbors from current_nodes
            to_remove = {v}
            for u in adj[v]:
                if u in current_nodes:
                    to_remove.add(u)
            
            current_nodes -= to_remove
            
        # Post-processing: Try to add back any nodes from valid_nodes \ s
        # This is a simple "maximal" check, but greedy already produces a maximal independent set?
        # Yes, greedy MIS produces a maximal independent set. 
        # Wait, if we remove neighbors, we might have removed a node that had no conflict with the *rest* of S?
        # No, if u was neighbor of v, and v is in S, u cannot be in S.
        # So the set is maximal.
        
        # However, we can try to "swap".
        # For each u not in s:
        #   Let C = {v in s | v+u in squares}
        #   If len(C) < 1: add u (shouldn't happen if maximal, unless we made a mistake)
        #   Actually, if the set is maximal, no u can be added.
        
        # To improve, we can try a different greedy strategy:
        # "Largest First" or "Smallest First" usually works well.
        # Let's try one restart with "Largest Degree First" (Maximal Independent Set via Max Degree)
        # and one with "Smallest Degree First".
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # One more specific strategy: 
    # Sort by degree (static) ascending, pick, remove neighbors.
    # This is deterministic.
    
    # Let's also try a local search on the best_set found
    # Try to remove a node from S and add multiple nodes?
    # If we remove v, we free up N(v).
    # Can we add all of N(v)? No, they might conflict with each other or other S nodes.
    # But we can add any subset of N(v) that is independent and doesn't conflict with S \ {v}.
    
    # Given time constraints, the randomized greedy is likely sufficient to beat 86.
    # 86 is from mod 3. 
    # Let's check if we can just return the mod 3 set if it's better?
    # Mod 3 set size is 86.
    # If best_size < 86, we should fallback to mod 3?
    # No, mod 3 is valid. 
    # We should ensure we return the best of (mod 3, greedy results).
    
    # Mod 3 set
    mod3_set = set(v for v in range(1, N+1) if v % 3 == 1)
    # Verify mod3_set validity
    # a,b in mod3_set => a+b == 2 mod 3. Squares are 0,1 mod 3. Valid.
    # Size is 86.
    
    if len(mod3_set) > best_size:
        best_set = mod3_set
        best_size = len(mod3_set)
        
    return list(best_set)
